import argparse
import json
import os
import re
import time
import pickle as pkl

import dgl
import numpy as np
import torch


def _extract_file_id(file_name):
    match = re.search(r"_(\d+)\.txt$", file_name)
    if match is None:
        return None
    return int(match.group(1))


def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _zscore_np(arr):
    if arr.shape[0] == 0:
        return arr
    mean = arr.mean(axis=0, keepdims=True)
    std = arr.std(axis=0, keepdims=True)
    std = np.maximum(std, 1e-6)
    return (arr - mean) / std


def _list_input_files(benign_dir, attack_dir, max_benign=0, max_attack=0):
    benign_files = []
    attack_files = []

    for file_name in os.listdir(benign_dir):
        if not file_name.startswith("clearscope-benign_") or not file_name.endswith(".txt"):
            continue
        file_id = _extract_file_id(file_name)
        if file_id is None:
            continue
        benign_files.append((file_id, os.path.join(benign_dir, file_name)))
    benign_files.sort(key=lambda x: x[0])

    for file_name in os.listdir(attack_dir):
        if not file_name.startswith("clearscope-e3-attack_") or not file_name.endswith(".txt"):
            continue
        file_id = _extract_file_id(file_name)
        if file_id is None:
            continue
        attack_files.append((file_id, os.path.join(attack_dir, file_name)))
    attack_files.sort(key=lambda x: x[0])

    if max_benign > 0:
        benign_files = benign_files[:max_benign]
    if max_attack > 0:
        attack_files = attack_files[:max_attack]
    return benign_files, attack_files


def _parse_clearscope_file(file_path, graph_label):
    nodes = {}
    node_features = []
    node_labels = []
    edge_feat_sum = {}
    edge_feat_cnt = {}

    def get_or_create(node_key):
        if node_key in nodes:
            return nodes[node_key]
        idx = len(nodes)
        nodes[node_key] = idx
        node_features.append([])
        node_labels.append([])
        return idx

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            actor_id, event_type, obj_token = parts[0], parts[1], parts[2]
            token_parts = obj_token.split(":")
            if len(token_parts) != 4:
                continue
            obj_id, obj_type, act_type, timestamp = token_parts
            src = get_or_create(actor_id)
            dst = get_or_create(obj_id)
            feat = np.array(
                [
                    _safe_float(event_type),
                    _safe_float(obj_type),
                    _safe_float(act_type),
                    _safe_float(timestamp),
                ],
                dtype=np.float32,
            )

            for u, v in ((src, dst), (dst, src)):
                k = (u, v)
                if k not in edge_feat_sum:
                    edge_feat_sum[k] = feat.copy()
                    edge_feat_cnt[k] = 1
                else:
                    edge_feat_sum[k] += feat
                    edge_feat_cnt[k] += 1

            node_features[src].append(feat)
            node_features[dst].append(feat)
            node_labels[src].append(graph_label)
            node_labels[dst].append(graph_label)

    num_nodes = len(nodes)
    if num_nodes == 0:
        graph = dgl.graph((torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64)), num_nodes=0)
        graph.ndata["attr"] = torch.zeros((0, 4), dtype=torch.float32)
        graph.edata["attr"] = torch.zeros((0, 4), dtype=torch.float32)
        graph.ndata["type"] = torch.zeros((0,), dtype=torch.int64)
        graph.edata["type"] = torch.zeros((0,), dtype=torch.int64)
        return graph, []

    x = []
    y = []
    for i in range(num_nodes):
        if len(node_features[i]) > 0:
            x.append(np.mean(np.stack(node_features[i], axis=0), axis=0))
        else:
            x.append(np.zeros(4, dtype=np.float32))
        if len(node_labels[i]) > 0:
            y.append(int(np.bincount(np.asarray(node_labels[i], dtype=np.int64)).argmax()))
        else:
            y.append(0)

    x = _zscore_np(np.stack(x, axis=0).astype(np.float32))
    edge_pairs = list(edge_feat_sum.keys())
    edge_src = [u for u, _ in edge_pairs]
    edge_dst = [v for _, v in edge_pairs]
    if len(edge_pairs) == 0:
        edge_attr = np.zeros((0, 4), dtype=np.float32)
    else:
        edge_attr = np.stack(
            [edge_feat_sum[k] / max(edge_feat_cnt[k], 1) for k in edge_pairs],
            axis=0,
        ).astype(np.float32)
        edge_attr = _zscore_np(edge_attr)

    graph = dgl.graph(
        (
            torch.tensor(edge_src, dtype=torch.int64),
            torch.tensor(edge_dst, dtype=torch.int64),
        ),
        num_nodes=num_nodes,
    )
    graph.ndata["attr"] = torch.tensor(x, dtype=torch.float32)
    graph.edata["attr"] = torch.tensor(edge_attr, dtype=torch.float32)
    graph.ndata["type"] = torch.zeros((num_nodes,), dtype=torch.int64)
    graph.edata["type"] = torch.zeros((graph.num_edges(),), dtype=torch.int64)
    return graph, y


def _limit_graph_edges(graph, max_edges_per_graph):
    if max_edges_per_graph <= 0:
        return graph
    if graph.num_edges() <= max_edges_per_graph:
        return graph

    # Keep a deterministic subset of edges to bound memory footprint.
    keep_eids = torch.arange(max_edges_per_graph, dtype=torch.int64)
    src, dst = graph.find_edges(keep_eids)
    new_graph = dgl.graph((src, dst), num_nodes=graph.num_nodes())
    new_graph.ndata["attr"] = graph.ndata["attr"]
    new_graph.ndata["type"] = graph.ndata["type"]
    new_graph.edata["attr"] = graph.edata["attr"][keep_eids]
    new_graph.edata["type"] = graph.edata["type"][keep_eids]
    return new_graph


def build_clearscope_preprocessed(
    data_dir="./data/clearscope",
    benign_train_ratio=0.8,
    max_benign=0,
    max_attack=0,
    max_edges_per_graph=300000,
    force=False,
):
    metadata_path = os.path.join(data_dir, "metadata.json")
    if os.path.exists(metadata_path) and not force:
        return

    benign_dir = os.path.join(data_dir, "benign")
    attack_dir = os.path.join(data_dir, "attack")
    benign_files, attack_files = _list_input_files(
        benign_dir=benign_dir,
        attack_dir=attack_dir,
        max_benign=max_benign,
        max_attack=max_attack,
    )
    benign_train_ratio = min(max(float(benign_train_ratio), 0.1), 0.95)
    benign_split = int(len(benign_files) * benign_train_ratio)
    train_benign_files = benign_files[:benign_split]
    test_benign_files = benign_files[benign_split:]

    print(
        f"[clearscope-parser] start: benign={len(benign_files)} attack={len(attack_files)} "
        f"train_ratio={benign_train_ratio:.2f}"
    )

    train_graphs = []
    for i, (_, file_path) in enumerate(train_benign_files):
        t0 = time.time()
        g, _ = _parse_clearscope_file(file_path, graph_label=0)
        g = _limit_graph_edges(g, max_edges_per_graph=max_edges_per_graph)
        train_graphs.append(g)
        print(
            f"[clearscope-parser] train {i + 1}/{len(train_benign_files)} "
            f"nodes={g.num_nodes()} edges={g.num_edges()} file={os.path.basename(file_path)} "
            f"time={time.time() - t0:.2f}s"
        )

    test_graphs_with_labels = []
    for i, (_, file_path) in enumerate(test_benign_files):
        t0 = time.time()
        g, labels = _parse_clearscope_file(file_path, graph_label=0)
        g = _limit_graph_edges(g, max_edges_per_graph=max_edges_per_graph)
        test_graphs_with_labels.append((g, labels))
        print(
            f"[clearscope-parser] test-benign {i + 1}/{len(test_benign_files)} "
            f"nodes={g.num_nodes()} edges={g.num_edges()} file={os.path.basename(file_path)} "
            f"time={time.time() - t0:.2f}s"
        )
    for i, (_, file_path) in enumerate(attack_files):
        t0 = time.time()
        g, labels = _parse_clearscope_file(file_path, graph_label=1)
        g = _limit_graph_edges(g, max_edges_per_graph=max_edges_per_graph)
        test_graphs_with_labels.append((g, labels))
        print(
            f"[clearscope-parser] test-attack {i + 1}/{len(attack_files)} "
            f"nodes={g.num_nodes()} edges={g.num_edges()} file={os.path.basename(file_path)} "
            f"time={time.time() - t0:.2f}s"
        )

    malicious = []
    offset = 0
    test_graphs = []
    for g, labels in test_graphs_with_labels:
        test_graphs.append(g)
        for i, label in enumerate(labels):
            if label == 1:
                malicious.append(offset + i)
        offset += g.num_nodes()

    metadata = {
        "node_feature_dim": 4,
        "edge_feature_dim": 4,
        "malicious": [malicious, []],
        "n_train": len(train_graphs),
        "n_test": len(test_graphs),
        "exclude_train_nodes_from_test": False,
        "parser": "utils/clearscope_parser.py",
        "max_edges_per_graph": max_edges_per_graph,
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f)
    for i, g in enumerate(train_graphs):
        with open(os.path.join(data_dir, f"train{i}.pkl"), "wb") as f:
            pkl.dump(g, f)
    for i, g in enumerate(test_graphs):
        with open(os.path.join(data_dir, f"test{i}.pkl"), "wb") as f:
            pkl.dump(g, f)

    print(
        f"[clearscope-parser] done: n_train={len(train_graphs)} "
        f"n_test={len(test_graphs)} malicious_nodes={len(malicious)}"
    )


def _default_data_dir():
    utils_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(utils_dir), "data", "clearscope")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clearscope parser")
    parser.add_argument("--data_dir", type=str, default=_default_data_dir())
    parser.add_argument("--train_benign_ratio", type=float, default=0.8)
    parser.add_argument("--max_benign", type=int, default=0)
    parser.add_argument("--max_attack", type=int, default=0)
    parser.add_argument("--max_edges_per_graph", type=int, default=300000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    build_clearscope_preprocessed(
        data_dir=args.data_dir,
        benign_train_ratio=args.train_benign_ratio,
        max_benign=args.max_benign,
        max_attack=args.max_attack,
        max_edges_per_graph=args.max_edges_per_graph,
        force=args.force,
    )
