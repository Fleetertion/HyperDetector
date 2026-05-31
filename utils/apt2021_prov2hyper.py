from __future__ import annotations

import math
import os
import pickle
from datetime import datetime
from pathlib import Path

import networkx as nx
import numpy as np
import torch

class SparseIncidence:
    def __init__(self, indices: torch.Tensor, shape):
        self.indices = indices.long()
        self.shape = tuple(shape)

    def to(self, device):
        return SparseIncidence(self.indices.to(device), self.shape)

    def _tensor(self):
        values = torch.ones(self.indices.size(1), device=self.indices.device, dtype=torch.float32)
        return torch.sparse_coo_tensor(self.indices, values, self.shape).coalesce()

    def matmul(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sparse.mm(self._tensor(), x)

    def transpose_matmul(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sparse.mm(self._tensor().transpose(0, 1), x)


_STATS = Path(os.getenv("PROV_STATS", "data/apt2021/feat_stats.pkl"))
if _STATS.is_file():
    _stats = pickle.load(open(_STATS, "rb"))
    _NUMERIC_COLS = _stats["cols"]
    MU = np.asarray(_stats["mu"], np.float32)
    STD = np.asarray(_stats["std"], np.float32)
    STD[STD < 1e-6] = 1.0
else:
    MU = None
    STD = None
    _NUMERIC_COLS = [
        "Flow Duration",
        "Flow Bytes/s",
        "Flow Packets/s",
        "Total Fwd Packet",
        "Total Bwd packets",
        "Fwd Packet Length Mean",
        "Bwd Packet Length Mean",
    ]


def _incidence(graph: nx.MultiDiGraph):
    vertex_index = {node: i for i, node in enumerate(graph.nodes())}
    edge_index = {}
    rows = []
    cols = []
    for src, dst, attrs in graph.edges(data=True):
        event_id = attrs.get("event_id", f"{src}->{dst}")
        edge_index.setdefault(event_id, len(edge_index))
        hyperedge_id = edge_index[event_id]
        rows.extend([vertex_index[src], vertex_index[dst]])
        cols.extend([hyperedge_id, hyperedge_id])
    indices = torch.tensor([rows, cols], dtype=torch.long)
    incidence = SparseIncidence(indices, (len(vertex_index), len(edge_index)))
    return incidence, vertex_index


def _numeric_features(graph: nx.MultiDiGraph, names: list[str]) -> torch.Tensor:
    result = np.zeros((len(names), len(_NUMERIC_COLS)), dtype=np.float32)
    is_flow = np.array([graph.nodes[name].get("type") == "flow" for name in names], dtype=bool)
    for i, name in enumerate(names):
        attrs = graph.nodes[name]
        for j, col in enumerate(_NUMERIC_COLS):
            try:
                result[i, j] = float(attrs.get(col, 0.0))
            except (TypeError, ValueError):
                result[i, j] = 0.0
    if is_flow.any():
        if MU is not None:
            result[is_flow] = (result[is_flow] - MU) / STD
        else:
            mu = result[is_flow].mean(axis=0, keepdims=True)
            std = result[is_flow].std(axis=0, keepdims=True)
            std[std < 1e-6] = 1.0
            result[is_flow] = (result[is_flow] - mu) / std
    result[~is_flow] = 0.0
    result = np.clip(result, -8.0, 8.0)
    return torch.from_numpy(np.nan_to_num(result))


def _parse_timestamp(raw) -> float:
    if raw is None:
        return math.nan
    if isinstance(raw, (int, float)):
        return float(raw) if math.isfinite(raw) else math.nan
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return math.nan
        try:
            return float(text)
        except ValueError:
            try:
                normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
                return datetime.fromisoformat(normalized).timestamp()
            except ValueError:
                return math.nan
    if hasattr(raw, "timestamp"):
        try:
            return float(raw.timestamp())
        except Exception:
            return math.nan
    return math.nan


def prov_graphml_to_hypergraph(path: str):
    graph = nx.read_graphml(path)
    names = list(graph.nodes())
    incidence, vertex_index = _incidence(graph)

    edge_src = []
    edge_dst = []
    for src, dst in graph.edges():
        src_idx = vertex_index[src]
        dst_idx = vertex_index[dst]
        edge_src.extend([src_idx, dst_idx])
        edge_dst.extend([dst_idx, src_idx])
    if edge_src:
        pair_edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    else:
        pair_edge_index = torch.empty((2, 0), dtype=torch.long)

    node_types = [graph.nodes[name].get("type", "unknown") for name in names]
    type_to_id = {node_type: i for i, node_type in enumerate(sorted(set(node_types)))}
    x_type = torch.eye(len(type_to_id))[torch.tensor([type_to_id[node_type] for node_type in node_types])]
    degree = torch.bincount(incidence.indices[0], minlength=len(names)).to(torch.float32)
    degree = (degree / degree.max().clamp(min=1.0)).unsqueeze(1)
    x_numeric = _numeric_features(graph, names)
    x = torch.cat([x_type, degree, x_numeric], dim=1)

    raw_ts = np.array([_parse_timestamp(graph.nodes[name].get("Timestamp")) for name in names], dtype=np.float64)
    valid = np.isfinite(raw_ts)
    if valid.any():
        raw_ts[~valid] = raw_ts[valid].min()
    else:
        raw_ts[:] = 0.0
    timestamp = torch.from_numpy(raw_ts.astype(np.float32))

    is_flow_mask = np.array([node_type == "flow" for node_type in node_types], dtype=bool)
    flow_idx = np.flatnonzero(is_flow_mask)
    if flow_idx.size:
        flow_order = torch.from_numpy(flow_idx[np.argsort(raw_ts[flow_idx], kind="mergesort")].astype(np.int64))
    else:
        flow_order = torch.empty(0, dtype=torch.long)

    return incidence, x, names, timestamp, flow_order, pair_edge_index
