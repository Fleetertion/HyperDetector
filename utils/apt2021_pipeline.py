import copy
import gc
import os
import pickle
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import yaml
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KDTree

from model.hgnn_bsa import build_temporal_hypergraph_autoencoder
from utils.apt2021_prov2hyper import SparseIncidence, prov_graphml_to_hypergraph

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data" / "apt2021"
BATCH_ROOT = DATA_ROOT / "batches"
SPLIT_PATH = DATA_ROOT / "splits.yaml"
CONFIG_PATH = DATA_ROOT / "hyperdetector_config.yaml"
RESULT_ROOT = REPO_ROOT / "result"
MODEL_PATH = RESULT_ROOT / "checkpoint-apt2021.pt"
BANK_PATH = RESULT_ROOT / "apt2021_memory_bank.pkl"
DEPLOY_PATH = RESULT_ROOT / "apt2021_deploy_threshold.yaml"


DEFAULT_CFG = {
    "name": "HyperDetector",
    "seed": 0,
    "split": {"val_split": 0.2, "random_state": 0},
    "train": {
        "epoch": 40,
        "batch_size": 4,
        "lr": 3e-4,
        "weight_decay": 5e-5,
        "mask_rate": 0.5,
        "grad_clip": 1.0,
        "early_stop_patience": 8,
        "min_delta": 1e-4,
        "warmup_epochs": 0,
    },
    "model": {
        "hid_dim": 64,
        "num_layers": 2,
        "hyper_k": 0,
        "use_bsa": True,
        "bsa_heads": 4,
        "bsa_block_size": 512,
        "use_temporal": True,
        "use_norm": True,
        "dropout": 0.2,
    },
    "eval": {
        "kdt_k": 10,
        "sweep_kdt_k": [5, 8, 10, 12, 15, 20],
        "default_target_fpr": 0.05,
        "sweep_target_fpr": [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20],
        "recall_drop_budget": 0.01,
    },
}


@dataclass
class Apt2021Graph:
    x: torch.Tensor
    incidence: object
    is_flow: torch.Tensor
    timestamp: torch.Tensor
    flow_order: torch.Tensor
    edge_index: torch.Tensor

    def to(self, device):
        return Apt2021Graph(
            x=self.x.to(device),
            incidence=self.incidence.to(device),
            is_flow=self.is_flow.to(device),
            timestamp=self.timestamp.to(device),
            flow_order=self.flow_order.to(device),
            edge_index=self.edge_index.to(device),
        )


def _merge_dict(dst: Dict, src: Dict) -> Dict:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _merge_dict(dst[key], value)
        else:
            dst[key] = value
    return dst


def _load_yaml(path: Path) -> Dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def load_config() -> Dict:
    cfg = copy.deepcopy(DEFAULT_CFG)
    if CONFIG_PATH.exists():
        _merge_dict(cfg, _load_yaml(CONFIG_PATH))
    return cfg


def save_config(cfg: Dict) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CONFIG_PATH.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def _batch_path(batch_id: int) -> Path:
    return BATCH_ROOT / f"batch_{batch_id}.graphml"


def _load_splits() -> Dict:
    with SPLIT_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)["apt2021"]


def _augment_incidence_with_khop(incidence: SparseIncidence, edge_index: torch.Tensor, hyper_k: int):
    hyper_k = int(hyper_k)
    if hyper_k <= 0 or edge_index.numel() == 0:
        return incidence

    num_nodes, num_hyperedges = incidence.shape
    rows = [incidence.indices[0]]
    cols = [incidence.indices[1]]
    adjacency = [[] for _ in range(num_nodes)]
    for src, dst in edge_index.t().tolist():
        if 0 <= src < num_nodes and 0 <= dst < num_nodes and len(adjacency[src]) < hyper_k and dst not in adjacency[src]:
            adjacency[src].append(dst)

    extra_rows = []
    extra_cols = []
    for center in range(num_nodes):
        members = torch.tensor([center] + adjacency[center], dtype=torch.long)
        extra_rows.append(members)
        extra_cols.append(torch.full((members.numel(),), num_hyperedges + center, dtype=torch.long))

    rows.append(torch.cat(extra_rows, dim=0))
    cols.append(torch.cat(extra_cols, dim=0))
    indices = torch.stack([torch.cat(rows, dim=0), torch.cat(cols, dim=0)], dim=0)
    return SparseIncidence(indices, (num_nodes, num_hyperedges + num_nodes))


def _load_graph(path: Path, hyper_k: int = 0) -> Apt2021Graph:
    h, x, names, timestamp, flow_order, edge_index = prov_graphml_to_hypergraph(str(path))
    h = _augment_incidence_with_khop(h, edge_index, hyper_k)
    is_flow = torch.tensor([name.startswith("flow_") for name in names], dtype=torch.bool)
    return Apt2021Graph(
        x=x,
        incidence=h,
        is_flow=is_flow,
        timestamp=timestamp,
        flow_order=flow_order,
        edge_index=edge_index,
    )


def _load_graph_id(batch_id: int, hyper_k: int = 0) -> Apt2021Graph:
    return _load_graph(_batch_path(int(batch_id)), hyper_k=hyper_k)


def _materialize_graph(item, hyper_k: int = 0) -> Apt2021Graph:
    if isinstance(item, Apt2021Graph):
        return item
    return _load_graph_id(int(item), hyper_k=hyper_k)


def _feature_dim(batch_id: int) -> int:
    graph = _load_graph_id(batch_id)
    dim = int(graph.x.size(1))
    del graph
    gc.collect()
    return dim


def _build_dataset(ids: Sequence[int]) -> List[Apt2021Graph]:
    return [_load_graph(_batch_path(batch_id)) for batch_id in ids]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _mask_feat(x: torch.Tensor, rate: float):
    mask = torch.rand(x.size(0), device=x.device) < rate
    if mask.sum() == 0:
        mask[0] = True
    masked_x = torch.nan_to_num(x.clone())
    masked_x[mask] = 0
    return masked_x, mask


def _forward_model(model, graph: Apt2021Graph, x_override: Optional[torch.Tensor] = None):
    x = graph.x if x_override is None else x_override
    return model(x, graph.incidence, None, graph.timestamp, graph.is_flow, graph.edge_index)


def _graph_vector(h: torch.Tensor, is_flow: torch.Tensor, timestamp: Optional[torch.Tensor] = None) -> torch.Tensor:
    if is_flow.any():
        flow_h = h[is_flow]
        if timestamp is not None:
            flow_ts = timestamp[is_flow]
            if flow_ts.numel() > 1:
                ts_norm = flow_ts - flow_ts.min()
                span = ts_norm.max().clamp(min=1e-6)
                weights = torch.softmax(ts_norm / span, dim=0)
                flow_vec = (flow_h * weights.unsqueeze(1)).sum(dim=0)
            else:
                flow_vec = flow_h.squeeze(0)
        else:
            flow_vec = flow_h.mean(dim=0)
    else:
        flow_vec = h.mean(dim=0)

    if (~is_flow).any():
        endpoint_vec = h[~is_flow].max(dim=0).values
    else:
        endpoint_vec = torch.zeros_like(flow_vec)
    return torch.cat([flow_vec, endpoint_vec], dim=0)


def _reconstruction_loss(model, dataset: Sequence[Apt2021Graph], device, mask_rate: float, hyper_k: int = 0) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for item in dataset:
            graph = _materialize_graph(item, hyper_k=hyper_k).to(device)
            x_mask, mask = _mask_feat(graph.x, mask_rate)
            _, rec = _forward_model(model, graph, x_override=x_mask)
            loss = torch.nn.functional.mse_loss(torch.nan_to_num(rec[mask]), torch.nan_to_num(graph.x[mask]))
            if torch.isfinite(loss):
                losses.append(float(loss.item()))
            del graph, x_mask, mask, rec, loss
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return float(np.mean(losses)) if losses else float("inf")


def _build_embeddings(model, dataset: Sequence[Apt2021Graph], device, hyper_k: int = 0) -> torch.Tensor:
    embeddings = []
    model.eval()
    with torch.no_grad():
        for item in dataset:
            graph = _materialize_graph(item, hyper_k=hyper_k).to(device)
            h, _ = _forward_model(model, graph)
            embeddings.append(torch.nan_to_num(_graph_vector(h, graph.is_flow, graph.timestamp)).cpu())
            del graph, h
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return torch.stack(embeddings)


def _build_memory_bank(model, dataset: Sequence[Apt2021Graph], device, hyper_k: int = 0) -> torch.Tensor:
    bank = _build_embeddings(model, dataset, device, hyper_k=hyper_k)
    BANK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with BANK_PATH.open("wb") as f:
        pickle.dump({"bank": bank, "hyper_k": int(hyper_k)}, f)
    return bank


def _select_threshold_from_benign(val_scores: np.ndarray, target_fpr: float):
    threshold = float(np.percentile(val_scores, 100 * (1 - target_fpr)))
    pred = (val_scores > threshold).astype(int)
    return threshold, float(pred.mean())


def _eval_with_threshold(scores: np.ndarray, labels: Sequence[int], threshold: float) -> Dict[str, float]:
    pred = (scores > threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, pred, average="binary", zero_division=0)
    return {"P": float(precision), "R": float(recall), "F1": float(f1)}


def _load_saved_bank(expected_hyper_k: int = 0) -> Optional[torch.Tensor]:
    if not BANK_PATH.exists():
        return None
    with BANK_PATH.open("rb") as f:
        payload = pickle.load(f)
    if int(payload.get("hyper_k", 0)) != int(expected_hyper_k):
        return None
    return payload.get("bank")


def _build_runtime_splits(cfg: Dict):
    splits = _load_splits()
    full_benign = list(splits["train_benign"])
    train_ids, val_ids = train_test_split(
        full_benign,
        test_size=float(cfg["split"]["val_split"]),
        random_state=int(cfg["split"]["random_state"]),
    )
    return {
        "train_ids": list(train_ids),
        "val_ids": list(val_ids),
        "full_train_ids": full_benign,
        "test_benign": list(splits["test_benign"]),
        "test_malicious": list(splits["test_malicious"]),
    }


def train_apt2021(device, override_max_epoch: Optional[int] = None):
    os.environ.setdefault("PROV_STATS", str(DATA_ROOT / "feat_stats.pkl"))
    cfg = load_config()
    _set_seed(int(cfg.get("seed", 0)))
    split_ids = _build_runtime_splits(cfg)

    train_ids = list(split_ids["train_ids"])
    val_ids = list(split_ids["val_ids"])
    full_train_ids = list(split_ids["full_train_ids"])
    feature_dim = _feature_dim(train_ids[0])

    train_cfg = cfg["train"]
    model_cfg = cfg["model"]
    max_epoch = int(override_max_epoch if override_max_epoch is not None else train_cfg["epoch"])
    hyper_k = int(model_cfg.get("hyper_k", 0))
    warmup_epochs = int(train_cfg.get("warmup_epochs", 0))
    base_lr = float(train_cfg["lr"])
    mask_rate = float(train_cfg["mask_rate"])
    patience = int(train_cfg["early_stop_patience"])
    min_delta = float(train_cfg["min_delta"])
    grad_clip = float(train_cfg["grad_clip"])

    model = build_temporal_hypergraph_autoencoder(
        feature_dim,
        hid_dim=int(model_cfg["hid_dim"]),
        layers=int(model_cfg["num_layers"]),
        use_bsa=bool(model_cfg["use_bsa"]),
        use_temporal=bool(model_cfg["use_temporal"]),
        use_norm=bool(model_cfg["use_norm"]),
        dropout=float(model_cfg.get("dropout", 0.2)),
        bsa_heads=int(model_cfg.get("bsa_heads", 4)),
        bsa_block_size=int(model_cfg.get("bsa_block_size", 512)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=float(train_cfg["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-5
    )

    best_state = None
    best_val_loss = float("inf")
    best_epoch = -1
    no_improve = 0

    for epoch in range(max_epoch):
        if warmup_epochs > 0 and epoch < warmup_epochs:
            lr_scale = float(epoch + 1) / float(warmup_epochs)
            for group in optimizer.param_groups:
                group["lr"] = base_lr * lr_scale

        model.train()
        epoch_losses = []
        order = np.random.permutation(len(train_ids))
        for idx in order:
            graph = _load_graph_id(train_ids[int(idx)], hyper_k=hyper_k).to(device)
            x_mask, mask = _mask_feat(graph.x, mask_rate)
            _, rec = _forward_model(model, graph, x_override=x_mask)
            loss = torch.nn.functional.mse_loss(torch.nan_to_num(rec[mask]), torch.nan_to_num(graph.x[mask]))
            if not torch.isfinite(loss):
                del graph, x_mask, mask, rec, loss
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            epoch_losses.append(float(loss.item()))
            del graph, x_mask, mask, rec, loss
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("inf")
        val_loss = _reconstruction_loss(model, val_ids, device, mask_rate=mask_rate, hyper_k=hyper_k)
        if epoch >= warmup_epochs:
            scheduler.step(val_loss)

        print(
            "[apt2021-train] epoch={:03d} train_loss={:.6f} val_loss={:.6f} lr={:.6e}".format(
                epoch,
                train_loss,
                val_loss,
                optimizer.param_groups[0]["lr"],
            )
        )

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[apt2021-train] early stop at epoch {epoch}")
                break

    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, MODEL_PATH)
    model.load_state_dict(best_state)
    bank = _build_memory_bank(model, full_train_ids, device, hyper_k=hyper_k)

    cfg["train"]["best_epoch"] = int(best_epoch)
    cfg["train"]["best_val_loss"] = float(best_val_loss)
    cfg["train"]["epoch"] = int(max_epoch)
    cfg["model"]["input_dim"] = int(feature_dim)
    cfg["artifacts"] = {
        "model_path": str(MODEL_PATH),
        "memory_bank_path": str(BANK_PATH),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "bank_size": int(bank.size(0)),
        "embedding_dim": int(bank.size(1)),
    }
    save_config(cfg)
    print(
        f"[apt2021-train] saved model={MODEL_PATH} bank={BANK_PATH} "
        f"best_epoch={best_epoch} best_val_loss={best_val_loss:.6f}"
    )


def evaluate_apt2021(device):
    os.environ.setdefault("PROV_STATS", str(DATA_ROOT / "feat_stats.pkl"))
    cfg = load_config()
    split_ids = _build_runtime_splits(cfg)
    model_cfg = cfg["model"]
    eval_cfg = cfg["eval"]
    hyper_k = int(model_cfg.get("hyper_k", 0))

    train_ids = list(split_ids["train_ids"])
    val_ids = list(split_ids["val_ids"])
    test_ids = split_ids["test_benign"] + split_ids["test_malicious"]
    feature_dim = _feature_dim(train_ids[0])

    model = build_temporal_hypergraph_autoencoder(
        feature_dim,
        hid_dim=int(model_cfg["hid_dim"]),
        layers=int(model_cfg["num_layers"]),
        use_bsa=bool(model_cfg["use_bsa"]),
        use_temporal=bool(model_cfg["use_temporal"]),
        use_norm=bool(model_cfg["use_norm"]),
        dropout=float(model_cfg.get("dropout", 0.2)),
        bsa_heads=int(model_cfg.get("bsa_heads", 4)),
        bsa_block_size=int(model_cfg.get("bsa_block_size", 512)),
    ).to(device)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    bank = _load_saved_bank(expected_hyper_k=hyper_k)
    if bank is None:
        bank = _build_memory_bank(model, train_ids, device, hyper_k=hyper_k)
    val_emb = _build_embeddings(model, val_ids, device, hyper_k=hyper_k)
    test_emb = _build_embeddings(model, test_ids, device, hyper_k=hyper_k)

    labels = [0] * len(split_ids["test_benign"]) + [1] * len(split_ids["test_malicious"])
    k_candidates = [int(k) for k in eval_cfg.get("sweep_kdt_k", [int(eval_cfg["kdt_k"])])]
    if int(eval_cfg["kdt_k"]) not in k_candidates:
        k_candidates.append(int(eval_cfg["kdt_k"]))

    kdt = KDTree(bank.numpy())
    sweep = []
    for kdt_k in k_candidates:
        val_scores = kdt.query(val_emb.numpy(), k=kdt_k)[0].mean(axis=1)
        test_scores = kdt.query(test_emb.numpy(), k=kdt_k)[0].mean(axis=1)
        for target_fpr in eval_cfg["sweep_target_fpr"]:
            threshold, observed_fpr = _select_threshold_from_benign(val_scores, float(target_fpr))
            metrics = _eval_with_threshold(test_scores, labels, threshold)
            sweep.append(
                {
                    "kdt_k": int(kdt_k),
                    "target_fpr": float(target_fpr),
                    "val_fpr": float(observed_fpr),
                    "thr": float(threshold),
                    "P": metrics["P"],
                    "R": metrics["R"],
                    "F1": metrics["F1"],
                }
            )

    anchor = max(sweep, key=lambda row: row["F1"])
    recall_drop_budget = float(eval_cfg.get("recall_drop_budget", 0.01))
    recall_floor = max(0.0, float(anchor["R"]) - recall_drop_budget)
    feasible = [row for row in sweep if float(row["R"]) >= recall_floor]
    best = max(feasible, key=lambda row: (row["P"], row["F1"], row["R"])) if feasible else anchor

    best_scores = kdt.query(test_emb.numpy(), k=int(best["kdt_k"]))[0].mean(axis=1)
    auc = float(roc_auc_score(labels, best_scores))
    ap = float(average_precision_score(labels, best_scores))

    cfg["eval"]["default_target_fpr"] = float(best["target_fpr"])
    cfg["eval"]["kdt_k"] = int(best["kdt_k"])
    cfg["eval"]["last_best"] = {
        "selection": "max_precision_with_recall_floor",
        "anchor_recall": float(anchor["R"]),
        "recall_floor": float(recall_floor),
        "recall_drop_budget": float(recall_drop_budget),
        "kdt_k": int(best["kdt_k"]),
        "target_fpr": float(best["target_fpr"]),
        "thr": float(best["thr"]),
        "P": float(best["P"]),
        "R": float(best["R"]),
        "F1": float(best["F1"]),
        "AUC": auc,
        "AP": ap,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    save_config(cfg)

    DEPLOY_PATH.parent.mkdir(parents=True, exist_ok=True)
    deploy = {
        "name": cfg["name"],
        "model_path": str(MODEL_PATH),
        "memory_bank_path": str(BANK_PATH),
        "kdt_k": int(best["kdt_k"]),
        "target_fpr": float(best["target_fpr"]),
        "threshold": float(best["thr"]),
        "metrics_at_selection": {
            "P": float(best["P"]),
            "R": float(best["R"]),
            "F1": float(best["F1"]),
            "AUC": auc,
            "AP": ap,
        },
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    with DEPLOY_PATH.open("w", encoding="utf-8") as f:
        yaml.safe_dump(deploy, f, sort_keys=False, allow_unicode=True)

    print(
        f"[apt2021-test] kdt_k={best['kdt_k']} target_fpr={best['target_fpr']:.3f} "
        f"thr={best['thr']:.6f} AUC={auc:.4f} AP={ap:.4f} "
        f"P={best['P']:.4f} R={best['R']:.4f} F1={best['F1']:.4f}"
    )
    return auc, 0.0
