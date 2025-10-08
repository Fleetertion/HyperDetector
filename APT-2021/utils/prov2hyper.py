"""
Convert a provenance-batch GraphML → (H, X, nodelist)

H : torch_sparse.SparseTensor  (|V| × |E|)
X : torch.Tensor               (|V| × d_in)
nodelist : list[str]           original node names

Features
--------
(1) one-hot node type
(2) degree / max_deg
(3) global-z-scored numeric cols  (flow 节点) :
    ['Flow Duration', 'Flow Bytes/s', 'Flow Packets/s',
     'Total Fwd Packet', 'Total Bwd packets',
     'Fwd Packet Length Mean', 'Bwd Packet Length Mean']
    端点统一置 0，裁剪到 [-8, +8]，所有 NaN/Inf→0
"""

from __future__ import annotations
import os, pickle, numpy as np, torch, networkx as nx
from datetime import datetime
import math
from pathlib import Path
from torch_sparse import SparseTensor

# ── global μ/σ file ─────────────────────────────────────────
_STATS = Path(os.getenv("PROV_STATS", "data/feat_stats.pkl"))
if _STATS.is_file():
    ST = pickle.load(open(_STATS, "rb"))
    _NUMERIC_COLS = ST["cols"]
    MU, STD = np.asarray(ST["mu"], np.float32), np.asarray(ST["std"], np.float32)
    STD[STD < 1e-6] = 1.0
    print(f"[prov2hyper] global stats → {_STATS}")
else:
    MU = STD = None
    _NUMERIC_COLS = [
        "Flow Duration", "Flow Bytes/s", "Flow Packets/s",
        "Total Fwd Packet", "Total Bwd packets",
        "Fwd Packet Length Mean", "Bwd Packet Length Mean",
    ]
    print("[prov2hyper] global stats not found; per-graph z-score")

# ── helpers ─────────────────────────────────────────────────
def _incidence(G: nx.MultiDiGraph):
    vidx = {n: i for i, n in enumerate(G.nodes())}
    eidx, rows, cols = {}, [], []
    for u, v, a in G.edges(data=True):
        eid = a.get("event_id", f"{u}->{v}")
        eidx.setdefault(eid, len(eidx))
        ei = eidx[eid]
        rows += [vidx[u], vidx[v]]
        cols += [ei, ei]
    H = SparseTensor(row=torch.tensor(rows),
                     col=torch.tensor(cols),
                     sparse_sizes=(len(vidx), len(eidx)))
    return H, vidx

def _numeric(G: nx.MultiDiGraph, names: list[str]) -> torch.Tensor:
    n = len(names); k = len(_NUMERIC_COLS)
    arr = np.zeros((n, k), dtype=np.float32)
    is_flow = np.array([G.nodes[nm].get("type") == "flow" for nm in names])
    for i, nm in enumerate(names):
        attrs = G.nodes[nm]
        for j, col in enumerate(_NUMERIC_COLS):
            try:
                arr[i, j] = float(attrs.get(col, 0.0))
            except (TypeError, ValueError):
                arr[i, j] = 0.0
    if is_flow.any():
        if MU is not None:
            arr[is_flow] = (arr[is_flow] - MU) / STD
        else:
            mu = arr[is_flow].mean(0, keepdims=True)
            sd = arr[is_flow].std(0, keepdims=True);  sd[sd < 1e-6] = 1.0
            arr[is_flow] = (arr[is_flow] - mu) / sd
    arr[~is_flow] = 0.0
    arr = np.clip(arr, -8.0, 8.0)
    return torch.from_numpy(np.nan_to_num(arr))


def _parse_timestamp(raw) -> float:
    """Convert heterogeneous timestamp values into unix seconds."""
    if raw is None:
        return math.nan
    if isinstance(raw, (int, float)):
        return float(raw) if math.isfinite(raw) else math.nan
    if isinstance(raw, str):
        txt = raw.strip()
        if not txt:
            return math.nan
        try:
            return float(txt)
        except ValueError:
            try:
                adj = txt[:-1] + "+00:00" if txt.endswith("Z") else txt
                return datetime.fromisoformat(adj).timestamp()
            except ValueError:
                return math.nan
    if hasattr(raw, "timestamp"):
        try:
            return float(raw.timestamp())
        except Exception:
            return math.nan
    return math.nan

# ── main ────────────────────────────────────────────────────
def prov_graphml_to_hypergraph(path: str):
    G = nx.read_graphml(path)
    names = list(G.nodes())
    H, vidx = _incidence(G)
    edge_src, edge_dst = [], []
    for u, v in G.edges():
        ui, vi = vidx[u], vidx[v]
        edge_src.extend([ui, vi])
        edge_dst.extend([vi, ui])
    if edge_src:
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    # type one-hot
    types = [G.nodes[n].get("type", "unknown") for n in names]
    t2i = {t:i for i,t in enumerate(sorted(set(types)))}
    X_type = torch.eye(len(t2i))[torch.tensor([t2i[t] for t in types])]
    # degree
    deg = torch.tensor([H[i].nnz() for i in range(len(names))], dtype=torch.float32)
    deg = (deg / deg.max().clamp(min=1.0)).unsqueeze(1)
    # numeric
    X_num = _numeric(G, names)
    X = torch.cat([X_type, deg, X_num], dim=1)

    # timestamps (unix seconds) + flow order
    raw_ts = np.array([_parse_timestamp(G.nodes[n].get("Timestamp")) for n in names],
                      dtype=np.float64)
    valid = np.isfinite(raw_ts)
    if valid.any():
        fallback = raw_ts[valid].min()
        raw_ts[~valid] = fallback
    else:
        raw_ts[:] = 0.0
    timestamps = torch.from_numpy(raw_ts.astype(np.float32))

    is_flow_mask = np.array([t == "flow" for t in types], dtype=bool)
    flow_idx = np.flatnonzero(is_flow_mask)
    if flow_idx.size:
        sorter = np.argsort(raw_ts[flow_idx], kind="mergesort")
        flow_order = torch.from_numpy(flow_idx[sorter].astype(np.int64))
    else:
        flow_order = torch.empty(0, dtype=torch.long)

    return H, X, names, timestamps, flow_order, edge_index
