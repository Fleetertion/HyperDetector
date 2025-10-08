#!/usr/bin/env python3
"""
csv2_batches_graphml.py
-----------------------
把 APT‑2021 CSV 拆分成 batch_<id>.graphml（与 splits.yaml 完全对应）

用法:
    python csv2_batches_graphml.py --csv apt2021.csv \
           --out_dir data/batches --window 5
"""

import argparse, os
from pathlib import Path
import pandas as pd
import networkx as nx
from tqdm import tqdm
import numpy as np

def _sanitize(v):
    if pd.isna(v):                 # NaN → 空串
        return ""
    if isinstance(v, pd.Timestamp):
        return v.isoformat()       # 2025‑07‑27T19:41:02
    if isinstance(v, np.generic):  # numpy.int64 / float32 …
        return v.item()
    if isinstance(v, (list, dict, set)):
        return str(v)
    return v

def row2_graph_edges(idx, row, G):
    src = f"{row['Src IP']}:{int(row['Src Port'])}"
    dst = f"{row['Dst IP']}:{int(row['Dst Port'])}"
    G.add_node(src, type='endpoint', ip=row['Src IP'], port=int(row['Src Port']))
    G.add_node(dst, type='endpoint', ip=row['Dst IP'], port=int(row['Dst Port']))
    flow = f"flow_{idx}"
    attrs = {k: _sanitize(v) for k, v in row.to_dict().items()}
    G.add_node(flow, **attrs)
    G.add_edge(src,  flow, relation='sent',
               bytes=float(row['Total Length of Fwd Packet']),
               pkts=int(row['Total Fwd Packet']))
    G.add_edge(flow, dst, relation='received',
               bytes=float(row['Total Length of Bwd Packet']),
               pkts=int(row['Total Bwd packets']))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out_dir", default="data/batches")
    ap.add_argument("--window", type=int, default=5,
                    help="时间窗（分钟），必须与 generate_splits_yaml.py 相同")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, parse_dates=["Timestamp"])
    df.sort_values("Timestamp", inplace=True)

    start = df["Timestamp"].min()
    win_sec = args.window * 60
    df["batch_id"] = ((df["Timestamp"]-start).dt.total_seconds() //
                      win_sec).astype(int)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    for bid, sub in tqdm(df.groupby("batch_id"),
                         desc="building graph per batch"):
        G = nx.MultiDiGraph(name=f"APT2021_batch_{bid}")
        for idx, row in sub.iterrows():
            row2_graph_edges(idx, row, G)
        out_file = Path(args.out_dir)/f"batch_{bid}.graphml"
        nx.write_graphml(G, out_file)

if __name__ == "__main__":
    main()
