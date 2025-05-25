#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HGNN + Block Self-Attention on APT2021
--------------------------------------
① 高阶图卷积  (HGNN_conv ×2)                        → 提取局部高阶关系
② Block Self-Attention (块内多头自注意力, 近线性)   → 建模全局依赖
③ 线性分类头                                      → 输出二分类 logits
训练 / 评估流程沿用原 NeighborLoader，小改 forward 调用即可。
"""

# ────────────────────────── 0. import ──────────────────────────
import math, os, random, time, sys
from itertools import combinations

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import add_self_loops
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (precision_score, recall_score,
                             f1_score, confusion_matrix)

# ─────────────────── 1. HGNN & Block Self-Attention ───────────────────
class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super().__init__()
        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        self.bias   = Parameter(torch.Tensor(out_ft)) if bias else None
        self.reset_parameters()
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        nn.init.uniform_(self.weight, -stdv, stdv)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -stdv, stdv)
    def forward(self, x, G):
        x = x @ self.weight
        if self.bias is not None:
            x = x + self.bias
        return torch.sparse.mm(G, x)

class HGNN(nn.Module):
    def __init__(self, in_ch, hid_ch=32, dropout=0.5):
        super().__init__()
        self.conv1 = HGNN_conv(in_ch, hid_ch)
        self.conv2 = HGNN_conv(hid_ch, hid_ch)
        self.dp    = dropout
    def forward(self, x, G):
        x = F.relu(self.conv1(x, G))
        x = F.dropout(x, p=self.dp, training=self.training)
        x = F.relu(self.conv2(x, G))
        return x                              # [N, hid_ch]

class BlockSelfAttention(nn.Module):
    """
    将长度 N 的序列按 block_size 切块，只在块内做多头自注意力。
    复杂度 O(N * block_size)；block_size≪N 时近似线性。
    """
    def __init__(self, d_model, nhead=4, block_size=256, dropout=0.1):
        super().__init__()
        from torch.nn import TransformerEncoderLayer
        self.block_size = block_size
        self.attn = TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout, activation='gelu',
            batch_first=True                 # (B, L, d)
        )
    def forward(self, x):                    # x: [N, d]
        if x.size(0) <= self.block_size:
            return self.attn(x.unsqueeze(0)).squeeze(0)
        chunks = torch.chunk(
            x, math.ceil(x.size(0) / self.block_size), dim=0)
        outs = [self.attn(c.unsqueeze(0)).squeeze(0) for c in chunks]
        return torch.cat(outs, dim=0)

class HGNN_Attn(nn.Module):
    """
    HGNN (两层) + Block Self-Attention + 线性分类
    """
    def __init__(self, in_ch, hid_ch=32, nhead=4,
                 block_size=256, dropout=0.5, n_class=2):
        super().__init__()
        self.hgnn = HGNN(in_ch, hid_ch, dropout)
        self.bsa  = BlockSelfAttention(hid_ch, nhead, block_size, dropout)
        self.cls  = nn.Linear(hid_ch, n_class)
        self.dp   = dropout
    def forward(self, x, G):
        h = self.hgnn(x, G)                  # HGNN
        h = self.bsa(h)                      # Block Self-Attn
        h = F.dropout(h, p=self.dp, training=self.training)
        return F.log_softmax(self.cls(h), dim=1)

# ─────────────────── 2. 数据准备（与原脚本一致） ───────────────────
def prepare_graph_from_apt2021(csv_file, max_edges_per_key=200):
    print("读取 APT2021 CSV...")
    df = pd.read_csv(csv_file).dropna().reset_index(drop=True)

    labels = (df['Label'] != 'NormalTraffic').astype(int)

    drop_cols = ['Flow ID','Src IP','Src Port','Dst IP','Dst Port',
                 'Protocol','Timestamp','Label']
    feats = df.drop(columns=drop_cols)\
              .replace([np.inf,-np.inf],0).fillna(0).values.astype(np.float32)
    feats = StandardScaler().fit_transform(feats)

    num_nodes = len(df)
    src_ips, dst_ips = df['Src IP'].values, df['Dst IP'].values

    ip_to_indices = {}
    for i in range(num_nodes):
        for ip in (src_ips[i], dst_ips[i]):
            ip_to_indices.setdefault(ip, []).append(i)

    edge_set = set()
    for idx_list in ip_to_indices.values():
        if len(idx_list) > 1:
            if len(idx_list) > max_edges_per_key:
                idx_list = random.sample(idx_list, max_edges_per_key)
            for i1, i2 in combinations(idx_list, 2):
                edge_set.add((i1, i2)); edge_set.add((i2, i1))

    edge_index = (torch.empty((2,0), dtype=torch.long)
                  if len(edge_set) == 0 else
                  torch.tensor(list(edge_set), dtype=torch.long).t().contiguous())
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

    return Data(x=torch.tensor(feats),
                y=torch.tensor(labels, dtype=torch.long),
                edge_index=edge_index)

# ───── 3. 图邻接 → 稀疏归一化矩阵 G（供 HGNN_conv 使用） ─────
def graph_to_sparse_G(edge_index, num_nodes, add_self_loop=True):
    if add_self_loop:
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    values = torch.ones(edge_index.size(1), device=edge_index.device)
    A = torch.sparse_coo_tensor(edge_index, values,
                                (num_nodes, num_nodes)).coalesce()
    deg = torch.sparse.sum(A, dim=1).to_dense() + 1e-10
    deg_inv_sqrt = deg.pow(-0.5)
    idx = A._indices()
    val = A._values() * deg_inv_sqrt[idx[0]] * deg_inv_sqrt[idx[1]]
    return torch.sparse_coo_tensor(idx, val,
                                   (num_nodes, num_nodes)).coalesce()

def extract_subG(G, n_id):
    """提取子图稀疏邻接矩阵 G_sub = G[n_id][:, n_id]"""
    id_map = {int(old): i for i, old in enumerate(n_id)}
    idx, val = G._indices(), G._values()
    mask = [(int(r) in id_map and int(c) in id_map)
            for r, c in zip(idx[0].tolist(), idx[1].tolist())]
    mask = torch.tensor(mask, dtype=torch.bool, device=val.device)
    idx_sel, val_sel = idx[:, mask], val[mask]
    row = torch.tensor([id_map[int(r)] for r in idx_sel[0].tolist()],
                       device=val.device)
    col = torch.tensor([id_map[int(c)] for c in idx_sel[1].tolist()],
                       device=val.device)
    return torch.sparse_coo_tensor(torch.stack([row, col]), val_sel,
                                   (len(n_id), len(n_id))).coalesce()

# ─────────────────── 4. 训练 & 评估函数 ───────────────────
def train_one_epoch(model, loader, opt, device, G_full):
    model.train()
    tot_loss = tot_samples = 0
    for batch in loader:
        batch = batch.to(device)
        n_id = batch.n_id.cpu().numpy()
        G_sub = extract_subG(G_full, n_id)
        out = model(batch.x, G_sub)

        loss = F.nll_loss(out[batch.train_mask], batch.y[batch.train_mask])
        opt.zero_grad(); loss.backward(); opt.step()

        bs = batch.train_mask.sum().item()
        tot_loss += loss.item() * bs
        tot_samples += bs
    return tot_loss / tot_samples

@torch.no_grad()
def test_and_eval(model, loader, device, G_full):
    model.eval(); preds=[]; trues=[]; masks=[]
    for batch in loader:
        batch = batch.to(device)
        n_id = batch.n_id.cpu().numpy()
        G_sub = extract_subG(G_full, n_id)
        out = model(batch.x, G_sub).argmax(dim=1)
        preds.append(out.cpu()); trues.append(batch.y.cpu())
        masks.append(batch.test_mask.cpu())
    preds = torch.cat(preds).numpy()
    trues = torch.cat(trues).numpy()
    masks = torch.cat(masks).numpy()
    p = precision_score(trues[masks], preds[masks], zero_division=0)
    r = recall_score(trues[masks], preds[masks], zero_division=0)
    f1 = f1_score(trues[masks], preds[masks], zero_division=0)
    cm = confusion_matrix(trues[masks], preds[masks])
    return cm, p, r, f1

# ─────────────────── 5. 主函数 ───────────────────
def main():
    csv_path = "SCVIC-APT-2021-Training.csv"   # ★ 修改为实际路径
    data = prepare_graph_from_apt2021(csv_path)

    n = data.num_nodes
    idxs = np.arange(n)
    tr, te = train_test_split(idxs, test_size=0.2,
                              random_state=42, stratify=data.y.numpy())
    data.train_mask = torch.zeros(n, dtype=torch.bool)
    data.test_mask  = torch.zeros(n, dtype=torch.bool)
    data.train_mask[tr] = True; data.test_mask[te] = True

    train_loader = NeighborLoader(
        data, input_nodes=data.train_mask,
        num_neighbors=[10, 5], batch_size=1024, shuffle=True)
    test_loader = NeighborLoader(
        data, input_nodes=data.test_mask,
        num_neighbors=[10, 5], batch_size=1024, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    G_full = graph_to_sparse_G(data.edge_index, data.num_nodes)

    model = HGNN_Attn(
        in_ch=data.x.size(1), hid_ch=32,
        nhead=4, block_size=256, dropout=0.5, n_class=2).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)

    for epoch in tqdm(range(1, 11), desc="Epochs", unit="epoch"):
        loss = train_one_epoch(model, train_loader, optim, device, G_full)
        cm, p, r, f1 = test_and_eval(model, test_loader, device, G_full)
        print(f"Epoch {epoch:02d} | "
              f"Loss {loss:.4f} | P {p:.4f} R {r:.4f} F1 {f1:.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/hgnn_attn_neighbor.pt")
    print("模型已保存到 models/hgnn_attn_neighbor.pt")

if __name__ == "__main__":
    main()