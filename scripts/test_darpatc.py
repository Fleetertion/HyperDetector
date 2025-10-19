#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HGNN_Attn 推理脚本  —— 与 train_pro()/validate() 保持同一接口
---------------------------------------------------------------
1. 构造高阶超图 H  → 归一化拉普拉斯 G
2. 加载多轮训练得到的 model_k
3. 对 data.test_mask 上全图推理，逐轮去除预测正确的 TN
4. 最终将剩余 test_mask 写入 alarm.txt
"""
import os, os.path as osp, time, argparse, torch
import torch.nn.functional as F
from data_process_test import *
# 若 construct_H、generate_G、HGNN_Attn 在 utils:
# from hgnn_utils import construct_H, generate_G, HGNN_Attn

# ─────────────────────────────────────────────────────────────
# 你在训练脚本里定义 / 修改过的这几个函数和类应当保持一致 ↓
# 把它们复制进来或确保 import 成功
# ─────────────────────────────────────────────────────────────
import math, random, numpy as np, scipy.sparse as sp
from torch.nn import TransformerEncoderLayer
from torch.nn.parameter import Parameter
import torch.nn as nn

def construct_H(edge_index, num_nodes, k: int = 4, shuffle: bool = True):
    edge_index = edge_index.numpy()
    adj = [[] for _ in range(num_nodes)]
    for u, v in edge_index.T:
        adj[u].append(v); adj[v].append(u)

    rows, cols = [], []
    e_id = 0
    for center in range(num_nodes):
        nbrs = adj[center]
        if len(nbrs) > k:
            nbrs = random.sample(nbrs, k) if shuffle else nbrs[:k]
        hyper_nodes = [center] + nbrs
        rows.extend(hyper_nodes)
        cols.extend([e_id] * len(hyper_nodes))
        e_id += 1

    data = np.ones(len(rows), dtype=np.float32)
    return sp.coo_matrix((data, (rows, cols)), shape=(num_nodes, e_id))

def generate_G(H):
    H = H.tocoo()
    row, col, data = H.row, H.col, H.data
    i = torch.LongTensor(np.vstack((row, col)))
    v = torch.FloatTensor(data)
    H = torch.sparse_coo_tensor(i, v, torch.Size(H.shape)).coalesce()
    HT = H.transpose(0, 1).coalesce()
    DV = torch.sparse.sum(H, dim=1).to_dense() + 1e-10
    DE = torch.sparse.sum(H, dim=0).to_dense() + 1e-10
    DV_inv_sqrt = torch.pow(DV, -0.5)
    DE_inv = torch.pow(DE, -1.0)
    H_DE = torch.sparse_coo_tensor(
        H.indices(),
        H.values() * DE_inv[H.indices()[1]],
        H.size()
    ).coalesce()
    temp = torch.sparse.mm(H_DE, HT).coalesce()
    vals = temp.values() * DV_inv_sqrt[temp.indices()[0]] * DV_inv_sqrt[temp.indices()[1]]
    return torch.sparse_coo_tensor(temp.indices(), vals, temp.size()).coalesce()

class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super().__init__()
        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        self.bias = Parameter(torch.Tensor(out_ft)) if bias else None
        self.reset_parameters()
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        nn.init.uniform_(self.weight, -stdv, stdv)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -stdv, stdv)
    def forward(self, x, G):
        x = x.matmul(self.weight)
        if self.bias is not None: x = x + self.bias
        return torch.sparse.mm(G, x)

class BlockSelfAttention(nn.Module):
    def __init__(self, dim_model, nhead=4, block_size=256):
        super().__init__()
        self.block_size = block_size
        self.attn = TransformerEncoderLayer(dim_model, nhead,
                                             dim_feedforward=dim_model*2,
                                             batch_first=True,
                                             dropout=0.1, activation='gelu')
    def forward(self, x):
        if x.size(0) <= self.block_size:
            return self.attn(x.unsqueeze(0)).squeeze(0)
        chunks = torch.chunk(x, math.ceil(x.size(0)/self.block_size), 0)
        return torch.cat([self.attn(c.unsqueeze(0)).squeeze(0) for c in chunks], 0)

class HGNN_Attn(nn.Module):
    def __init__(self, in_ch, n_class, n_hid=32, d_model=64,
                 nhead=4, dropout=0.5, block_size=256):
        super().__init__()
        self.hgc1 = HGNN_conv(in_ch, n_hid)
        self.hgc2 = HGNN_conv(n_hid, d_model)
        self.attn = BlockSelfAttention(d_model, nhead, block_size)
        self.cls  = nn.Linear(d_model, n_class)
        self.drop = dropout
    def forward(self, x, G):
        x = F.relu(self.hgc1(x, G))
        x = F.dropout(x, self.drop, self.training)
        x = F.relu(self.hgc2(x, G))
        x = self.attn(x)
        return F.log_softmax(self.cls(x), dim=1)
# ─────────────────────────────────────────────────────────────


def show(msg):
    print(f"{msg} {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

# ────────────────── 参数解析 ──────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--scene', required=True, choices=['cadets','trace','theia','fivedirections'])
args = parser.parse_args()
thre_map = {"cadets":1.5,"trace":1.0,"theia":1.5,"fivedirections":1.0}
thre = thre_map[args.scene]
b_size = 5000            # 已不再使用，但保留变量以防外部引用

# ────────────────── 读入图数据 ──────────────────
graphId = 1
path = f'../dataset/darpatc/{args.scene}_test.txt'
show(f"Start testing graph {graphId}")

data_raw, feat_dim, num_class, adj, adj2, nodeA, _nodeA, _neibor = MyDatasetA(path, 0)
dataset = TestDatasetA(data_raw)
data = dataset[0]

# ────────────────── 构造 H / G ──────────────────
H = construct_H(data.edge_index, data.num_nodes)
G = generate_G(H).to(torch.device('cpu'))

# ────────────────── 模型准备 ──────────────────
device = torch.device('cpu')
model = HGNN_Attn(feat_dim, num_class).to(device)

def test(mask):
    model.eval()
    with torch.no_grad():
        out = model(data.x.to(device), G)
        pred = out[mask].max(1)[1]
        pro  = F.softmax(out[mask], dim=1)
        pro1 = pro.max(1)
        pro[range(mask.sum()), pro1[1]] = -1          # 把最大类别概率置 -1
        pro2 = pro.max(1)
        pred[pro1[0] / (pro2[0] + 1e-12) < thre] = 100
        correct_mask = pred.eq(data.y[mask].to(device))
        fp = mask.nonzero(as_tuple=False)[~correct_mask].view(-1).tolist()
        tn = mask.nonzero(as_tuple=False)[correct_mask].view(-1).tolist()
        acc = correct_mask.sum().item() / mask.sum().item()
        return acc, fp, tn

# ────────────────── 多轮模型迭代 ──────────────────
loop_num, best_acc = 0, 0.0
while loop_num <= 100:
    model_path = f'../models/model_{loop_num}'
    if not osp.exists(model_path):
        loop_num += 1
        continue
    model.load_state_dict(torch.load(model_path, map_location=device))
    acc, fp, tn = test(data.test_mask)
    show(f"model_{loop_num:02d} acc={acc:.4f} fp={len(fp)}")
    best_acc = max(best_acc, acc)
    # 把预测正确节点从 test_mask 移除
    data.test_mask[tn] = False
    if acc == 1.0:
        break
    loop_num += 1

# ────────────────── 输出 alarm.txt ──────────────────
with open('alarm.txt', 'w') as fw:
    fw.write(f"{data.test_mask.sum().item()}\n")
    for idx in data.test_mask.nonzero(as_tuple=False).view(-1).tolist():
        fw.write('\n' + str(idx) + ':')
        # 两跳邻居收集（与训练脚本一致）
        neibor = set()
        for _adj in (adj, adj2):
            if idx in _adj:
                for j in _adj[idx]:
                    neibor.add(j)
                    if j in _adj:
                        neibor.update(_adj[j])
        fw.write(' '.join(map(str, neibor)))
show(f"Finish testing graph {graphId}  best_acc={best_acc:.4f}")