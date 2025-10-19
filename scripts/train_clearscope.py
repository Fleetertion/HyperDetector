# -*- coding: utf-8 -*-
"""
train_HGNN_BSA_khop.py
- 构图：k-邻域超边（每个节点一条超边，包含自身+k个邻居），k=4, shuffle=False
- 模型：两层HGNN + Block Self-Attention（接口名仍 HGNN，兼容旧代码）
- 数据：支持 DARPA (cadets/trace/theia/fivedirections) 与 clearscope
- 稳健：Z-score、lr=1e-3、wd=1e-4、梯度裁剪、固定随机种子
"""
import os, os.path as osp, argparse, time, math, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
from matplotlib import pyplot as plt
from torch_geometric.data import Data

from data_process_train import *
from data_process_test import *

# -------------------- 日志与种子 --------------------
def show(*s):
    for i in range(len(s)):
        print(str(s[i]) + ' ', end='')
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

# -------------------- 模型（HGNN+BSA） --------------------
class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(in_ft, out_ft))
        self.bias   = nn.Parameter(torch.Tensor(out_ft)) if bias else None
        self.reset_parameters()
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        nn.init.uniform_(self.weight, -stdv, stdv)
        if self.bias is not None: nn.init.uniform_(self.bias, -stdv, stdv)
    def forward(self, x, G):
        # 训练/推理一致：先 G 传播再线性
        x = torch.sparse.mm(G, x)
        x = x.matmul(self.weight)
        if self.bias is not None: x = x + self.bias
        return x

from torch.nn import TransformerEncoderLayer
class BlockSelfAttention(nn.Module):
    def __init__(self, dim_model: int, nhead: int = 4, block_size: int = 256, dropout: float = 0.1):
        super().__init__()
        self.block_size = block_size
        self.enc = TransformerEncoderLayer(
            d_model=dim_model, nhead=nhead,
            dim_feedforward=dim_model*2,
            batch_first=True, dropout=dropout, activation='gelu'
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(0) <= self.block_size:
            return self.enc(x.unsqueeze(0)).squeeze(0)
        chunks = torch.chunk(x, math.ceil(x.size(0)/self.block_size), 0)
        return torch.cat([self.enc(c.unsqueeze(0)).squeeze(0) for c in chunks], 0)

class HGNN(nn.Module):
    def __init__(self, in_ch, n_class, n_hid=32, dropout=0.5,
                 d_model=64, nhead=4, block_size=256, attn_dropout=0.1):
        super().__init__()
        self.hgc1 = HGNN_conv(in_ch, n_hid, bias=True)
        self.hgc2 = HGNN_conv(n_hid, d_model, bias=True)
        self.attn = BlockSelfAttention(d_model, nhead=nhead, block_size=block_size, dropout=attn_dropout)
        self.cls  = nn.Linear(d_model, n_class)
        self.drop = dropout
    def forward(self, x, G):
        x = F.relu(self.hgc1(x, G))
        x = F.dropout(x, self.drop, self.training)
        x = F.relu(self.hgc2(x, G))
        x = self.attn(x)
        x = self.cls(x)
        return F.log_softmax(x, dim=1)

# -------------------- k-邻域构图（关键） --------------------
def construct_H_khop(edge_index, num_nodes, k: int = 4, shuffle: bool = False):
    """
    每个节点 v 生成超边 e_v = {v} ∪ sample_k(N(v))
    E = N；nnz(H) ≈ N*(k+1)；推荐 shuffle=False + 固定随机种子以复现。
    """
    if edge_index.numel() == 0 or num_nodes == 0:
        return sp.coo_matrix((num_nodes, 0))
    ei = edge_index.cpu().numpy()
    adj = [[] for _ in range(num_nodes)]
    for u, v in ei.T:
        adj[u].append(v); adj[v].append(u)

    rows, cols = [], []
    e_id = 0
    for center in range(num_nodes):
        nbrs = adj[center]
        if len(nbrs) > k:
            nbrs = random.sample(nbrs, k) if shuffle else nbrs[:k]
        # 超边包含中心 + 邻居
        hyper_nodes = [center] + nbrs
        rows.extend(hyper_nodes)
        cols.extend([e_id]*len(hyper_nodes))
        e_id += 1

    data = np.ones(len(rows), dtype=np.float32)
    H = sp.coo_matrix((data, (rows, cols)), shape=(num_nodes, e_id))
    return H

def generate_G(H):
    H = H.tocoo()
    if H.shape[1] == 0:
        i = torch.arange(H.shape[0])
        return torch.sparse_coo_tensor(torch.stack([i,i]), torch.ones_like(i, dtype=torch.float),
                                       (H.shape[0], H.shape[0])).coalesce()
    idx = np.vstack((H.row, H.col))
    H_t = torch.sparse_coo_tensor(torch.LongTensor(idx), torch.FloatTensor(H.data), torch.Size(H.shape)).coalesce()
    HT  = H_t.transpose(0,1).coalesce()
    DV  = torch.sparse.sum(H_t, dim=1).to_dense() + 1e-10
    DE  = torch.sparse.sum(H_t, dim=0).to_dense() + 1e-10
    DV_inv_sqrt = torch.pow(DV, -0.5)
    DE_inv      = torch.pow(DE, -1.0)

    H_idx = H_t.indices(); H_val = H_t.values()
    H_DE  = torch.sparse_coo_tensor(H_idx, H_val * DE_inv[H_idx[1]], H_t.size()).coalesce()
    temp  = torch.sparse.mm(H_DE, HT).coalesce()
    ti, tv = temp.indices(), temp.values()
    vr = DV_inv_sqrt[ti[0]]; vc = DV_inv_sqrt[ti[1]]
    G = torch.sparse_coo_tensor(ti, tv * vr * vc, temp.size()).coalesce()
    return G

# -------------------- Feature 标准化 --------------------
def standardize_features_(data: Data):
    if data.x is None or data.x.numel() == 0: return
    with torch.no_grad():
        mu = data.x.mean(0, keepdim=True)
        sigma = data.x.std(0, keepdim=True).clamp_min(1e-6)
        data.x = (data.x - mu) / sigma

# -------------------- Clearscope 数据加载 --------------------
def load_clearscope_data(benign_dir, attack_dir, benign_ids, attack_ids, feature_num=4):
    nodes = {}; node_feats = {}; node_labs = {}; edges = []
    def gid(k):
        if k not in nodes:
            idx = len(nodes); nodes[k]=idx; node_feats[idx]=[]; node_labs[idx]=[]
        return nodes[k]
    def add_line(aid, atype, obj_token, lab):
        spx = obj_token.split(':')
        if len(spx)!=4: return
        oid, obj, act, ts = spx
        a = gid(aid); b = gid(oid)
        edges.append([a,b]); edges.append([b,a])
        try: f = [float(atype), float(obj), float(act), float(ts)]
        except: f = [0.0,0.0,0.0,0.0]
        f = np.array(f, dtype=np.float32)
        node_feats[a].append(f); node_labs[a].append(lab)
        node_feats[b].append(f); node_labs[b].append(lab)

    # benign
    for fid in benign_ids:
        fpath = osp.join(benign_dir, f"clearscope-benign_{fid}.txt")
        if not osp.exists(fpath): print(f"[Warn] missing: {fpath}"); continue
        for ln in open(fpath,'r',encoding='utf-8'):
            s = ln.strip().split()
            if len(s)>=3: add_line(s[0], s[1], s[2], 0)
    # attack
    for fid in attack_ids:
        fpath = osp.join(attack_dir, f"clearscope-e3-attack_{fid}.txt")
        if not osp.exists(fpath): print(f"[Warn] missing: {fpath}"); continue
        for ln in open(fpath,'r',encoding='utf-8'):
            s = ln.strip().split()
            if len(s)>=3: add_line(s[0], s[1], s[2], 1)

    N = len(nodes)
    if N==0:
        x = torch.zeros((0,feature_num),dtype=torch.float); y=torch.zeros((0,),dtype=torch.long)
        ei = torch.zeros((2,0),dtype=torch.long)
    else:
        xs, ys = [], []
        for i in range(N):
            if node_feats[i]: xs.append(np.mean(np.stack(node_feats[i],0),0))
            else: xs.append(np.zeros(feature_num,dtype=np.float32))
            if node_labs[i]: ys.append(int(np.bincount(np.array(node_labs[i],int)).argmax()))
            else: ys.append(0)
        x = torch.tensor(np.stack(xs,0),dtype=torch.float)
        y = torch.tensor(ys,dtype=torch.long)
        ei = torch.tensor(edges,dtype=torch.long).t().contiguous() if len(edges)>0 else torch.zeros((2,0),dtype=torch.long)

    idx = np.arange(N); np.random.shuffle(idx)
    split = int(0.8*N) if N>0 else 0
    train_mask = torch.zeros(N,dtype=torch.bool); test_mask=torch.zeros(N,dtype=torch.bool)
    if N>0: train_mask[idx[:split]]=True; test_mask[idx[split:]]=True
    data = Data(x=x,y=y,edge_index=ei,train_mask=train_mask,test_mask=test_mask); data.num_nodes=N
    feat_dim = x.size(1) if N>0 else 4; n_class = int(y.max().item()+1) if y.numel()>0 else 2
    return data, feat_dim, n_class

# -------------------- 训练/测试/验证主流程 --------------------
precision_vals, recall_vals = [], []
thre_map = {"cadets":1.5,"trace":1.0,"theia":1.5,"fivedirections":1.0,"clearscope":1.0}

def train_step(model, data, G, optimizer):
    model.train(); optimizer.zero_grad()
    out = model(data.x, G)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()
    return loss.item()

def test_acc(model, data, G, mask):
    model.eval()
    with torch.no_grad():
        out = model(data.x, G)
        pred = out[mask].max(1)[1]
        denom = mask.sum().item()
        if denom==0: return 0.0
        return pred.eq(data.y[mask]).float().mean().item()

def final_test_uncertain(model, data, G, thre, mask):
    model.eval(); fp=[]; tn=[]
    with torch.no_grad():
        out = model(data.x, G)
        pro = F.softmax(out[mask], dim=1)
        if pro.size(0)==0: return 0.0, fp, tn
        top1 = pro.max(1)
        tmp = pro.clone()
        tmp[torch.arange(tmp.size(0)), top1[1]] = -1
        top2 = tmp.max(1)
        ratio = top1[0] / (top2[0] + 1e-10)
        pred = top1[1]
        pred[ratio < thre] = 100
        correct = pred.eq(data.y[mask])
        idx = mask.nonzero(as_tuple=False).view(-1)
        for i, ok in enumerate(correct):
            (tn if ok else fp).append(idx[i].item())
    acc = correct.float().mean().item() if pro.size(0)>0 else 0.0
    return acc, fp, tn

def main():
    set_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=str, default='cadets', choices=['cadets','trace','theia','fivedirections','clearscope'])
    parser.add_argument('--k', type=int, default=4)
    parser.add_argument('--shuffle', action='store_true', help='k邻域邻居随机采样（默认False顺序取前k个）')
    args = parser.parse_args()
    thre = thre_map[args.scene]

    # === 加载数据 ===
    if args.scene != 'clearscope':
        path_tr = f'../graphchi-cpp-master/graph_data/darpatc/{args.scene}_train.txt'
        path_te = f'../graphchi-cpp-master/graph_data/darpatc/{args.scene}_test.txt'
        data_list, feat_dim, n_class, adj, adj2 = MyDataset(path_tr, 0)
        dataset = TestDataset(data_list); data = dataset[0]
        # 验证数据在 validate() 再加载
    else:
        benign_dir = "../graphchi-cpp-master/graph_data/clearscope/benign"
        attack_dir = "../graphchi-cpp-master/graph_data/clearscope/attack"
        benign_ids_train = [0,1]; attack_ids_train = [0]
        data, feat_dim, n_class = load_clearscope_data(benign_dir, attack_dir, benign_ids_train, attack_ids_train)

    # 标准化
    standardize_features_(data)

    # 构图 G（k-邻域）
    H = construct_H_khop(data.edge_index, data.num_nodes, k=args.k, shuffle=args.shuffle)
    G = generate_G(H).to(torch.device('cpu'))

    # 模型与优化器
    model = HGNN(in_ch=feat_dim, n_class=n_class, n_hid=32, dropout=0.5, d_model=64, nhead=4, block_size=256)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # 初训
    for epoch in range(1, 30):
        loss = train_step(model, data, G, optimizer)
        acc  = test_acc(model, data, G, data.test_mask)
        show(epoch, loss, acc)

    # 自举循环 + 落盘
    loop_num = 0; bad_cnt = 0; max_thre = 3
    while True:
        acc, fp, tn = final_test_uncertain(model, data, G, thre, data.test_mask)
        if len(tn)==0: bad_cnt += 1
        else: bad_cnt = 0
        if bad_cnt >= max_thre: break

        if len(tn)>0:
            # 从 mask 中剔除 TN，并落盘 fp/tn
            for i in tn: data.train_mask[i]=False; data.test_mask[i]=False
            # 保存特征以复用你原来的分析流程
            os.makedirs('../models', exist_ok=True)
            torch.save(model.state_dict(), f'../models/model_{loop_num}')
            loop_num += 1
            if len(fp)==0: break

        # 继续训练
        for epoch in range(1, 150):
            loss = train_step(model, data, G, optimizer)
            acc  = test_acc(model, data, G, data.test_mask)
            show(epoch, loss, acc)
            if loss < 1: break

    show('Finish training')

    # === 验证（与之前一致）===
    if args.scene != 'clearscope':
        # DARPA 的邻域归因式 P/R
        data_list_A, feat_dim_A, n_class_A, adjA, adj2A, nodeA, _nodeA, _neibor = MyDatasetA(path_te, 0)
        datasetA = TestDatasetA(data_list_A); dataA = datasetA[0]
        standardize_features_(dataA)
        H_A = construct_H_khop(dataA.edge_index, dataA.num_nodes, k=args.k, shuffle=args.shuffle)
        G_A = generate_G(H_A)

        out_loop = -1
        while True:
            out_loop += 1
            mpath = f'../models/model_{out_loop}'
            if not osp.exists(mpath): break
            model.load_state_dict(torch.load(mpath, map_location='cpu'))
            _, fpA, tnA = final_test_uncertain(model, dataA, G_A, thre, dataA.test_mask)
            _fp = 0; tempNodeA = {i:1 for i in nodeA}
            for i in fpA:
                if i not in _nodeA: _fp += 1
                if i in _neibor:
                    for j in _neibor[i]:
                        if j in tempNodeA: tempNodeA[j]=0
            _tp = sum(1 for v in tempNodeA.values() if v==0)
            precision = _tp/(_tp+_fp+1e-10); recall = _tp/len(nodeA) if len(nodeA)>0 else 0.0
            precision_vals.append(precision); recall_vals.append(recall)
            print('Precision:', precision, 'Recall:', recall)
            for j in tnA: dataA.test_mask[j]=False

        # 画 PR
        if recall_vals:
            plt.figure(); plt.plot(recall_vals, precision_vals, marker='o')
            plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('PR Curve'); plt.grid(); plt.show()
    else:
        # Clearscope：直接按 test_mask 计算 P/R（最后一轮）
        with torch.no_grad():
            out = model(data.x, G)
            preds = out[data.test_mask].max(1)[1]
            gt = data.y[data.test_mask]
            TP = ((preds==1)&(gt==1)).sum().item()
            FP = ((preds==1)&(gt==0)).sum().item()
            FN = ((preds==0)&(gt==1)).sum().item()
            precision = TP/(TP+FP+1e-10); recall = TP/(TP+FN+1e-10)
            print('Clearscope Precision:', precision, 'Recall:', recall)

if __name__ == "__main__":
    graphchi_root = os.path.abspath(os.path.join(os.getcwd(), '../graphchi-cpp-master'))
    os.environ['GRAPHCHI_ROOT'] = graphchi_root
    main()