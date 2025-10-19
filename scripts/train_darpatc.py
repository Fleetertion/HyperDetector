import os.path as osp
import os
import argparse
import random
import torch
import time
import torch.nn.functional as F
import torch.nn as nn
import math
import numpy as np
from torch_geometric.data import Data
from torch.nn.parameter import Parameter
from data_process_train import *
from data_process_test import *
import scipy.sparse as sp


thre_map = {"cadets":1.5,"trace":1.0,"theia":1.5,"fivedirections":1.0}

def show(*s):
    for i in range(len(s)):
        print(str(s[i]) + ' ', end='')
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer
from torch.nn.parameter import Parameter

# ──────────────────────────────────────────────────────────
# ① HGNN 基本卷积，与原实现一致
# ──────────────────────────────────────────────────────────
class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super().__init__()
        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, G):
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = torch.sparse.mm(G, x)
        return x


# ──────────────────────────────────────────────────────────
# ② 轻量级块式自注意力
# ──────────────────────────────────────────────────────────
class BlockSelfAttention(nn.Module):
    """
    在长度为 block_size 的块内应用 Multi-Head Self-Attention。
    复杂度: O(N * block_size)；当 block_size ≪ N 时近似线性。
    """
    def __init__(self, dim_model: int, nhead: int = 4, block_size: int = 256):
        super().__init__()
        self.block_size = block_size
        self.attn = TransformerEncoderLayer(
            d_model=dim_model,
            nhead=nhead,
            dim_feedforward=dim_model * 2,
            batch_first=True,      # 方便 (B, L, D) 排布
            dropout=0.1,
            activation='gelu',
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [N, D]
        返回同形状张量
        """
        if x.size(0) <= self.block_size:
            return self.attn(x.unsqueeze(0)).squeeze(0)

        chunks = torch.chunk(x, math.ceil(x.size(0) / self.block_size), dim=0)
        out_chunks = [self.attn(c.unsqueeze(0)).squeeze(0) for c in chunks]
        return torch.cat(out_chunks, dim=0)


# ──────────────────────────────────────────────────────────
# ③ HGNN + Block-Attention 组合模型
# ──────────────────────────────────────────────────────────
class HGNN_Attn(nn.Module):
    """
    两层 HGNN 提取局部高阶关系 + Block Self-Attention 建模全图全局依赖
    输出与原 HGNN 相同，因此训练 / 推理代码完全不变。
    """
    def __init__(
        self,
        in_ch: int,
        n_class: int,
        n_hid: int = 32,
        d_model: int = 64,
        nhead: int = 4,
        dropout: float = 0.5,
        block_size: int = 256,
    ):
        super().__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(in_ch, n_hid)
        self.hgc2 = HGNN_conv(n_hid, d_model)   # 注意：第二层输出改为 d_model
        self.attn = BlockSelfAttention(d_model, nhead, block_size)
        self.cls  = nn.Linear(d_model, n_class)

    def forward(self, x, G):
        x = F.relu(self.hgc1(x, G))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.hgc2(x, G))             # 先升维到 d_model
        x = self.attn(x)                        # 块式自注意力
        x = self.cls(x)                         # 分类头
        return F.log_softmax(x, dim=1)
    

def construct_H(edge_index, num_nodes, k: int = 4, shuffle: bool = True):
    """
    构造高阶超图的 H∈ℝ^{N×E}
    每个节点 v 生成一条超边 e_v = {v} ∪ Neigh_k(v)
    参数
    ----
    edge_index : Tensor [2, E′]    原始图边
    num_nodes  : int               节点总数
    k          : int               每条超边包含的邻居上限 (超边阶数≈k+1)
    shuffle    : bool              是否随机抽样邻居(邻居>k 时)
    """
    edge_index = edge_index.numpy()
    # 1. 建立邻接表
    adj = [[] for _ in range(num_nodes)]
    for u, v in edge_index.T:
        adj[u].append(v)
        adj[v].append(u)

    rows, cols = [], []
    e_id = 0
    for center in range(num_nodes):
        nbrs = adj[center]
        if len(nbrs) == 0:                    # 孤立点 → 只含自身
            nbr_subset = []
        elif len(nbrs) <= k:
            nbr_subset = nbrs
        else:                                 # 邻居过多时随机/顺序采样
            nbr_subset = random.sample(nbrs, k) if shuffle else nbrs[:k]

        hyper_nodes = [center] + nbr_subset   # ≥1+|nbr_subset| 节点
        rows.extend(hyper_nodes)
        cols.extend([e_id] * len(hyper_nodes))
        e_id += 1

    data = np.ones(len(rows), dtype=np.float32)
    H = sp.coo_matrix((data, (rows, cols)), shape=(num_nodes, e_id))
    return H

def generate_G(H):
    H = H.tocoo()
    row = H.row
    col = H.col
    data = H.data
    indices = np.vstack((row, col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(data)
    shape = H.shape
    H = torch.sparse_coo_tensor(i, v, torch.Size(shape)).coalesce()
    HT = H.transpose(0, 1).coalesce()

    DV = torch.sparse.sum(H, dim=1).to_dense()  # [N]
    DE = torch.sparse.sum(H, dim=0).to_dense()  # [E]

    epsilon = 1e-10
    DV = DV + epsilon
    DE = DE + epsilon

    DV_inv_sqrt = torch.pow(DV, -0.5)  
    DE_inv = torch.pow(DE, -1.0)      

    H_indices = H._indices()
    H_values = H._values()
    DE_inv_values = DE_inv[H_indices[1]]
    H_DE_values = H_values * DE_inv_values
    H_DE = torch.sparse_coo_tensor(H_indices, H_DE_values, H.size()).coalesce()

    temp = torch.sparse.mm(H_DE, HT).coalesce()

    temp_indices = temp._indices()
    temp_values = temp._values()
    row_indices = temp_indices[0]
    col_indices = temp_indices[1]
    DV_inv_sqrt_row = DV_inv_sqrt[row_indices]
    DV_inv_sqrt_col = DV_inv_sqrt[col_indices]
    values = temp_values * DV_inv_sqrt_row * DV_inv_sqrt_col

    G = torch.sparse_coo_tensor(temp_indices, values, temp.size())
    return G

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x.to(device), G)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask].to(device))
    loss.backward()
    optimizer.step()
    return loss.item()

def test(mask):
    model.eval()
    with torch.no_grad():
        out = model(data.x.to(device), G)
        pred = out[mask].max(1)[1]
        pro  = F.softmax(out[mask], dim=1)
        pro1 = pro.max(1)
        for i in range(mask.sum()):
            pro[i][pro1[1][i]] = -1
        pro2 = pro.max(1)
        for i in range(mask.sum()):
            if pro1[0][i]/pro2[0][i] < thre:
                pred[i] = 100
        correct = pred.eq(data.y[mask].to(device)).sum().item()
        return correct / mask.sum().item()

def final_test(mask):
    global fp, tn
    model.eval()
    fp = []
    tn = []
    with torch.no_grad():
        out = model(data.x.to(device), G)
        pred = out[mask].max(1)[1]
        pro  = F.softmax(out[mask], dim=1)
        pro1 = pro.max(1)
        for i in range(mask.sum()):
            pro[i][pro1[1][i]] = -1
        pro2 = pro.max(1)
        for i in range(mask.sum()):
            if pro1[0][i]/pro2[0][i] < thre:
                pred[i] = 100
        correct = pred.eq(data.y[mask].to(device)).sum().item()
        mask_indices = mask.nonzero().view(-1)
        for idx, is_correct in zip(mask_indices, pred.eq(data.y[mask].to(device))):
            if not is_correct:
                fp.append(int(idx))
            else:
                tn.append(int(idx))
        return correct / mask.sum().item()

def validate():
    global fp, tn
    global device, model, optimizer, data, G

    show('Start validating')
    path = '../dataset/darpatc/' + args.scene + '_test.txt'
    data, feature_num, label_num, adj, adj2, nodeA, _nodeA, _neibor = MyDatasetA(path, 0)
    dataset = TestDatasetA(data)
    data = dataset[0]
    print(data)
    
    device = torch.device('cpu')	
    model = HGNN_Attn(feature_num, label_num).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    # 构建 H 和 G
    H = construct_H(data.edge_index, data.num_nodes)
    G = generate_G(H)
    G = G.to(device)

    fp = []
    tn = []

    out_loop = -1
    while True:
        out_loop += 1
        print('validating in model ', str(out_loop))
        model_path = '../models/model_'+str(out_loop)
        if not osp.exists(model_path): break
        model.load_state_dict(torch.load(model_path))
        fp = []
        tn = []
        auc = final_test(data.test_mask)
        print('fp and fn: ', len(fp), len(tn))
        _fp = 0
        _tp = 0
        eps = 1e-10
        tempNodeA = {}
        for i in nodeA:
            tempNodeA[i] = 1
        for i in fp:
            if i not in _nodeA:
                _fp += 1
            if i not in _neibor:
                continue
            for j in _neibor[i]:
                if j in tempNodeA:
                    tempNodeA[j] = 0
        for i in tempNodeA:
            if tempNodeA[i] == 0:
                _tp += 1
        print('Precision: ', _tp/(_tp+_fp+eps))
        print('Recall: ', _tp/len(nodeA))
        if (_tp/len(nodeA) > 0.8) and (_tp/(_tp+_fp+eps) > 0.7):
            while True:
                out_loop += 1
                model_path = '../models/model_'+str(out_loop)
                if not osp.exists(model_path):
                    break
                os.system('rm ../models/model_'+str(out_loop))
                os.system('rm ../models/tn_feature_label_'+str(graphId)+'_'+str(out_loop)+'.txt')
                os.system('rm ../models/fp_feature_label_'+str(graphId)+'_'+str(out_loop)+'.txt')
            return 1
        if (_tp/len(nodeA) <= 0.8):
            return 0
        for j in tn:
            data.test_mask[j] = False
    return 0

def train_pro():
    global data, nodeA, _nodeA, _neibor, b_size, feature_num, label_num, graphId
    global model, optimizer, device, fp, tn, loop_num, G
    os.system('python setup.py')
    path = '../dataset/darpatc/' + args.scene + '_train.txt'
    graphId = 0
    show('Start training graph ' + str(graphId))
    data1, feature_num, label_num, adj, adj2 = MyDataset(path, 0)
    dataset = TestDataset(data1)
    data = dataset[0]
    print(data)
    print('feature ', feature_num, '; label ', label_num)

    device = torch.device('cpu')
    model = HGNN_Attn(feature_num, label_num).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    # 构建 G 矩阵
    H = construct_H(data.edge_index, data.num_nodes)
    G_local = generate_G(H)
    G_local = G_local.to(device)
    global G
    G = G_local

    # 初始训练
    for epoch in range(1, 30):
        loss = train()
        auc = test(data.test_mask)
        show(epoch, loss, auc)

    loop_num = 0
    max_thre = 3
    bad_cnt = 0
    while True:
        fp = []
        tn = []
        auc = final_test(data.test_mask)
        if len(tn) == 0:
            bad_cnt += 1
        else:
            bad_cnt = 0
        if bad_cnt >= max_thre:
            break

        if len(tn) > 0:
            for i in tn:
                data.train_mask[i] = False
                data.test_mask[i] = False

            fw = open('../models/fp_feature_label_'+str(graphId)+'_'+str(loop_num)+'.txt', 'w')
            x_list = data.x[fp]
            y_list = data.y[fp]
            print(len(x_list))
            if len(x_list) >1:
                sorted_index = np.argsort(y_list, axis = 0)
                x_list = np.array(x_list)[sorted_index]
                y_list = np.array(y_list)[sorted_index]

            for i in range(len(y_list)):
                fw.write(str(y_list[i])+':')
                for j in x_list[i]:
                    fw.write(' '+str(j))
                fw.write('\n')
            fw.close()

            fw = open('../models/tn_feature_label_'+str(graphId)+'_'+str(loop_num)+'.txt', 'w')
            x_list = data.x[tn]
            y_list = data.y[tn]
            print(len(x_list))

            if len(x_list) >1:
                sorted_index = np.argsort(y_list, axis = 0)
                x_list = np.array(x_list)[sorted_index]
                y_list = np.array(y_list)[sorted_index]

            for i in range(len(y_list)):
                fw.write(str(y_list[i])+':')
                for j in x_list[i]:
                    fw.write(' '+str(j))
                fw.write('\n')
            fw.close()
            torch.save(model.state_dict(),'../models/model_'+str(loop_num))
            loop_num += 1
            if len(fp) == 0:
                break
        # 再次训练
        for epoch in range(1, 150):
            loss = train()
            auc = test(data.test_mask)
            show(epoch, loss, auc)
            if loss < 1:
                break
    show('Finish training graph ' + str(graphId))


def main():
    global b_size, args, thre
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='SAGE')  # 这里可以忽略
    parser.add_argument('--scene', type=str, default='cadets')
    args = parser.parse_args()
    assert args.scene in ['cadets','trace','theia','fivedirections']
    b_size = 5000
    thre = thre_map[args.scene]
    os.system('cp ../groundtruth/'+args.scene+'.txt groundtruth_uuid.txt')
    while True:
        train_pro()
        flag = validate()
        if flag == 1:
            break
        else:
            os.system('rm ../models/model_*')
            os.system('rm ../models/tn_feature_label_*')
            os.system('rm ../models/fp_feature_label_*')


if __name__ == "__main__":
    graphchi_root = os.path.abspath(os.path.join(os.getcwd(), '../graphchi-cpp-master'))
    os.environ['GRAPHCHI_ROOT'] = graphchi_root
    main()