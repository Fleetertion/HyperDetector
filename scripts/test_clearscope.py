#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
infer_clearscope_khop.py
- 与训练一致：k 邻域超图（k=4, shuffle=False）
- 逐轮加载 ../models/model_k 推理，把 TN 从 test_mask 移除
- 输出 alarm_clearscope.txt（剩余 test_mask 及 1/2 跳邻居）
"""
import os, os.path as osp, argparse, time, math, random
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
from torch_geometric.data import Data
from torch.nn import TransformerEncoderLayer

def show(msg): print(f"{msg} {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

# ---------- 数据加载（与训练一致） ----------
def load_clearscope_data(benign_dir, attack_dir, benign_ids, attack_ids, feature_num=4):
    nodes={}; node_feats={}; node_labs={}; edges=[]
    def gid(k):
        if k not in nodes:
            idx=len(nodes); nodes[k]=idx; node_feats[idx]=[]; node_labs[idx]=[]
        return nodes[k]
    def add_line(aid, atype, obj_token, lab):
        spx=obj_token.split(':')
        if len(spx)!=4: return
        oid,obj,act,ts=spx
        a=gid(aid); b=gid(oid)
        edges.append([a,b]); edges.append([b,a])
        try: f=[float(atype),float(obj),float(act),float(ts)]
        except: f=[0.0,0.0,0.0,0.0]
        f=np.array(f,dtype=np.float32)
        node_feats[a].append(f); node_labs[a].append(lab)
        node_feats[b].append(f); node_labs[b].append(lab)
    for fid in benign_ids:
        fp=osp.join(benign_dir,f"clearscope-benign_{fid}.txt")
        if not osp.exists(fp): print(f"[Warn] missing: {fp}"); continue
        for ln in open(fp,'r',encoding='utf-8'):
            s=ln.strip().split()
            if len(s)>=3: add_line(s[0],s[1],s[2],0)
    for fid in attack_ids:
        fp=osp.join(attack_dir,f"clearscope-e3-attack_{fid}.txt")
        if not osp.exists(fp): print(f"[Warn] missing: {fp}"); continue
        for ln in open(fp,'r',encoding='utf-8'):
            s=ln.strip().split()
            if len(s)>=3: add_line(s[0],s[1],s[2],1)
    N=len(nodes)
    if N==0:
        x=torch.zeros((0,feature_num),dtype=torch.float); y=torch.zeros((0,),dtype=torch.long)
        ei=torch.zeros((2,0),dtype=torch.long)
    else:
        xs,ys=[],[]
        for i in range(N):
            xs.append(np.mean(np.stack(node_feats[i],0),0) if node_feats[i] else np.zeros(feature_num,dtype=np.float32))
            ys.append(int(np.bincount(np.array(node_labs[i],int)).argmax()) if node_labs[i] else 0)
        x=torch.tensor(np.stack(xs,0),dtype=torch.float)
        y=torch.tensor(ys,dtype=torch.long)
        ei=torch.tensor(edges,dtype=torch.long).t().contiguous() if len(edges)>0 else torch.zeros((2,0),dtype=torch.long)
    idx=np.arange(N); np.random.shuffle(idx)
    split=int(0.8*N) if N>0 else 0
    train_mask=torch.zeros(N,dtype=torch.bool); test_mask=torch.zeros(N,dtype=torch.bool)
    if N>0: train_mask[idx[:split]]=True; test_mask[idx[split:]]=True
    data=Data(x=x,y=y,edge_index=ei,train_mask=train_mask,test_mask=test_mask); data.num_nodes=N
    return data, x.size(1) if N>0 else 4, int(y.max().item()+1) if y.numel()>0 else 2

def standardize_features_(data: Data):
    if data.x is None or data.x.numel()==0: return
    with torch.no_grad():
        mu=data.x.mean(0,keepdim=True); sigma=data.x.std(0,keepdim=True).clamp_min(1e-6)
        data.x=(data.x-mu)/sigma

# ---------- k-邻域构图（关键） ----------
def construct_H_khop(edge_index, num_nodes, k: int = 4, shuffle: bool = False):
    if edge_index.numel()==0 or num_nodes==0:
        return sp.coo_matrix((num_nodes,0))
    ei=edge_index.cpu().numpy()
    adj=[[] for _ in range(num_nodes)]
    for u,v in ei.T:
        adj[u].append(v); adj[v].append(u)
    rows,cols=[],[]; e_id=0
    for c in range(num_nodes):
        nbrs=adj[c]
        if len(nbrs)>k:
            nbrs=random.sample(nbrs,k) if shuffle else nbrs[:k]
        nodes=[c]+nbrs
        rows.extend(nodes); cols.extend([e_id]*len(nodes)); e_id+=1
    data=np.ones(len(rows),dtype=np.float32)
    return sp.coo_matrix((data,(rows,cols)),shape=(num_nodes,e_id))

def generate_G(H):
    H=H.tocoo()
    if H.shape[1]==0:
        i=torch.arange(H.shape[0])
        return torch.sparse_coo_tensor(torch.stack([i,i]), torch.ones_like(i,dtype=torch.float),(H.shape[0],H.shape[0])).coalesce()
    idx=np.vstack((H.row,H.col))
    Ht=torch.sparse_coo_tensor(torch.LongTensor(idx), torch.FloatTensor(H.data), torch.Size(H.shape)).coalesce()
    HT=Ht.transpose(0,1).coalesce()
    DV=torch.sparse.sum(Ht,dim=1).to_dense()+1e-10
    DE=torch.sparse.sum(Ht,dim=0).to_dense()+1e-10
    DV_inv_sqrt=torch.pow(DV,-0.5); DE_inv=torch.pow(DE,-1.0)
    H_idx=Ht.indices(); H_val=Ht.values()
    H_DE=torch.sparse_coo_tensor(H_idx, H_val*DE_inv[H_idx[1]], Ht.size()).coalesce()
    temp=torch.sparse.mm(H_DE, HT).coalesce()
    ti,tv=temp.indices(),temp.values()
    vr=DV_inv_sqrt[ti[0]]; vc=DV_inv_sqrt[ti[1]]
    return torch.sparse_coo_tensor(ti, tv*vr*vc, temp.size()).coalesce()

# ---------- 模型 ----------
class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super().__init__()
        self.weight=nn.Parameter(torch.Tensor(in_ft,out_ft))
        self.bias=nn.Parameter(torch.Tensor(out_ft)) if bias else None
        self.reset_parameters()
    def reset_parameters(self):
        stdv=1.0/math.sqrt(self.weight.size(1))
        nn.init.uniform_(self.weight,-stdv,stdv)
        if self.bias is not None: nn.init.uniform_(self.bias,-stdv,stdv)
    def forward(self,x,G):
        x=torch.sparse.mm(G,x); x=x.matmul(self.weight)
        if self.bias is not None: x=x+self.bias
        return x

class BlockSelfAttention(nn.Module):
    def __init__(self, dim_model, nhead=4, block_size=256, dropout=0.1):
        super().__init__()
        self.block_size=block_size
        self.enc=TransformerEncoderLayer(d_model=dim_model,nhead=nhead,
                                         dim_feedforward=dim_model*2,batch_first=True,
                                         dropout=dropout,activation='gelu')
    def forward(self,x):
        if x.size(0)<=self.block_size: return self.enc(x.unsqueeze(0)).squeeze(0)
        ch=torch.chunk(x, math.ceil(x.size(0)/self.block_size), 0)
        return torch.cat([self.enc(c.unsqueeze(0)).squeeze(0) for c in ch], 0)

class HGNN_Attn(nn.Module):
    def __init__(self,in_ch,n_class,n_hid=32,d_model=64,nhead=4,dropout=0.5,block_size=256,attn_dropout=0.1):
        super().__init__()
        self.hgc1=HGNN_conv(in_ch,n_hid,True)
        self.hgc2=HGNN_conv(n_hid,d_model,True)
        self.attn=BlockSelfAttention(d_model,nhead,block_size,attn_dropout)
        self.cls=nn.Linear(d_model,n_class)
        self.drop=dropout
    def forward(self,x,G):
        x=F.relu(self.hgc1(x,G)); x=F.dropout(x,self.drop,self.training)
        x=F.relu(self.hgc2(x,G)); x=self.attn(x); x=self.cls(x)
        return F.log_softmax(x,dim=1)

# ---------- 1/2 跳邻居（用于输出） ----------
def build_adj_lists(edge_index, num_nodes):
    adj1=[[] for _ in range(num_nodes)]
    for u,v in edge_index.t().tolist():
        adj1[u].append(v)
    adj2=[[] for _ in range(num_nodes)]
    for u in range(num_nodes):
        s=set()
        for v in adj1[u]:
            s.add(v)
            for w in adj1[v]: s.add(w)
        if u in s: s.remove(u)
        adj2[u]=list(s)
    return adj1, adj2

# ---------- 不确定性推理 ----------
def test_step(model, data, G, thre, mask):
    model.eval()
    with torch.no_grad():
        out=model(data.x,G)
        probs=F.softmax(out[mask],dim=1)
        if probs.size(0)==0: return 0.0, [], []
        top1=probs.max(1)
        tmp=probs.clone(); tmp[torch.arange(tmp.size(0)), top1[1]]=-1
        top2=tmp.max(1)
        ratio=top1[0]/(top2[0]+1e-12)
        pred=top1[1]; pred[ratio<thre]=100
        correct=pred.eq(data.y[mask])
        idx=mask.nonzero(as_tuple=False).view(-1)
        tn=idx[correct].tolist(); fp=idx[~correct].tolist()
        return correct.float().mean().item(), fp, tn

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--benign_dir',type=str,default="../graphchi-cpp-master/graph_data/clearscope/benign")
    parser.add_argument('--attack_dir',type=str,default="../graphchi-cpp-master/graph_data/clearscope/attack")
    parser.add_argument('--benign_ids',type=int,nargs='+',default=[2,3])
    parser.add_argument('--attack_ids',type=int,nargs='+',default=[1])
    parser.add_argument('--k',type=int,default=4)
    parser.add_argument('--shuffle',action='store_true')
    parser.add_argument('--thre',type=float,default=1.0)
    parser.add_argument('--max_models',type=int,default=100)
    args=parser.parse_args()

    # data
    show("Start testing (clearscope, k-hop)")
    data, feat_dim, n_class = load_clearscope_data(args.benign_dir,args.attack_dir,args.benign_ids,args.attack_ids)
    standardize_features_(data)

    # G
    H=construct_H_khop(data.edge_index, data.num_nodes, k=args.k, shuffle=args.shuffle)
    G=generate_G(H)

    # model
    device=torch.device('cpu')
    model=HGNN_Attn(in_ch=feat_dim, n_class=n_class).to(device)

    # 邻接（输出用）
    adj1,adj2=build_adj_lists(data.edge_index, data.num_nodes)

    loop=0; best=0.0; tried=False
    while loop<=args.max_models:
        mpath=f'../models/model_{loop}'
        if not osp.exists(mpath):
            loop+=1; continue
        tried=True
        model.load_state_dict(torch.load(mpath, map_location=device))
        acc, fp, tn = test_step(model, data, G, args.thre, data.test_mask)
        show(f"model_{loop:02d} acc={acc:.4f} fp={len(fp)} tn={len(tn)}")
        best=max(best,acc)
        if tn: data.test_mask[tn]=False
        if acc>=1.0 or data.test_mask.sum().item()==0: break
        loop+=1

    # 输出
    out_path='alarm_clearscope.txt'
    with open(out_path,'w') as fw:
        rest=data.test_mask.sum().item(); fw.write(f"{rest}\n")
        for idx in data.test_mask.nonzero(as_tuple=False).view(-1).tolist():
            hop=set(adj1[idx])|set(adj2[idx])
            fw.write('\n'+str(idx)+':')
            if hop: fw.write(' '+' '.join(map(str,sorted(hop))))
    show(f"Finish testing, best_acc={best:.4f}, alarms={data.test_mask.sum().item()}, saved to {out_path}")
    if not tried: print("[Warn] 未找到 ../models/model_k，请先完成训练保存模型。")

if __name__=="__main__":
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    main()