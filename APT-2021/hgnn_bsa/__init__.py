"""
hgnn_bsa: HyperGraph Neural Net + Block Self‑Attention
"""

import math, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.nn import GCNConv

# ────────────────── HGNN 基元 ──────────────────
class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super().__init__()
        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        nn.init.uniform_(self.weight, -stdv, stdv)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -stdv, stdv)

    def forward(self, x, H):          # H: torch_sparse.SparseTensor  (|V| × |E|)
        x = x @ self.weight
        if self.bias is not None:
            x = x + self.bias
        # H  H^T  implements high‑order aggregation
        return H @ (H.t() @ x)        # 或 H.transpose() @ x

class HGNN(nn.Module):
    def __init__(self, in_ch, hid_ch=64, num_layers=2, dropout=0.5):
        super().__init__()
        assert num_layers in (1, 2)
        self.conv1 = HGNN_conv(in_ch, hid_ch)
        self.conv2 = HGNN_conv(hid_ch, hid_ch) if num_layers == 2 else None
        self.dp = dropout

    def forward(self, x, H):
        x = F.relu(self.conv1(x, H))
        if self.conv2 is not None:
            x = F.dropout(x, p=self.dp, training=self.training)
            x = F.relu(self.conv2(x, H))
        return x                               # [N, hid]

# --- Block Self-Attention -------------------------------------------------
class BlockSelfAttention(nn.Module):
    def __init__(self, d_model, nhead=4, block_size=256, dropout=0.1):
        super().__init__()
        from torch.nn import TransformerEncoderLayer
        self.block_size = block_size
        self.attn = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True, activation='gelu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(0) <= self.block_size:
            return self.attn(x.unsqueeze(0)).squeeze(0)
        chunks = torch.chunk(x, math.ceil(x.size(0) / self.block_size), dim=0)
        outs = [self.attn(c.unsqueeze(0)).squeeze(0) for c in chunks]
        return torch.cat(outs, dim=0)


class TemporalHyperBlock(nn.Module):
    """Temporal self-attention over flow nodes ordered by timestamp."""

    def __init__(self, d_model, nhead=4, dropout=0.1):
        super().__init__()
        from torch.nn import TransformerEncoderLayer
        self.time_proj = nn.Sequential(
            nn.Linear(1, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        self.pos_proj = nn.Sequential(
            nn.Linear(1, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        self.encoder = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation='gelu')

    def forward(self, h: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
        if h.numel() == 0:
            return h
        norm_ts = ts - ts.min()
        span = norm_ts.max()
        if span <= 1e-6:
            norm_ts = torch.linspace(0, 1, steps=ts.size(0), device=ts.device, dtype=ts.dtype)
        else:
            norm_ts = norm_ts / span
        time_bias = self.time_proj(norm_ts.unsqueeze(-1))
        pos = torch.arange(ts.size(0), device=ts.device, dtype=ts.dtype)
        if pos.size(0) > 1:
            pos = pos / (pos.size(0) - 1)
        else:
            pos = pos.zero_()
        pos_bias = self.pos_proj(pos.unsqueeze(-1))
        seq = h + time_bias + pos_bias
        encoded = self.encoder(seq.unsqueeze(0)).squeeze(0)
        return h + encoded


class HGNN_Attn(nn.Module):
    def __init__(self, in_ch, hid_ch=64, nhead=4, block_size=256,
                 dropout=0.5, num_layers=2, use_bsa=True,
                 use_temporal=True, use_norm=False):
        super().__init__()
        self.use_bsa = use_bsa
        self.use_temporal = use_temporal
        self.hgnn = HGNN(in_ch, hid_ch, num_layers=num_layers, dropout=dropout)
        if use_bsa:
            self.bsa = BlockSelfAttention(hid_ch, nhead, block_size, dropout)
        if use_temporal:
            self.temporal = TemporalHyperBlock(hid_ch, nhead, dropout)
        # 仅用于自监督重建；外部可另行定义解码器
        self.decoder = nn.Linear(hid_ch, in_ch)
        self.dp = dropout
        self.norm = nn.LayerNorm(hid_ch) if use_norm else None

    def forward(self, x, H, batch_index=None, timestamp=None, is_flow=None, edge_index=None):
        h = self.hgnn(x, H)
        if self.use_bsa:
            h = self.bsa(h)
        if self.use_temporal and timestamp is not None and is_flow is not None:
            h = self._apply_temporal(h, timestamp, is_flow, batch_index)
        h = F.dropout(h, p=self.dp, training=self.training)
        if self.norm is not None:
            h = self.norm(h)
        return h, self.decoder(h)              # (node emb, 重建)

    def _apply_temporal(self, h, timestamp, is_flow, batch_index):
        if h.size(0) == 0:
            return h
        device = h.device
        if batch_index is None:
            node_groups = [torch.arange(h.size(0), device=device)]
        else:
            total_graphs = int(batch_index.max().item()) + 1
            node_groups = [(batch_index == i).nonzero(as_tuple=False).view(-1)
                           for i in range(total_graphs)]
        out = h.clone()
        for idx in node_groups:
            if idx.numel() == 0:
                continue
            flow_idx = idx[is_flow[idx]]
            if flow_idx.numel() <= 1:
                continue
            flow_ts = timestamp[flow_idx].float()
            order = torch.argsort(flow_ts, dim=0)
            ordered = flow_idx[order]
            enhanced = self.temporal(out[ordered], flow_ts[order])
            out[ordered] = enhanced
        return out


def build_model(in_dim: int,
                hid_dim: int = 128,
                layers: int = 2,
                use_bsa: bool = True,
                use_temporal: bool = True,
                use_norm: bool = False) -> HGNN_Attn:
    return HGNN_Attn(in_dim, hid_dim, nhead=4,
                     block_size=512, dropout=0.2,
                     num_layers=layers, use_bsa=use_bsa,
                     use_temporal=use_temporal, use_norm=use_norm)


class PairwiseGNN(nn.Module):
    """Baseline pairwise GNN without hyperedges or global attention."""

    def __init__(self, in_ch, hid_ch=64, layers=2, dropout=0.2):
        super().__init__()
        assert layers >= 1
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_ch, hid_ch))
        for _ in range(1, layers):
            self.convs.append(GCNConv(hid_ch, hid_ch))
        self.decoder = nn.Linear(hid_ch, in_ch)
        self.dp = dropout

    def forward(self, x, incidence=None, batch_index=None, timestamp=None,
                is_flow=None, edge_index=None):
        if edge_index is None:
            raise ValueError("edge_index is required for PairwiseGNN")
        h = x
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dp, training=self.training)
        return h, self.decoder(h)


def build_pairwise_model(in_dim: int,
                         hid_dim: int = 128,
                         layers: int = 2) -> PairwiseGNN:
    return PairwiseGNN(in_dim, hid_ch=hid_dim, layers=layers, dropout=0.2)
