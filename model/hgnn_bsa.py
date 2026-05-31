import math
import dgl
import dgl.function as fn
import torch
import torch.nn as nn
from dgl.utils import expand_as_pair

from utils.utils import create_activation


class BlockSelfAttention(nn.Module):
    """Chunked self-attention with a learnable residual gate.

    The gate keeps the encoder close to the original GAT at initialization and
    lets training decide how much block attention should affect node embeddings.
    """

    def __init__(self, dim_model, nhead=4, block_size=256, dropout=0.1, gate_init=0.0, scale=0.2):
        super().__init__()
        self.block_size = block_size
        self.scale = float(scale)
        self.input_norm = nn.LayerNorm(dim_model)
        self.encoder = nn.TransformerEncoderLayer(
            d_model=dim_model,
            nhead=nhead,
            dim_feedforward=dim_model * 2,
            batch_first=True,
            dropout=dropout,
            activation='gelu',
            norm_first=True
        )
        self.output_norm = nn.LayerNorm(dim_model)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, x):
        residual = x
        x = self.input_norm(x)
        if x.size(0) <= self.block_size:
            attended = self.encoder(x.unsqueeze(0)).squeeze(0)
        else:
            out = []
            n_chunks = math.ceil(x.size(0) / self.block_size)
            for chunk in torch.chunk(x, n_chunks, dim=0):
                out.append(self.encoder(chunk.unsqueeze(0)).squeeze(0))
            attended = torch.cat(out, dim=0)
        delta = self.output_norm(attended - x)
        return residual + self.scale * self.gate.tanh() * delta


class GAT(nn.Module):
    """
    DGL bipartite hypergraph encoder with optional Block Self-Attention.
    Keeps the original module interface for compatibility.
    """

    def __init__(self,
                 n_dim,
                 e_dim,
                 hidden_dim,
                 out_dim,
                 n_layers,
                 n_heads,
                 n_heads_out,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,
                 norm,
                 concat_out=False,
                 encoding=False,
                 hyper_k=4,
                 hyper_shuffle=False,
                 use_bsa=False,
                 bsa_heads=4,
                 bsa_block_size=256,
                 bsa_dropout=0.1,
                 bsa_gate_init=0.0,
                 bsa_scale=0.2,
                 ):
        super(GAT, self).__init__()
        self.out_dim = out_dim
        self.n_heads = n_heads
        self.n_heads_out = n_heads_out
        self.n_layers = n_layers
        self.gats = nn.ModuleList()
        self.concat_out = concat_out
        self.hyper_k = hyper_k
        self.hyper_shuffle = hyper_shuffle

        last_activation = create_activation(activation) if encoding else None
        last_residual = (encoding and residual)
        last_norm = norm if encoding else None

        if self.n_layers == 1:
            self.gats.append(GATConv(
                n_dim, e_dim, out_dim, n_heads_out, feat_drop, attn_drop, negative_slope,
                last_residual, norm=last_norm, concat_out=self.concat_out
            ))
        else:
            self.gats.append(GATConv(
                n_dim, e_dim, hidden_dim, n_heads, feat_drop, attn_drop, negative_slope,
                residual, create_activation(activation),
                norm=norm, concat_out=self.concat_out
            ))
            for _ in range(1, self.n_layers - 1):
                self.gats.append(GATConv(
                    hidden_dim * self.n_heads, e_dim, hidden_dim, n_heads,
                    feat_drop, attn_drop, negative_slope,
                    residual, create_activation(activation),
                    norm=norm, concat_out=self.concat_out
                ))
            self.gats.append(GATConv(
                hidden_dim * self.n_heads, e_dim, out_dim, n_heads_out,
                feat_drop, attn_drop, negative_slope,
                last_residual, last_activation, norm=last_norm, concat_out=self.concat_out
            ))

        bsa_dim = out_dim * n_heads_out if self.concat_out else out_dim
        if use_bsa:
            if bsa_dim % bsa_heads != 0:
                raise ValueError(f"BSA heads ({bsa_heads}) must divide feature dim ({bsa_dim}).")
            self.bsa = BlockSelfAttention(
                dim_model=bsa_dim,
                nhead=bsa_heads,
                block_size=bsa_block_size,
                dropout=bsa_dropout,
                gate_init=bsa_gate_init,
                scale=bsa_scale,
            )
        else:
            self.bsa = None

        self.head = nn.Identity()

    def _construct_khop_hypergraph(self, graph):
        num_nodes = graph.num_nodes()
        num_edges = graph.num_edges()

        if num_nodes == 0:
            hg = dgl.heterograph(
                {
                    ('node', 'inc', 'hyperedge'): (torch.tensor([], dtype=torch.int64), torch.tensor([], dtype=torch.int64)),
                    ('hyperedge', 'contains', 'node'): (torch.tensor([], dtype=torch.int64), torch.tensor([], dtype=torch.int64)),
                },
                num_nodes_dict={'node': 0, 'hyperedge': 0},
                device=graph.device,
            )
            return hg

        device = graph.device
        centers = torch.arange(num_nodes, dtype=torch.int64, device=device)

        if num_edges > 0 and self.hyper_k > 0:
            undirected = dgl.add_reverse_edges(graph, copy_ndata=False, copy_edata=False)
            sampled = dgl.sampling.sample_neighbors(
                undirected,
                centers,
                fanout=self.hyper_k,
                edge_dir='out',
                replace=False
            )
            he_ids, neighbor_nodes = sampled.edges()
            node_to_he_src = torch.cat([centers, neighbor_nodes], dim=0)
            node_to_he_dst = torch.cat([centers, he_ids], dim=0)
        else:
            node_to_he_src = centers
            node_to_he_dst = centers

        he_to_node_src = node_to_he_dst
        he_to_node_dst = node_to_he_src

        hg = dgl.heterograph(
            {
                ('node', 'inc', 'hyperedge'): (node_to_he_src, node_to_he_dst),
                ('hyperedge', 'contains', 'node'): (he_to_node_src, he_to_node_dst),
            },
            num_nodes_dict={'node': num_nodes, 'hyperedge': num_nodes},
            device=device,
        )

        edge_attr = graph.edata['attr']
        if num_edges == 0:
            he_attr = torch.zeros(num_nodes, edge_attr.size(-1), device=device, dtype=edge_attr.dtype)
        else:
            he_attr = torch.zeros(num_nodes, edge_attr.size(-1), device=device, dtype=edge_attr.dtype)
            degrees = torch.zeros(num_nodes, device=device, dtype=torch.float32)
            src_e, dst_e = graph.edges()
            eids = torch.arange(num_edges, device=device)

            he_attr.index_add_(0, src_e, edge_attr[eids])
            he_attr.index_add_(0, dst_e, edge_attr[eids])
            degrees.index_add_(0, src_e, torch.ones_like(src_e, dtype=torch.float32))
            degrees.index_add_(0, dst_e, torch.ones_like(dst_e, dtype=torch.float32))
            he_attr = he_attr / degrees.clamp_min(1.0).unsqueeze(-1)

        hg.nodes['hyperedge'].data['e_attr'] = he_attr
        return hg

    def forward(self, g, input_feature, return_hidden=False):
        cache_key = f"_cached_hg_k{self.hyper_k}"
        if (not self.hyper_shuffle) and hasattr(g, cache_key):
            hg = getattr(g, cache_key)
        else:
            hg = self._construct_khop_hypergraph(g)
            if not self.hyper_shuffle:
                setattr(g, cache_key, hg)
        h = input_feature
        hidden_list = []
        for layer in range(self.n_layers):
            h = self.gats[layer](hg, h)
            hidden_list.append(h)

        if self.bsa is not None:
            h = self.bsa(h)
            hidden_list[-1] = h

        if return_hidden:
            return self.head(h), hidden_list
        return self.head(h)

    def reset_classifier(self, num_classes):
        out_dim = self.out_dim * self.n_heads_out if self.concat_out else self.out_dim
        self.head = nn.Linear(out_dim, num_classes)


class GATConv(nn.Module):
    """Bipartite hypergraph convolution on node-hyperedge incidence graph."""

    def __init__(self,
                 in_dim,
                 e_dim,
                 out_dim,
                 n_heads,
                 feat_drop=0.0,
                 attn_drop=0.0,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=False,
                 bias=True,
                 norm=None,
                 concat_out=True):
        super(GATConv, self).__init__()
        self.n_heads = n_heads
        self.src_feat, self.dst_feat = expand_as_pair(in_dim)
        self.edge_feat = e_dim
        self.out_feat = out_dim
        self.allow_zero_in_degree = allow_zero_in_degree
        self.concat_out = concat_out

        hidden_size = self.n_heads * self.out_feat
        self.node_self_fc = nn.Linear(self.src_feat, hidden_size, bias=False)
        self.node_to_edge_fc = nn.Linear(self.src_feat, hidden_size, bias=False)
        self.edge_attr_fc = nn.Linear(self.edge_feat, hidden_size, bias=False)
        self.edge_to_node_fc = nn.Linear(hidden_size, hidden_size, bias=False)

        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        if bias:
            self.bias = nn.Parameter(torch.zeros(1, hidden_size))
        else:
            self.register_buffer('bias', None)

        if residual:
            if self.dst_feat != hidden_size:
                self.res_fc = nn.Linear(self.dst_feat, hidden_size, bias=False)
            else:
                self.res_fc = nn.Identity()
        else:
            self.register_buffer('res_fc', None)

        self.activation = activation
        self.norm = norm(hidden_size) if norm is not None else None
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.node_self_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.node_to_edge_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.edge_attr_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.edge_to_node_fc.weight, gain=gain)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self.allow_zero_in_degree = set_value

    def forward(self, hypergraph, feat, get_attention=False):
        del get_attention  # kept for signature compatibility
        node_input = self.feat_drop(feat)

        if hypergraph.num_nodes('hyperedge') == 0:
            rst = self.node_self_fc(node_input)
            if self.bias is not None:
                rst = rst + self.bias
            if self.res_fc is not None:
                rst = rst + self.res_fc(feat)
            if not self.concat_out:
                rst = rst.view(-1, self.n_heads, self.out_feat).mean(dim=1)
            if self.norm is not None:
                rst = self.norm(rst)
            if self.activation:
                rst = self.activation(rst)
            return rst

        hypergraph.nodes['node'].data['h'] = self.node_to_edge_fc(node_input)
        hypergraph.nodes['hyperedge'].data['e_proj'] = self.edge_attr_fc(
            self.feat_drop(hypergraph.nodes['hyperedge'].data['e_attr'])
        )

        hypergraph['inc'].update_all(fn.copy_u('h', 'm'), fn.mean('m', 'node_aggr'))
        edge_state = hypergraph.nodes['hyperedge'].data['e_proj'] + hypergraph.nodes['hyperedge'].data['node_aggr']
        edge_state = self.leaky_relu(edge_state)
        edge_state = self.attn_drop(edge_state)
        hypergraph.nodes['hyperedge'].data['he'] = edge_state

        hypergraph['contains'].update_all(fn.copy_u('he', 'm'), fn.mean('m', 'edge_aggr'))
        node_state = self.node_self_fc(node_input) + self.edge_to_node_fc(hypergraph.nodes['node'].data['edge_aggr'])

        if self.bias is not None:
            node_state = node_state + self.bias

        if self.res_fc is not None:
            node_state = node_state + self.res_fc(feat)

        if self.concat_out:
            rst = node_state
        else:
            rst = node_state.view(-1, self.n_heads, self.out_feat).mean(dim=1)

        if self.norm is not None:
            rst = self.norm(rst)
        if self.activation:
            rst = self.activation(rst)
        return rst


class HGNNConv(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_dim))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        nn.init.uniform_(self.weight, -stdv, stdv)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -stdv, stdv)

    def forward(self, x, incidence):
        h = x @ self.weight
        if self.bias is not None:
            h = h + self.bias
        return incidence.matmul(incidence.transpose_matmul(h))


class HypergraphEncoder(nn.Module):
    def __init__(self, in_dim, hid_dim=64, num_layers=2, dropout=0.5):
        super().__init__()
        assert num_layers in (1, 2, 3)
        self.conv1 = HGNNConv(in_dim, hid_dim)
        self.conv2 = HGNNConv(hid_dim, hid_dim) if num_layers >= 2 else None
        self.conv3 = HGNNConv(hid_dim, hid_dim) if num_layers == 3 else None
        self.dropout = dropout

    def forward(self, x, incidence):
        x = torch.relu(self.conv1(x, incidence))
        if self.conv2 is not None:
            x = torch.dropout(x, p=self.dropout, train=self.training)
            x = torch.relu(self.conv2(x, incidence))
        if self.conv3 is not None:
            x = torch.dropout(x, p=self.dropout, train=self.training)
            x = torch.relu(self.conv3(x, incidence))
        return x


class TemporalHyperBlock(nn.Module):
    def __init__(self, dim_model, nhead=4, dropout=0.1):
        super().__init__()
        self.time_proj = nn.Sequential(
            nn.Linear(1, dim_model),
            nn.SiLU(),
            nn.Linear(dim_model, dim_model),
        )
        self.pos_proj = nn.Sequential(
            nn.Linear(1, dim_model),
            nn.SiLU(),
            nn.Linear(dim_model, dim_model),
        )
        self.encoder = nn.TransformerEncoderLayer(
            d_model=dim_model,
            nhead=nhead,
            dim_feedforward=dim_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )

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
        encoded = self.encoder((h + time_bias + pos_bias).unsqueeze(0)).squeeze(0)
        return h + encoded


class TemporalHypergraphAutoencoder(nn.Module):
    def __init__(
        self,
        in_dim,
        hid_dim=64,
        nhead=4,
        block_size=256,
        dropout=0.5,
        num_layers=2,
        use_bsa=True,
        use_temporal=True,
        use_norm=False,
    ):
        super().__init__()
        self.use_bsa = use_bsa
        self.use_temporal = use_temporal
        self.hgnn = HypergraphEncoder(in_dim, hid_dim, num_layers=num_layers, dropout=dropout)
        if use_bsa:
            self.bsa = BlockSelfAttention(hid_dim, nhead, block_size, dropout)
        if use_temporal:
            self.temporal = TemporalHyperBlock(hid_dim, nhead, dropout)
        self.decoder = nn.Linear(hid_dim, in_dim)
        self.dropout = dropout
        self.norm = nn.LayerNorm(hid_dim) if use_norm else None

    def forward(self, x, incidence, batch_index=None, timestamp=None, is_flow=None, edge_index=None):
        h = self.hgnn(x, incidence)
        if self.use_bsa:
            h = self.bsa(h)
        if self.use_temporal and timestamp is not None and is_flow is not None:
            h = self._apply_temporal(h, timestamp, is_flow, batch_index)
        h = torch.dropout(h, p=self.dropout, train=self.training)
        if self.norm is not None:
            h = self.norm(h)
        return h, self.decoder(h)

    def _apply_temporal(self, h, timestamp, is_flow, batch_index):
        if h.size(0) == 0:
            return h
        device = h.device
        if batch_index is None:
            node_groups = [torch.arange(h.size(0), device=device)]
        else:
            total_graphs = int(batch_index.max().item()) + 1
            node_groups = [(batch_index == i).nonzero(as_tuple=False).view(-1) for i in range(total_graphs)]
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
            out[ordered] = self.temporal(out[ordered], flow_ts[order])
        return out


def build_temporal_hypergraph_autoencoder(
    in_dim: int,
    hid_dim: int = 128,
    layers: int = 2,
    use_bsa: bool = True,
    use_temporal: bool = True,
    use_norm: bool = False,
    dropout: float = 0.2,
    bsa_heads: int = 4,
    bsa_block_size: int = 512,
) -> TemporalHypergraphAutoencoder:
    return TemporalHypergraphAutoencoder(
        in_dim,
        hid_dim,
        nhead=bsa_heads,
        block_size=bsa_block_size,
        dropout=dropout,
        num_layers=layers,
        use_bsa=use_bsa,
        use_temporal=use_temporal,
        use_norm=use_norm,
    )
