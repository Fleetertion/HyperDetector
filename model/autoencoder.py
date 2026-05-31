from .hgnn_bsa import GAT
from utils.utils import create_norm
from functools import partial
from itertools import chain
from .loss_func import sce_loss
import torch
import torch.nn as nn
import dgl
import random


def build_model(args):
    num_hidden = args.num_hidden
    num_layers = args.num_layers
    negative_slope = args.negative_slope
    mask_rate = args.mask_rate
    alpha_l = args.alpha_l
    n_dim = args.n_dim
    e_dim = args.e_dim

    model = GMAEModel(
        n_dim=n_dim,
        e_dim=e_dim,
        hidden_dim=num_hidden,
        n_layers=num_layers,
        n_heads=4,
        activation="prelu",
        feat_drop=0.1,
        negative_slope=negative_slope,
        residual=True,
        norm='BatchNorm',
        mask_rate=mask_rate,
        loss_fn='sce',
        alpha_l=alpha_l,
        hyper_k=args.hyper_k,
        hyper_shuffle=args.hyper_shuffle,
        use_bsa=args.use_bsa,
        bsa_heads=args.bsa_heads,
        bsa_block_size=args.bsa_block_size,
        bsa_dropout=args.bsa_dropout,
        bsa_gate_init=args.bsa_gate_init,
        bsa_scale=args.bsa_scale
    )
    return model


class GMAEModel(nn.Module):
    def __init__(self, n_dim, e_dim, hidden_dim, n_layers, n_heads, activation,
                 feat_drop, negative_slope, residual, norm, mask_rate=0.5, loss_fn="sce", alpha_l=2,
                 hyper_k=4, hyper_shuffle=False, use_bsa=False, bsa_heads=4, bsa_block_size=256,
                 bsa_dropout=0.1, bsa_gate_init=0.0, bsa_scale=0.2):
        super(GMAEModel, self).__init__()
        self._mask_rate = mask_rate
        self._output_hidden_size = hidden_dim
        self.recon_loss = nn.BCELoss(reduction='mean')
        self._warned_skip_struct = False

        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
                nn.init.constant_(m.bias, 0)

        self.edge_recon_fc = nn.Sequential(
            nn.Linear(hidden_dim * n_layers * 2, hidden_dim),
            nn.LeakyReLU(negative_slope),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.edge_recon_fc.apply(init_weights)

        assert hidden_dim % n_heads == 0
        enc_num_hidden = hidden_dim // n_heads
        enc_nhead = n_heads

        dec_in_dim = hidden_dim
        dec_num_hidden = hidden_dim

        # build encoder
        self.encoder = GAT(
            n_dim=n_dim,
            e_dim=e_dim,
            hidden_dim=enc_num_hidden,
            out_dim=enc_num_hidden,
            n_layers=n_layers,
            n_heads=enc_nhead,
            n_heads_out=enc_nhead,
            concat_out=True,
            activation=activation,
            feat_drop=feat_drop,
            attn_drop=0.0,
            negative_slope=negative_slope,
            residual=residual,
            norm=create_norm(norm),
            encoding=True,
            hyper_k=hyper_k,
            hyper_shuffle=hyper_shuffle,
            use_bsa=use_bsa,
            bsa_heads=bsa_heads,
            bsa_block_size=bsa_block_size,
            bsa_dropout=bsa_dropout,
            bsa_gate_init=bsa_gate_init,
            bsa_scale=bsa_scale,
        )

        # build decoder for attribute prediction
        self.decoder = GAT(
            n_dim=dec_in_dim,
            e_dim=e_dim,
            hidden_dim=dec_num_hidden,
            out_dim=n_dim,
            n_layers=1,
            n_heads=n_heads,
            n_heads_out=1,
            concat_out=True,
            activation=activation,
            feat_drop=feat_drop,
            attn_drop=0.0,
            negative_slope=negative_slope,
            residual=residual,
            norm=create_norm(norm),
            encoding=False,
            hyper_k=hyper_k,
            hyper_shuffle=hyper_shuffle,
            use_bsa=False,
        )

        self.enc_mask_token = nn.Parameter(torch.zeros(1, n_dim))
        self.encoder_to_decoder = nn.Linear(dec_in_dim * n_layers, dec_in_dim, bias=False)

        # * setup loss function
        self.criterion = self.setup_loss_fn(loss_fn, alpha_l)

    @property
    def output_hidden_dim(self):
        return self._output_hidden_size

    def setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion

    def encoding_mask_noise(self, g, mask_rate=0.3):
        num_nodes = g.num_nodes()
        perm = torch.randperm(num_nodes, device=g.device)

        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]
        masked_x = g.ndata["attr"].clone()
        masked_x[mask_nodes] = self.enc_mask_token
        return masked_x, (mask_nodes, keep_nodes)

    def forward(self, g):
        loss = self.compute_loss(g)
        return loss

    def compute_loss(self, g):
        # Feature Reconstruction
        masked_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(g, self._mask_rate)
        enc_rep, all_hidden = self.encoder(g, masked_x.to(g.device), return_hidden=True)
        enc_rep = torch.cat(all_hidden, dim=1)
        rep = self.encoder_to_decoder(enc_rep)

        recon = self.decoder(g, rep)
        x_init = g.ndata['attr'][mask_nodes]
        x_rec = recon[mask_nodes]
        loss = self.criterion(x_rec, x_init)

        # Structural Reconstruction
        n_nodes = g.num_nodes()
        n_edges = g.number_of_edges()
        too_large = n_edges > 2_000_000
        too_dense = n_nodes > 1 and n_edges > (n_nodes * (n_nodes - 1) // 2)
        if too_large or too_dense:
            if not self._warned_skip_struct:
                print(
                    f"[train] skip structural reconstruction on dense/large graph: "
                    f"nodes={n_nodes} edges={n_edges}"
                )
                self._warned_skip_struct = True
            return loss

        threshold = min(10000, n_nodes, n_edges)
        if threshold <= 0:
            return loss

        negative_edge_pairs = dgl.sampling.global_uniform_negative_sampling(g, threshold)
        if len(negative_edge_pairs[0]) == 0:
            return loss
        threshold = min(threshold, len(negative_edge_pairs[0]))

        positive_edge_pairs = random.sample(range(n_edges), threshold)
        positive_edge_pairs = (g.edges()[0][positive_edge_pairs], g.edges()[1][positive_edge_pairs])
        negative_edge_pairs = (negative_edge_pairs[0][:threshold], negative_edge_pairs[1][:threshold])
        sample_src = enc_rep[torch.cat([positive_edge_pairs[0], negative_edge_pairs[0]])].to(g.device)
        sample_dst = enc_rep[torch.cat([positive_edge_pairs[1], negative_edge_pairs[1]])].to(g.device)
        y_pred = self.edge_recon_fc(torch.cat([sample_src, sample_dst], dim=-1)).squeeze(-1)
        y = torch.cat([torch.ones(threshold), torch.zeros(threshold)]).to(g.device)
        loss += self.recon_loss(y_pred, y)
        return loss

    def embed(self, g):
        x = g.ndata['attr'].to(g.device)
        rep = self.encoder(g, x)
        return rep

    @property
    def enc_params(self):
        return self.encoder.parameters()

    @property
    def dec_params(self):
        return chain(*[self.encoder_to_decoder.parameters(), self.decoder.parameters()])
