# Part of the code here is adapted from: https://github.com/A4Bio/ProteinInvBench

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum, scatter_softmax, scatter_mean


def enable_opt_einsum_backend(strategy: str = "auto"):
    """Safely enable PyTorch's opt_einsum backend if available.

    This does not change any parameter names/shapes, and therefore does not
    affect checkpoint compatibility.
    """
    backend = getattr(torch.backends, "opt_einsum", None)
    if backend is None:
        return False
    if not backend.is_available():
        return False
    backend.enabled = True
    backend.strategy = strategy
    return True


def rbf(values, v_min, v_max, n_bins=16):
    rbf_centers = torch.linspace(v_min, v_max, n_bins, device=values.device, dtype=values.dtype)
    rbf_centers = rbf_centers.view([1] * len(values.shape) + [-1])
    rbf_std = (v_max - v_min) / n_bins
    z = (values.unsqueeze(-1) - rbf_centers) / rbf_std
    return torch.exp(-torch.pow(z, 2))


class FourierEmbedding(nn.Module):
    def __init__(self, dim):
        super(FourierEmbedding, self).__init__()
        self.dim = dim
        self.register_buffer("weight", torch.randn(dim))
        self.register_buffer("bias", torch.randn(dim))

    def forward(self, x):
        return torch.cos(2 * torch.pi * (x[..., None] * self.weight + self.bias))


class TransitionBlock(nn.Module):
    def __init__(self, dim: int, input_dim: int = None, dropout=0.10):
        super(TransitionBlock, self).__init__()
        if input_dim is None:
            input_dim = dim
        self.input_dim = input_dim
        self.dim = dim
        self.ln = nn.LayerNorm(input_dim)
        self.swish_linear = nn.Linear(input_dim, dim, bias=False)
        self.swish_scale_linear = nn.Linear(input_dim, dim, bias=False)
        self.sigmoid_linear = nn.Linear(input_dim, dim, bias=True)
        self.sigmoid_scale_linear = nn.Linear(dim, dim, bias=False)
        nn.init.constant_(self.sigmoid_linear.bias, -2.0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_in = x
        x = self.ln(x)
        b = F.silu(self.swish_linear(x)) * self.swish_scale_linear(x)
        b = self.dropout(b)
        x = torch.sigmoid(self.sigmoid_linear(x)) * self.sigmoid_scale_linear(b)
        if self.input_dim == self.dim:
            return x_in + self.dropout(x)
        else:
            return x


def build_transition_stack(n_layers: int, dim_in: int, dim_hid: int, dim_out: int, dropout=0.10):
    layers: List[nn.Module] = [TransitionBlock(dim=dim_hid, input_dim=dim_in, dropout=dropout)]
    for _ in range(n_layers - 2):
        layers.append(TransitionBlock(dim=dim_hid, input_dim=dim_hid, dropout=dropout))
    layers.append(nn.Linear(dim_hid, dim_out))
    return nn.Sequential(*layers)


class InterfaceAttentionBlock(nn.Module):
    def __init__(self, fourier_dim, atom_feature_dim, hidden_dim, num_heads=4, dropout=0.10):
        super(InterfaceAttentionBlock, self).__init__()
        self.fourier_embed = FourierEmbedding(fourier_dim)
        self.num_heads = num_heads
        self.attention_mlp = build_transition_stack(3, fourier_dim + atom_feature_dim, hidden_dim, self.num_heads, dropout)

    def forward(self, v_equivariant, v_atom_features, edge_group_idx):
        if v_equivariant.shape[0] == 0:
            num_edges = edge_group_idx.max().item() + 1 if edge_group_idx.nelement() > 0 else 0
            return torch.zeros(num_edges, 3, device=v_equivariant.device)

        num_edges = int(edge_group_idx.max().item()) + 1
        v_norms = torch.linalg.norm(v_equivariant, dim=-1)
        f_dist = self.fourier_embed(v_norms)
        f_invariant = torch.cat([f_dist, v_atom_features], dim=-1)
        scores = self.attention_mlp(f_invariant)
        alphas = scatter_softmax(scores, edge_group_idx, dim=0)
        v_weighted = v_equivariant[:, None, :] * alphas[..., None]
        v_pooled = scatter_sum(v_weighted, edge_group_idx, dim=0, dim_size=num_edges)
        return v_pooled.reshape(v_pooled.shape[0], self.num_heads * 3)


class GeoFeat(nn.Module):
    def __init__(self, geo_layer, num_hidden, virtual_atom_num, dropout=0.10):
        super(GeoFeat, self).__init__()
        self.__dict__.update(locals())
        self.virtual_atom = nn.Linear(num_hidden, virtual_atom_num * 3)
        self.virtual_direct = nn.Linear(num_hidden, virtual_atom_num * 3)
        self.we_condition = build_transition_stack(geo_layer, 4 * virtual_atom_num * 3 + 9 + 16 + 32, num_hidden, num_hidden, dropout)
        self.MergeEG = nn.Linear(num_hidden + num_hidden, num_hidden)

    def forward(self, h_V, h_E, T_ts, edge_idx):
        if edge_idx.shape[1] == 0:
            return h_E

        src_idx = edge_idx[0]
        dst_idx = edge_idx[1]
        num_edge = src_idx.shape[0]
        num_node = h_V.shape[0]

        T_ts_rotations = T_ts.get_rots().get_rot_mats()  # [num_edges, 3, 3]
        T_ts_translations = T_ts.get_trans()             # [num_edges, 3]

        V_local = self.virtual_atom(h_V).view(num_node, -1, 3)
        V_edge = self.virtual_direct(h_E).view(num_edge, -1, 3)
        Ks = torch.cat([V_edge, V_local[src_idx]], dim=1)

        # Two-input einsum -> batched matmul. Same math, lower dispatcher overhead.
        rot_t = T_ts_rotations.transpose(-1, -2)
        Qt = torch.matmul(Ks, rot_t)
        src_virtual = V_local[src_idx]
        RKs = torch.matmul(src_virtual, rot_t)
        QRK = (V_local[dst_idx] * RKs).sum(dim=-1)

        edge_distance = torch.linalg.norm(T_ts_translations, dim=-1)
        edge_rbf_feats = rbf(edge_distance, v_min=0., v_max=20., n_bins=16)

        H = torch.cat([
            Ks.reshape(num_edge, -1), Qt.reshape(num_edge, -1), T_ts_rotations.reshape(num_edge, 9), edge_rbf_feats, QRK
        ], dim=1)

        G_e = self.we_condition(H)
        h_E = self.MergeEG(torch.cat([h_E, G_e], dim=-1))

        return h_E


class PiFoldAttn(nn.Module):
    def __init__(self, num_hidden, num_V, num_E, num_heads=4):
        super(PiFoldAttn, self).__init__()
        self.__dict__.update(locals())
        self.num_heads = num_heads
        self.head_dim = int(self.num_hidden / self.num_heads)
        self.attn_scale = self.head_dim ** -0.5

        self.W_V = nn.Sequential(
            nn.Linear(num_E, num_hidden),
            nn.GELU()
        )

        self.Bias = nn.Sequential(
            nn.Linear(2 * num_V + num_E, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, self.num_heads)
        )

        self.W_O = nn.Linear(num_hidden, num_V, bias=False)
        self.gate = nn.Linear(num_hidden, num_V)

    def forward(self, h_V, h_E, edge_idx):
        src_idx = edge_idx[0]
        dst_idx = edge_idx[1]
        h_V_skip = h_V

        E = h_E.shape[0]
        n_heads = self.num_heads
        d = self.head_dim
        num_nodes = h_V.shape[0]

        w = self.Bias(torch.cat([h_V[src_idx], h_E, h_V[dst_idx]], dim=-1)).view(E, n_heads, 1)
        attend_logits = w * self.attn_scale

        V = self.W_V(h_E).view(-1, n_heads, d)
        attend = scatter_softmax(attend_logits, index=src_idx, dim=0)
        h_V = scatter_sum(attend * V, src_idx, dim=0, dim_size=num_nodes).view([num_nodes, -1])

        h_V_gate = torch.sigmoid(self.gate(h_V))
        dh = self.W_O(h_V) * h_V_gate

        h_V = h_V_skip + dh
        return h_V


class UpdateNode(nn.Module):
    def __init__(self, num_hidden):
        super(UpdateNode, self).__init__()
        self.dense = nn.Sequential(
            nn.LayerNorm(num_hidden),
            nn.Linear(num_hidden, num_hidden * 4),
            nn.ReLU(),
            nn.Linear(num_hidden * 4, num_hidden),
            nn.LayerNorm(num_hidden)
        )
        self.V_MLP_g = nn.Sequential(
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden)
        )
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)

    def forward(
            self,
            h_V,
            batch_id,
            virtual_node_mask,
            real_node_idx: Optional[torch.Tensor] = None,
            batch_id_real: Optional[torch.Tensor] = None,
    ):
        h_V_skip_1 = h_V
        dh = self.dense(h_V)
        h_V = h_V_skip_1 + dh  # FFN update

        h_V_skip_2 = h_V
        h_V_norm = self.norm1(h_V)

        if real_node_idx is None:
            h_V_real = h_V_norm[~virtual_node_mask]
            batch_id_real = batch_id[~virtual_node_mask]
        else:
            h_V_real = h_V_norm.index_select(0, real_node_idx)
            if batch_id_real is None:
                batch_id_real = batch_id.index_select(0, real_node_idx)

        num_graphs = int(batch_id.max().item()) + 1 if batch_id.numel() > 0 else 0
        global_context = scatter_mean(h_V_real, batch_id_real, dim=0, dim_size=num_graphs)

        h_V_global = global_context[batch_id]
        gate_signal = F.silu(self.V_MLP_g(h_V_global))
        h_V = h_V_skip_2 + h_V_norm * gate_signal

        h_V = self.norm2(h_V)

        return h_V


class UpdateEdge(nn.Module):
    def __init__(self, edge_layer, num_hidden, dropout=0.10):
        super(UpdateEdge, self).__init__()
        self.mlp = build_transition_stack(
            n_layers=edge_layer,
            dim_in=num_hidden * 3,
            dim_hid=num_hidden,
            dim_out=num_hidden,
            dropout=dropout
        )

    def forward(self, h_V, h_E, edge_idx):
        src_idx, dst_idx = edge_idx[0], edge_idx[1]
        h_EV = torch.cat([h_V[src_idx], h_E, h_V[dst_idx]], dim=-1)
        h_E = h_E + self.mlp(h_EV)
        return h_E


class GeneralGNN(nn.Module):
    def __init__(self, geo_layer, edge_layer, num_hidden, num_heads, dropout=0.10, mask_rate=0.15, is_fine_tuning=False, virtual_atom_num=32):
        super(GeneralGNN, self).__init__()
        self.__dict__.update(locals())
        self.geofeat = GeoFeat(geo_layer, num_hidden, virtual_atom_num, dropout)
        self.attention = PiFoldAttn(num_hidden, num_hidden, num_hidden, num_heads)
        self.update_node = UpdateNode(num_hidden)
        self.update_edge = UpdateEdge(edge_layer, num_hidden, dropout)
        self.mask_token = nn.Embedding(2, num_hidden)
        self.is_fine_tuning = is_fine_tuning
        self.mask_rate = mask_rate

    def get_random_idx(self, h_V, mask_rate):
        num_N = int(h_V.shape[0] * mask_rate)
        indices = torch.randperm(h_V.shape[0], device=h_V.device)
        selected_indices = indices[:num_N]
        return selected_indices

    def forward(
            self,
            h_V,
            h_E,
            T_ts,
            edge_idx,
            batch_id,
            virtual_node_mask,
            real_node_idx: Optional[torch.Tensor] = None,
            batch_id_real: Optional[torch.Tensor] = None,
    ):
        if self.training and self.mask_rate != 0.0 and (not self.is_fine_tuning):
            selected_indices_v = self.get_random_idx(h_V, self.mask_rate)
            mask_v = self.mask_token.weight[0].to(h_V.dtype)
            h_V[selected_indices_v] = mask_v

            selected_indices_e = self.get_random_idx(h_E, self.mask_rate)
            mask_e = self.mask_token.weight[1].to(h_E.dtype)
            h_E[selected_indices_e] = mask_e

        h_E = self.geofeat(h_V, h_E, T_ts, edge_idx)
        h_V = self.attention(h_V, h_E, edge_idx)
        h_V = self.update_node(h_V, batch_id, virtual_node_mask, real_node_idx=real_node_idx, batch_id_real=batch_id_real)
        h_E = self.update_edge(h_V, h_E, edge_idx)
        return h_V, h_E


class StructureEncoder(nn.Module):
    def __init__(self, geo_layer, edge_layer, encoder_layer, hidden_dim, dropout=0.10, mask_rate=0.15, num_heads=4, is_fine_tuning=False):
        super(StructureEncoder, self).__init__()
        self.__dict__.update(locals())
        self.encoder_layers = nn.ModuleList([
            GeneralGNN(geo_layer, edge_layer, hidden_dim, num_heads, dropout, mask_rate, is_fine_tuning) for _ in range(encoder_layer)
        ])
        self.s = nn.Linear(hidden_dim, 1)

    def forward(self, h_V, h_E, T_ts, edge_idx, batch_id, virtual_node_mask):
        real_node_idx = (~virtual_node_mask).nonzero(as_tuple=True)[0]
        batch_id_real = batch_id.index_select(0, real_node_idx)

        acc = None
        for layer in self.encoder_layers:
            h_V, h_E = layer(
                h_V, h_E, T_ts, edge_idx, batch_id, virtual_node_mask,
                real_node_idx=real_node_idx, batch_id_real=batch_id_real,
            )
            h_V_real = h_V.index_select(0, real_node_idx)
            gate = torch.sigmoid(self.s(h_V_real))
            cur = h_V_real * gate
            acc = cur if acc is None else (acc + cur)
        return acc


class MLPDecoder(nn.Module):
    def __init__(self, hidden_dim, vocab=33):
        super().__init__()
        self.readout = build_transition_stack(2, hidden_dim, hidden_dim, vocab, dropout=0.0)

    def forward(self, h_V, return_log_probs=False):
        logits = self.readout(h_V)
        if return_log_probs:
            log_probs = F.log_softmax(logits, dim=-1)
            return log_probs, logits
        return logits
