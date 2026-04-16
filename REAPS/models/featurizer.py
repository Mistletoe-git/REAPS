from typing import List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import knn_graph

from REAPS.data.affine_utils import Rotation, Rigid
from REAPS.data.constants import *
from REAPS.models.module import FourierEmbedding, InterfaceAttentionBlock


def get_virtual_cb(n_coords, ca_coords, c_coords):
    b = ca_coords - n_coords
    c = c_coords - ca_coords
    a = torch.cross(b, c, dim=-1)
    return -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + ca_coords

def calculate_dihedral_and_sin_cos(p0, p1, p2, p3):
    b0 = -1.0 * (p1 - p0); b1 = p2 - p1; b2 = p3 - p2
    b1_norm = b1 / torch.linalg.norm(b1, dim=-1, keepdim=True).clamp(min=1e-7)
    v = b0 - torch.sum(b0 * b1_norm, dim=-1, keepdim=True) * b1_norm
    w = b2 - torch.sum(b2 * b1_norm, dim=-1, keepdim=True) * b1_norm
    x = torch.sum(v * w, dim=-1)
    y = torch.sum(torch.cross(b1_norm, v, dim=-1) * w, dim=-1)
    angle = torch.atan2(y, x)
    return torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1)

def positional_embeddings(E_idx, num_embeddings):
    d = E_idx[0] - E_idx[1]
    frequency = torch.exp(
        torch.arange(0, num_embeddings, 2, dtype=torch.float32, device=E_idx.device) * (-(np.log(10000.0) / num_embeddings))
    )
    angles = d[:, None] * frequency[None, :]
    return torch.cat((torch.cos(angles), torch.sin(angles)), -1)


class GraphFeaturizer(nn.Module):
    def __init__(self, ablation_mode: bool, backbone_noise_scale: float, k_neighbors: int, virtual_frame_num: int,
                 fourier_dim: int, dropout: float, positional_buckets: int, E_idx_embed_dim: int, hidden_dim: int, num_heads: int):

        super(GraphFeaturizer, self).__init__()
        self.ablation_mode = ablation_mode # True for ablation experiment
        self.backbone_noise_scale = backbone_noise_scale
        self.k_neighbors = k_neighbors
        self.virtual_frame_num = virtual_frame_num
        self.positional_buckets = positional_buckets
        self.E_idx_embed_dim = E_idx_embed_dim
        self.num_heads = num_heads
        self.atom_to_idx = ATOM_ORDER
        self.dist_embed = FourierEmbedding(dim=fourier_dim)
        self.node_type_embedding = nn.Embedding(3, 16)  # 0 -> peptide, 1 -> receptor, 2 -> virtual
        self.structure_feature_dim = ((3 + fourier_dim) * 4) + 4 * 2 + 4 + 16
        self.virtual_node_embedding = nn.Embedding(self.virtual_frame_num, self.structure_feature_dim - 16)
        self.edge_feature_dim = ((3 + fourier_dim) * 8) + 9 + 3 + (positional_buckets + 1) + self.E_idx_embed_dim + 4 + 3 * self.num_heads

        self.N_idx = self.atom_to_idx["N"]
        self.CA_idx = self.atom_to_idx["CA"]
        self.C_idx = self.atom_to_idx["C"]
        self.O_idx = self.atom_to_idx["O"]
        self.CB_idx = self.atom_to_idx["CB"]

        self.register_buffer(
            "bb_indices",
            torch.tensor([self.N_idx, self.CA_idx, self.C_idx, self.O_idx], dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "virtual_node_tokens",
            torch.full((self.virtual_frame_num,), 21, dtype=torch.long),
            persistent=False,
        )

        periodic_table = torch.tensor(PERIODIC_TABLE_FEATURES, dtype=torch.float32).T
        self.register_buffer("periodic_table", periodic_table, persistent=False)

        max_side = max((len(v) for v in ATOM_NAME_TO_ELEMENT_SYMBOL.values()), default=0)
        self.max_side = max_side

        num_restype = len(STANDARD_AMINO_ACIDS)
        carbon_z = ELEMENT_DICT.get("C", 6)

        side_idx_lut = torch.zeros((num_restype, max_side), dtype=torch.long)
        atomic_num_lut = torch.full((num_restype, max_side), carbon_z, dtype=torch.long)

        for res_type_id, res1 in enumerate(STANDARD_AMINO_ACIDS):
            res3 = RESTYPE_1_TO_3.get(res1)
            if res3 not in ATOM_NAME_TO_ELEMENT_SYMBOL:
                continue

            atom_names = list(ATOM_NAME_TO_ELEMENT_SYMBOL[res3].keys())
            n_atom = min(len(atom_names), max_side)

            idx_vec = [self.atom_to_idx.get(name, 0) for name in atom_names[:n_atom]]
            z_vec = [
                ELEMENT_DICT.get(ATOM_NAME_TO_ELEMENT_SYMBOL[res3][name], carbon_z)
                for name in atom_names[:n_atom]
            ]

            if n_atom > 0:
                side_idx_lut[res_type_id, :n_atom] = torch.tensor(idx_vec, dtype=torch.long)
                atomic_num_lut[res_type_id, :n_atom] = torch.tensor(z_vec, dtype=torch.long)

        self.register_buffer("side_idx_lut", side_idx_lut, persistent=False)
        self.register_buffer("atomic_num_lut", atomic_num_lut, persistent=False)

        self.chi_atom_index_map = {}
        for res_type_id, res1 in enumerate(STANDARD_AMINO_ACIDS):
            res3 = RESTYPE_1_TO_3.get(res1)
            if res3 not in CHI_ATOM_DEFINITIONS:
                continue
            self.chi_atom_index_map[res_type_id] = [
                [self.atom_to_idx[name] for name in chi_def]
                for chi_def in CHI_ATOM_DEFINITIONS[res3]
            ]

        self.attention_pooling = InterfaceAttentionBlock(
            fourier_dim=fourier_dim,
            # Periodic table number + period number + group number + Angle information of polypeptide residues Ca/C/N/O and side chain atoms
            atom_feature_dim=119 + 19 + 8 + 4,
            hidden_dim=hidden_dim,
            num_heads=self.num_heads,
            dropout= dropout
        )

    def forward(self, data_list: List[dict], inference_peptide_chain_ids: List[str] = None) -> Dict[str, torch.Tensor]:
        if inference_peptide_chain_ids is not None: # Manually specify the polypeptide chain id during inference
            feature_dict_list = [self._build_graph(data, chain_id) for data, chain_id in zip(data_list, inference_peptide_chain_ids)]
        else:
            feature_dict_list = [self._build_graph(data, None) for data in data_list]
        return self._custom_collate_fn(feature_dict_list)

    def _custom_collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        batch = [b for b in batch if b is not None and b]
        if not batch:
            return {}
        device = batch[0]['_V'].device
        num_nodes_per_graph = torch.tensor([data['num_total_nodes'] for data in batch], device=device)
        node_shifts = torch.cat([torch.tensor([0], device=device), num_nodes_per_graph.cumsum(dim=0)[:-1]])
        batched_dict = {}
        first_item_keys = batch[0].keys()

        keys_to_skip = ['num_total_nodes', 'peptide_chain_id', 'complex_id', 'cyclic_type',
                        'pep_len', 'rec_len', 'seq', 'chain_encoding', 'seq_indices']

        for key in first_item_keys:
            if key in keys_to_skip:
                continue
            tensors = [data[key] for data in batch]
            if key == 'E_idx':
                shifted_edges = [edges + node_shifts[i] for i, edges in enumerate(tensors)]
                batched_dict[key] = torch.cat(shifted_edges, dim=1)
            else:
                batched_dict[key] = torch.cat(tensors, dim=0)

        batch_idx = torch.arange(len(batch), device=node_shifts.device).repeat_interleave(num_nodes_per_graph)
        batched_dict['batch_id'] = batch_idx
        batched_dict['peptide_chain_ids'] = [data.get('peptide_chain_id', 'L') for data in batch]
        return batched_dict

    def _decouple(self, U):
        norm = U.norm(dim=-1, keepdim=True)
        direct = U / (norm.clamp(min=1e-6))
        fourier_result = self.dist_embed(norm.squeeze(-1))
        return torch.cat([direct, fourier_result], dim=-1)

    def _get_positional_buckets(self, offset, same_chain):
        num_buckets = self.positional_buckets
        max_offset = num_buckets // 2
        bucketed = torch.clamp(offset, -max_offset, max_offset) + max_offset
        positional_one_hot = torch.zeros(*offset.shape, num_buckets + 1, device=offset.device)
        same_chain_mask = same_chain.bool().squeeze(-1)
        if same_chain_mask.any(): # Mask the index information between multiple chains. The R_idx_offset of different chains is meaningless
            positional_one_hot.scatter_(-1, bucketed[same_chain_mask].unsqueeze(-1).long(), 1.0)
        positional_one_hot[~same_chain_mask, -1] = 1.0
        return positional_one_hot

    def _compute_chi_angles(self, RP_S, xyz_37, mask_37, peptide_mask):
        num_res, device = RP_S.shape[0], xyz_37.device
        mask_37 = mask_37.bool()
        chi_sin_cos = torch.zeros(num_res, 4, 2, device=device)
        chi_valid = torch.zeros(num_res, 4, device=device)
        n_coords, ca_coords, c_coords = xyz_37[:, 0], xyz_37[:, 1], xyz_37[:, 2]
        cb_idx = self.atom_to_idx['CB']
        gly_idx = STANDARD_AMINO_ACIDS.index('G')
        is_gly_mask = (RP_S == gly_idx)
        cb_coords = xyz_37[:, cb_idx].clone()
        bb_mask = mask_37[:, [0, 1, 2]].all(dim=-1)
        gly_with_bb_mask = (is_gly_mask & bb_mask).bool()
        if gly_with_bb_mask.any():
            cb_coords[gly_with_bb_mask] = get_virtual_cb(n_coords[gly_with_bb_mask], ca_coords[gly_with_bb_mask], c_coords[gly_with_bb_mask])
        cb_available_mask = mask_37[:, cb_idx] | gly_with_bb_mask
        virtual_chi1_mask = bb_mask & cb_available_mask
        if virtual_chi1_mask.any():
            p0, p1, p2, p3 = n_coords, ca_coords, cb_coords, c_coords
            sin_cos = calculate_dihedral_and_sin_cos(p0[virtual_chi1_mask], p1[virtual_chi1_mask], p2[virtual_chi1_mask], p3[virtual_chi1_mask])
            chi_sin_cos[virtual_chi1_mask, 0] = sin_cos
            chi_valid[virtual_chi1_mask, 0] = 0.5
        receptor_mask = ~peptide_mask[:num_res]
        for res_type_idx in torch.unique(RP_S[receptor_mask]):
            res_type_id = int(res_type_idx.item())
            chi_atom_defs = self.chi_atom_index_map.get(res_type_id, None)
            if chi_atom_defs is None:
                continue
            is_this_res_type = (RP_S == res_type_idx) & receptor_mask
            res_indices = is_this_res_type.nonzero(as_tuple=True)[0]
            if res_indices.numel() == 0:
                continue
            for i, atom_indices in enumerate(chi_atom_defs):
                p0_r, p1_r, p2_r, p3_r = (xyz_37[res_indices, idx] for idx in atom_indices)
                m = mask_37[res_indices][:, atom_indices].all(dim=-1)
                if m.any():
                    sin_cos_real = calculate_dihedral_and_sin_cos(p0_r[m], p1_r[m], p2_r[m], p3_r[m])
                    chi_sin_cos[res_indices[m], i] = sin_cos_real
                    chi_valid[res_indices[m], i] = 1.0
        return torch.cat([chi_sin_cos.flatten(start_dim=1), chi_valid], dim=1)

    def _compute_equivariant_interface_features(self, T, xyz_37, mask_37, interface_E_idx, RP_S):
        device = xyz_37.device
        E = interface_E_idx.shape[1]
        atom_type_feature_dim = NUM_ATOMIC_NUMBERS + NUM_ATOM_GROUPS + NUM_ATOM_PERIODS

        if E == 0:
            return {
                'interface_vectors': torch.empty(0, 3, device=device),
                'interface_atom_features': torch.empty(0, atom_type_feature_dim + 4, device=device),
                'interface_atom_group_idx': torch.empty(0, dtype=torch.long, device=device),
            }

        r_idx = interface_E_idx[1].clamp(min=0, max=T.shape[0] - 1)   # receptor residue
        p_idx = interface_E_idx[0].clamp(min=0, max=T.shape[0] - 1)   # peptide residue

        res_types_r = RP_S[r_idx]
        side_idx = self.side_idx_lut[res_types_r]                     # [E, max_side]
        atomic_num_full = self.atomic_num_lut[res_types_r]            # [E, max_side]

        side_xyz = xyz_37[r_idx].gather(
            1,
            side_idx.unsqueeze(-1).expand(-1, -1, 3)
        )                                                            # [E, max_side, 3]
        side_mask = mask_37[r_idx].gather(1, side_idx).bool()        # [E, max_side]

        edge_has_atoms = side_mask.any(dim=1)
        if not edge_has_atoms.all():
            missing_mask = ~edge_has_atoms
            miss_r = r_idx[missing_mask]

            n_coord = xyz_37[miss_r, self.N_idx]
            ca_coord = xyz_37[miss_r, self.CA_idx]
            c_coord = xyz_37[miss_r, self.C_idx]
            virtual_cb = get_virtual_cb(n_coord, ca_coord, c_coord)

            side_xyz[missing_mask, 0] = virtual_cb
            side_mask[missing_mask, 0] = True

        flat_mask = side_mask.reshape(-1)                            # [E * max_side]
        flat_xyz = side_xyz.reshape(-1, 3)[flat_mask]                # [M, 3]
        flat_edge = (
            torch.arange(E, device=device)
            .unsqueeze(1)
            .expand(E, self.max_side)
            .reshape(-1)[flat_mask]
        )                                                            # [M]
        atomic_num = atomic_num_full.reshape(-1)[flat_mask]          # [M]

        rot = T[p_idx].get_rots().get_rot_mats()                     # [E, 3, 3]
        trans = T[p_idx].get_trans()                                 # [E, 3]
        centered = flat_xyz - trans[flat_edge]                       # [M, 3]
        all_vectors = torch.bmm(
            rot[flat_edge],
            centered.unsqueeze(-1)
        ).squeeze(-1)                                                # [M, 3]

        peptide_N_coords = xyz_37[p_idx, self.N_idx]                 # [E, 3]
        peptide_CA_coords = xyz_37[p_idx, self.CA_idx]               # [E, 3]
        peptide_C_coords = xyz_37[p_idx, self.C_idx]                 # [E, 3]

        A_expanded = peptide_N_coords[flat_edge]                     # [M, 3]
        B_expanded = peptide_CA_coords[flat_edge]                    # [M, 3]
        C_expanded = peptide_C_coords[flat_edge]                     # [M, 3]
        Y_targets = flat_xyz                                         # [M, 3]

        v1 = A_expanded - B_expanded
        v2 = C_expanded - B_expanded
        e1 = F.normalize(v1, dim=-1)

        e1_v2_dot = (e1 * v2).sum(dim=-1, keepdim=True)
        u2 = v2 - e1 * e1_v2_dot
        e2 = F.normalize(u2, dim=-1)
        e3 = torch.cross(e1, e2, dim=-1)

        R_expanded = torch.stack([e1, e2, e3], dim=1)               # [M, 3, 3]
        local_Y = torch.bmm(
            R_expanded,
            (Y_targets - B_expanded).unsqueeze(-1)
        ).squeeze(-1)

        rxy = torch.sqrt(local_Y[..., 0] ** 2 + local_Y[..., 1] ** 2 + 1e-8)
        rxyz = torch.norm(local_Y, dim=-1) + 1e-8
        f1 = local_Y[..., 0] / rxy
        f2 = local_Y[..., 1] / rxy
        f3 = rxy / rxyz
        f4 = local_Y[..., 2] / rxyz
        angle_features = torch.stack([f1, f2, f3, f4], dim=-1)      # [M, 4]

        raw_feat = self.periodic_table[atomic_num, :3]              # [M, 3]
        an_oh = F.one_hot(raw_feat[:, 0].long(), NUM_ATOMIC_NUMBERS).float()
        grp_oh = F.one_hot(raw_feat[:, 1].long(), NUM_ATOM_GROUPS).float()
        per_oh = F.one_hot(raw_feat[:, 2].long(), NUM_ATOM_PERIODS).float()

        all_atom_feats = torch.cat([angle_features, an_oh, grp_oh, per_oh], dim=-1)
        all_group_indices = flat_edge

        return {
            'interface_vectors': all_vectors,
            'interface_atom_features': all_atom_feats,
            'interface_atom_group_idx': all_group_indices,
        }

    def _build_graph(self, data: dict, inference_peptide_chain_id: str = None):

        device = next(self.parameters()).device

        if 'cyclic_type' in data :
            peptide_chain_id_for_output = "L"
            seq_str = data['seq']
            seq_encoded = [AA_TO_IDX.get(aa, AA_TO_IDX['X']) for aa in seq_str]
            RP_S = torch.tensor(seq_encoded, device=device, dtype=torch.long)
            xyz_37 = torch.as_tensor(data['xyz_37'], device=device, dtype=torch.float32)
            xyz_37_m = torch.as_tensor(data['xyz_37_mask'], device=device, dtype=torch.bool)
            chain_idx = torch.as_tensor(data['chain_encoding'], device=device, dtype=torch.long)
            R_idx = torch.as_tensor(data['seq_indices'], device=device, dtype=torch.long)
            is_peptide_residue = (chain_idx == 1)
            num_real_nodes = len(seq_str)
        else:
            chain_features = data.get('chain_features', [])

            if len(chain_features) < 2:
                return None

            if inference_peptide_chain_id is not None:
                chain_id_to_index = {chain['chain_id']: i for i, chain in enumerate(chain_features)}
                if inference_peptide_chain_id not in chain_id_to_index:
                    return None
                peptide_chain_index = chain_id_to_index[inference_peptide_chain_id]
                peptide_chain_id_for_output = inference_peptide_chain_id
            else:
                chain_lengths = [len(c['seq']) for c in chain_features]
                shortest_chain_index = np.argmin(chain_lengths)
                peptide_chain_index = shortest_chain_index
                peptide_chain_id_for_output = chain_features[shortest_chain_index]['chain_id']  # Letter chain ID

            all_seqs, all_xyz_37, all_xyz_37_m, all_chain_idx, all_R_idx = [], [], [], [], []

            is_peptide_residue_list = []

            for i, chain in enumerate(chain_features):
                num_residues = len(chain['seq'])
                seq_str = chain['seq']
                seq_indices = [AA_TO_IDX.get(aa, AA_TO_IDX['X']) for aa in seq_str]
                all_seqs.append(torch.tensor(seq_indices, device=device, dtype=torch.long))
                all_xyz_37.append(torch.as_tensor(chain['xyz_37'], device=device, dtype=torch.float32))
                all_xyz_37_m.append(torch.as_tensor(chain['xyz_37_mask'], device=device, dtype=torch.bool))
                all_R_idx.append(torch.as_tensor(chain['R_idx'], device=device, dtype=torch.long))
                all_chain_idx.append(torch.full((num_residues,), fill_value=i, device=device, dtype=torch.long))
                is_peptide = (i == peptide_chain_index)
                is_peptide_residue_list.append(torch.full((num_residues,), fill_value=is_peptide, device=device, dtype=torch.bool))

            RP_S = torch.cat(all_seqs, dim=0)
            xyz_37 = torch.cat(all_xyz_37, dim=0)
            xyz_37_m = torch.cat(all_xyz_37_m, dim=0)
            chain_idx = torch.cat(all_chain_idx, dim=0)
            R_idx = torch.cat(all_R_idx, dim=0)
            is_peptide_residue = torch.cat(is_peptide_residue_list, dim=0)
            num_real_nodes = len(RP_S)

        chain_idx_with_virtual = torch.cat([
            chain_idx,
            torch.full((self.virtual_frame_num,), -1, dtype=torch.long, device=device)  # The chain_idx of the virtual node is -1
        ], dim=0)
        is_receptor_residue = ~is_peptide_residue

        if self.training and self.backbone_noise_scale != 0.0:
            noise = torch.randn_like(xyz_37) * self.backbone_noise_scale
            xyz_37 = xyz_37 + noise

        N, CA, C = xyz_37[:, 0], xyz_37[:, 1], xyz_37[:, 2]
        T = Rigid.make_transform_from_reference(N, CA, C)
        E_idx_local = knn_graph(CA, k=self.k_neighbors, loop=True, flow='target_to_source')
        num_total_nodes = num_real_nodes + self.virtual_frame_num
        real_nodes_idx = torch.arange(num_real_nodes, device=device)
        virtual_nodes_idx =torch.arange(self.virtual_frame_num, device=device) + num_real_nodes
        node_type_ids = torch.full((num_total_nodes,), 2, dtype=torch.long, device=device)
        node_type_ids_real = torch.zeros(num_real_nodes, dtype=torch.long, device=device)
        node_type_ids_real[is_receptor_residue] = 1
        node_type_ids[:num_real_nodes] = node_type_ids_real
        src_global = real_nodes_idx.repeat_interleave(self.virtual_frame_num)
        dst_global = virtual_nodes_idx.repeat(num_real_nodes)
        E_idx_g = torch.stack([torch.cat([src_global, dst_global]), torch.cat([dst_global, src_global])], dim=0)
        E_idx = torch.cat([E_idx_local, E_idx_g], dim=1)
        src_all, dst_all = E_idx[0], E_idx[1]

        X_c = T._trans
        X_m = X_c.mean(dim=0, keepdim=True)
        X_c = X_c - X_m
        svd_input = X_c.T @ X_c
        U, S, V = torch.svd(svd_input.float())
        d = (torch.det(U) * torch.det(V)) < 0.0
        D = torch.eye(3, device=device)
        D[2, 2] = -2 * d + 1
        R = torch.matmul(U, V.T) @ D
        rot_g = R.unsqueeze(0).repeat(self.virtual_frame_num, 1, 1)
        trans_g = X_m.repeat(self.virtual_frame_num, 1)
        T_g = Rigid(Rotation(rot_g), trans_g)
        T_all = Rigid.cat([T, T_g], dim=0)

        if self.ablation_mode:  # ablation receptor_aware
            V_sidechain = torch.zeros(len(RP_S), 12, device=device, dtype=torch.float)
        else:
            V_sidechain = self._compute_chi_angles(RP_S, xyz_37, xyz_37_m, is_peptide_residue)

        X = xyz_37[:, self.bb_indices, :]
        diff_X = F.pad(X.reshape(-1, 3).diff(dim=0), (0, 0, 1, 0)).reshape(num_real_nodes, -1, 3)
        diff_X_proj = T[:, None].invert().get_rots().apply(diff_X)
        V_backbone = self._decouple(diff_X_proj).reshape(num_real_nodes, -1)
        _V_real_base = torch.cat([V_backbone, V_sidechain], dim=-1).float()
        all_node_type_embeds = self.node_type_embedding(node_type_ids)
        _V_real = torch.cat([_V_real_base, all_node_type_embeds[:num_real_nodes]], dim=1)
        _V_g_base = self.virtual_node_embedding(torch.arange(self.virtual_frame_num, device=device))
        _V_g = torch.cat([_V_g_base, all_node_type_embeds[num_real_nodes:]], dim=1)
        _V = torch.cat([_V_real, _V_g], dim=0)

        T_ts_local = T[E_idx_local[1]].invert().compose(T[E_idx_local[0]])
        X_src_bb_local, X_dst_bb_local = xyz_37[E_idx_local[0]][:, self.bb_indices], xyz_37[E_idx_local[1]][:, self.bb_indices]
        diffE_local = T[E_idx_local[0], None].invert().apply(torch.cat([X_src_bb_local, X_dst_bb_local], dim=1))
        diffE_proj_local = self._decouple(diffE_local).reshape(E_idx_local.shape[1], -1)
        E_quant_local = T_ts_local.invert().get_rots().get_rot_mats().reshape(E_idx_local.shape[1], 9)
        E_trans_local = T_ts_local.get_trans()
        same_chain = (chain_idx[E_idx_local[0]] == chain_idx[E_idx_local[1]]).int().unsqueeze(-1)
        R_idx_offset = R_idx[E_idx_local[0]] - R_idx[E_idx_local[1]]
        E_positional = self._get_positional_buckets(R_idx_offset, same_chain)
        E_idx_embed = positional_embeddings(E_idx_local, self.E_idx_embed_dim)
        _E_local = torch.cat([diffE_proj_local, E_quant_local, E_trans_local, E_positional, E_idx_embed], dim=-1).float()
        _E_local[torch.isnan(_E_local)] = 0.0
        _E_g = _E_local.new_zeros(E_idx_g.shape[1], _E_local.shape[1])
        _E_base = torch.cat([_E_local, _E_g], dim=0)

        src_type, dst_type = node_type_ids[src_all], node_type_ids[dst_all]
        is_intra_peptide = (src_type == 0) & (dst_type == 0)
        is_intra_receptor = (src_type == 1) & (dst_type == 1)
        is_interface = ((src_type == 0) & (dst_type == 1)) | ((src_type == 1) & (dst_type == 0))
        is_global = (src_type == 2) | (dst_type == 2)
        edge_type_feature = _E_local.new_zeros(E_idx.shape[1], 4)
        edge_type_feature[is_intra_peptide, 0] = 1
        edge_type_feature[is_intra_receptor, 1] = 1
        edge_type_feature[is_interface, 2] = 1
        edge_type_feature[is_global, 3] = 1

        is_interface_mask = edge_type_feature[:, 2] == 1
        interface_E_idx_all = E_idx[:, is_interface_mask]
        pooled_vectors = _E_local.new_zeros(interface_E_idx_all.shape[1], 3 * self.num_heads)

        should_compute_attention = (interface_E_idx_all.shape[1] > 0) and (not self.ablation_mode)

        if should_compute_attention:
            peptide_mask_src = is_peptide_residue[interface_E_idx_all[0]]
            needs_swap = ~peptide_mask_src
            if needs_swap.any():
                interface_E_idx_all[:, needs_swap] = interface_E_idx_all[:, needs_swap].roll(1, 0)
            atom_data = self._compute_equivariant_interface_features(T, xyz_37, xyz_37_m, interface_E_idx_all, RP_S)

            if atom_data['interface_vectors'].shape[0] > 0:
                pooled_vectors = self.attention_pooling(
                    v_equivariant=atom_data['interface_vectors'],
                    v_atom_features=atom_data['interface_atom_features'],
                    edge_group_idx=atom_data['interface_atom_group_idx']
                )

        final_interface_features = _E_local.new_zeros(E_idx.shape[1], 3 * self.num_heads)
        final_interface_features[is_interface_mask] = pooled_vectors.to(final_interface_features.dtype)
        _E = torch.cat([_E_base, edge_type_feature, final_interface_features], dim=1)

        S_t = torch.cat([RP_S, self.virtual_node_tokens], dim=0)
        virtual_node_mask = torch.zeros(num_total_nodes, dtype=torch.bool, device=device)
        virtual_node_mask[num_real_nodes:] = True
        peptide_mask = torch.zeros(num_total_nodes, dtype=torch.bool, device=device)
        peptide_mask[:num_real_nodes] = is_peptide_residue

        return {
            'S_t': S_t,
            'peptide_chain_id': peptide_chain_id_for_output,
            'chain_idx_with_virtual': chain_idx_with_virtual,
            'E_idx': E_idx,
            '_E': _E,
            '_V': _V,
            'peptide_mask': peptide_mask,
            'virtual_node_mask': virtual_node_mask,
            'T_ts_all_rot': (T_all[dst_all].invert().compose(T_all[src_all])).get_rots().get_rot_mats(),
            'T_ts_all_trans': (T_all[dst_all].invert().compose(T_all[src_all])).get_trans(),
            'num_total_nodes': num_total_nodes
        }
