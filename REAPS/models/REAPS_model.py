from typing import List, Dict
import numpy as np
import torch
import torch.nn as nn
from lightning import LightningModule
from torch.nn import functional as F
from transformers import get_inverse_sqrt_schedule, get_linear_schedule_with_warmup

from REAPS.data.affine_utils import Rigid, Rotation
from REAPS.data.constants import AA_TO_IDX
from REAPS.models.featurizer import GraphFeaturizer
from REAPS.models.module import (
    InterfaceAttentionBlock, TransitionBlock, build_transition_stack, PiFoldAttn,
    StructureEncoder, MLPDecoder, GeoFeat, enable_opt_einsum_backend
)

class REAPS_Model(LightningModule):
    def __init__(self, ablation_mode: bool, is_fine_tuning: bool, backbone_noise_scale: float, k_neighbors: int,
                 virtual_frame_num: int, fourier_dim: int, positional_buckets: int, E_idx_embed_dim: int,
                 hidden_dim: int, num_heads: int, dropout: float, geo_layer: int, edge_layer: int,
                 encoder_layer: int, mask_rate: float, vocab_size: int, lr: float, weight_decay: float, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = lr
        self.weight_decay = weight_decay
        self.is_fine_tuning = is_fine_tuning

        # Safe global backend toggle. This does not touch state_dict keys.
        try:
            enable_opt_einsum_backend("auto")
        except Exception:
            pass

        self.featurizer = GraphFeaturizer(
            ablation_mode=ablation_mode, backbone_noise_scale=backbone_noise_scale, k_neighbors=k_neighbors,
            virtual_frame_num=virtual_frame_num, fourier_dim=fourier_dim, positional_buckets=positional_buckets,
            E_idx_embed_dim=E_idx_embed_dim, hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout
        )
        V_input_dim = self.featurizer.structure_feature_dim
        E_input_dim = self.featurizer.edge_feature_dim

        self.node_norm, self.edge_norm = nn.LayerNorm(V_input_dim), nn.LayerNorm(E_input_dim)
        self.node_embedding = build_transition_stack(2, V_input_dim, hidden_dim, hidden_dim, dropout)
        self.edge_embedding = build_transition_stack(2, E_input_dim, hidden_dim, hidden_dim, dropout)
        self.encoder = StructureEncoder(
            geo_layer=geo_layer, edge_layer=edge_layer, encoder_layer=encoder_layer, hidden_dim=hidden_dim,
            dropout=dropout, mask_rate=mask_rate, num_heads=num_heads, is_fine_tuning=self.is_fine_tuning
        )
        self.draft_decoder = MLPDecoder(hidden_dim, vocab=vocab_size)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.10)
        if not self.is_fine_tuning:
            self.apply(self._init_weights)

    @staticmethod
    def _unify_batch_format(raw_batch: List[Dict]) -> List[Dict]:
        standard_batch = []
        for data in raw_batch:
            standard_data = data.copy()

            if 'chain_features' not in standard_data and 'rec_len' in standard_data and 'pep_len' in standard_data:
                r_len = standard_data['rec_len']

                # Fetch masks if they exist, otherwise generate default valid masks
                total_len = len(standard_data['seq'])
                mask_37 = standard_data.get('xyz_37_mask', np.ones((total_len, 37), dtype=bool))
                seq_idx = standard_data.get('seq_indices', np.arange(total_len, dtype=np.int64))

                standard_data['chain_features'] = [
                    {
                        'chain_id': 'R',
                        'seq': standard_data['seq'][:r_len],
                        'xyz_37': standard_data['xyz_37'][:r_len],
                        'xyz_37_mask': mask_37[:r_len],
                        'R_idx': seq_idx[:r_len]
                    },
                    {
                        'chain_id': 'L',
                        'seq': standard_data['seq'][r_len:],
                        'xyz_37': standard_data['xyz_37'][r_len:],
                        'xyz_37_mask': mask_37[r_len:],
                        'R_idx': seq_idx[r_len:]
                    }
                ]
            if 'chain_features' in standard_data:
                chain_list = standard_data['chain_features']

                if 'seq' not in standard_data:
                    standard_data['seq'] = "".join([c['seq'] for c in chain_list])

                if 'xyz_37' not in standard_data:
                    standard_data['xyz_37'] = np.concatenate([c['xyz_37'] for c in chain_list], axis=0)

                if 'xyz_37_mask' not in standard_data:
                    if 'xyz_37_mask' in chain_list[0]:
                        standard_data['xyz_37_mask'] = np.concatenate([c['xyz_37_mask'] for c in chain_list], axis=0)
                    else:
                        masks = [np.isfinite(np.sum(c['xyz_37'], axis=-1)) for c in chain_list]
                        standard_data['xyz_37_mask'] = np.concatenate(masks, axis=0)

                if 'chain_encoding' not in standard_data:
                    # Assign 0 to the first chain (Receptor), and 1 to the second (Peptide)
                    standard_data['chain_encoding'] = np.concatenate(
                        [np.full(len(c['seq']), i, dtype=np.int64) for i, c in enumerate(chain_list)]
                    )

                if 'seq_indices' not in standard_data:
                    idx_list = []
                    for c in chain_list:
                        # R_idx is exactly what the featurizer uses for positional offset calculations
                        if 'R_idx' in c:
                            idx_list.append(c['R_idx'])
                        else:
                            idx_list.append(np.arange(len(c['seq']), dtype=np.int64))
                    standard_data['seq_indices'] = np.concatenate(idx_list)

            standard_batch.append(standard_data)

        return standard_batch

    def forward(self, raw_batch: List[Dict], inference_peptide_chain_ids: List[str] = None):
        raw_batch = self._unify_batch_format(raw_batch)

        batched_graph = self.featurizer(raw_batch, inference_peptide_chain_ids)
        if not batched_graph:
            return None

        _V, _E, edge_idx, batch_id, virtual_node_mask = (
            batched_graph['_V'], batched_graph['_E'], batched_graph['E_idx'],
            batched_graph['batch_id'], batched_graph['virtual_node_mask']
        )

        T_ts = Rigid(
            rots=Rotation(rot_mats=batched_graph['T_ts_all_rot']),
            trans=batched_graph['T_ts_all_trans']
        )

        h_V = self.node_embedding(self.node_norm(_V))
        h_E = self.edge_embedding(self.edge_norm(_E))
        encoded_V_real = self.encoder(h_V, h_E, T_ts, edge_idx, batch_id, virtual_node_mask)

        decoder_out = self.draft_decoder(encoded_V_real)
        if isinstance(decoder_out, tuple):
            _, logits_real = decoder_out
        else:
            logits_real = decoder_out

        real_node_idx = (~virtual_node_mask).nonzero(as_tuple=True)[0]
        peptide_mask_real = batched_graph['peptide_mask'].index_select(0, real_node_idx)
        logits = logits_real[peptide_mask_real]
        S_real = batched_graph['S_t'].index_select(0, real_node_idx)
        y_true = S_real[peptide_mask_real]
        assert logits.shape[0] == y_true.shape[0], f"logits {logits.shape} v.s. y_true {y_true.shape}"

        return {
            'batched_graph': batched_graph,
            "logits": logits,
            "y_true": y_true,
            "embeddings": encoded_V_real
        }

    def _calculate_metrics(self, results: Dict):
        if not results:
            print("Warning: Empty logits tensor encountered in validation. This batch may not contain any peptides.")
            return None

        logits = results['logits'].float()
        y_true = results['y_true']

        if logits.numel() == 0:
            return None

        loss = self.criterion(logits, y_true)
        nll_loss = F.cross_entropy(logits, y_true, label_smoothing=0.0)

        preds = torch.argmax(logits, dim=-1)
        recovery = (preds == y_true).float().mean()

        metrics_dict = {
            'loss': loss,
            'recovery': recovery,
            'perplexity': torch.exp(nll_loss.float()),
        }

        return metrics_dict

    def training_step(self, batch, batch_idx):
        results = self(batch)
        metrics = self._calculate_metrics(results)
        if metrics is None:
            return None
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('train/lr', current_lr, on_step=True, on_epoch=False, prog_bar=True)
        self.log('train/loss', metrics['loss'], on_step=True, on_epoch=True, prog_bar=True, batch_size=len(batch))
        self.log('train/recovery', metrics['recovery'], on_step=True, on_epoch=True, prog_bar=True, batch_size=len(batch))
        self.log('train/ppl', metrics['perplexity'], on_step=True, on_epoch=True, prog_bar=True, batch_size=len(batch))
        return metrics['loss']

    def validation_step(self, batch, batch_idx):
        results = self(batch)
        metrics = self._calculate_metrics(results)

        if metrics is None:
            return None

        self.log('val/loss', metrics['loss'], on_epoch=True, prog_bar=True, sync_dist=True, batch_size=len(batch))
        self.log('val/recovery', metrics['recovery'], on_epoch=True, prog_bar=True, sync_dist=True, batch_size=len(batch))
        self.log('val/ppl', metrics['perplexity'], on_epoch=True, prog_bar=True, sync_dist=True, batch_size=len(batch))
        return metrics['loss']

    def test_step(self, batch, batch_idx):
        results = self(batch)
        metrics = self._calculate_metrics(results)
        if metrics is None:
            return None
        self.log('test/loss', metrics['loss'], on_epoch=True, sync_dist=True, batch_size=len(batch))
        self.log('test/recovery', metrics['recovery'], on_epoch=True, sync_dist=True, batch_size=len(batch))
        self.log('test/ppl', metrics['perplexity'], on_epoch=True, sync_dist=True, batch_size=len(batch))
        return metrics['loss']

    def configure_optimizers(self):
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_steps * 0.025)
        if self.is_fine_tuning:
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=self.learning_rate,
                weight_decay=self.weight_decay * 10
            )
            scheduler = get_linear_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
        else:
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
            scheduler = get_inverse_sqrt_schedule(
                optimizer=optimizer,
                num_warmup_steps=warmup_steps
            )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}
        }

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad(set_to_none=True)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, TransitionBlock):
            if hasattr(module, 'sigmoid_linear'):
                nn.init.constant_(module.sigmoid_linear.bias, -2.0)
            if hasattr(module, 'sigmoid_scale_linear'):
                nn.init.normal_(module.sigmoid_scale_linear.weight, std=1e-5)
        elif isinstance(module, GeoFeat):
            if hasattr(module, 'virtual_atom'):
                nn.init.normal_(module.virtual_atom.weight, std=0.01)
                if module.virtual_atom.bias is not None:
                    nn.init.constant_(module.virtual_atom.bias, 0)
            if hasattr(module, 'virtual_direct'):
                nn.init.normal_(module.virtual_direct.weight, std=0.01)
                if module.virtual_direct.bias is not None:
                    nn.init.constant_(module.virtual_direct.bias, 0)
        elif isinstance(module, PiFoldAttn):
            if hasattr(module, 'Bias'):
                last_attn_linear = module.Bias[-1]
                if isinstance(last_attn_linear, nn.Linear):
                    nn.init.normal_(last_attn_linear.weight, std=1e-4)
                    if last_attn_linear.bias is not None:
                        nn.init.constant_(last_attn_linear.bias, 0)
            if hasattr(module, 'gate'):
                nn.init.constant_(module.gate.bias, 0)
        elif isinstance(module, InterfaceAttentionBlock):
            if hasattr(module, 'attention_mlp'):
                last_layer = module.attention_mlp[-1]
                if isinstance(last_layer, nn.Linear):
                    nn.init.normal_(last_layer.weight, std=1e-4)
        elif isinstance(module, StructureEncoder):
            if hasattr(module, 's'):
                nn.init.constant_(module.s.weight, 0.0)
                if module.s.bias is not None:
                    nn.init.constant_(module.s.bias, 0.0)
        elif isinstance(module, MLPDecoder):
            last_linear = module.readout[-1]
            if isinstance(last_linear, nn.Linear):
                nn.init.normal_(last_linear.weight, std=0.01)

    @torch.inference_mode()
    def test_sample_peptide_sequences(self, raw_batch, inference_peptide_chain_ids=None, sample_temperature=0.1, num_samples=10):
        self.eval()
        results = self(raw_batch, inference_peptide_chain_ids)
        if not results:
            return None

        y_true = results['y_true']
        logits = results['logits'].clone()

        X_index = AA_TO_IDX['X']
        logits[..., X_index] = -1e9
        ref_log_probs = F.log_softmax(logits, dim=-1)

        if sample_temperature < 1e-9:
            greedy_tokens = torch.argmax(logits, dim=-1)
            log_probs_for_greedy = torch.gather(ref_log_probs, dim=-1, index=greedy_tokens.unsqueeze(-1)).squeeze(-1)
            joint_log_likelihood = log_probs_for_greedy.sum().item()
            greedy_tokens_np = greedy_tokens.cpu().numpy()
            all_sampled_tokens = [greedy_tokens_np for _ in range(num_samples)]
            all_sampled_log_likelihoods = [joint_log_likelihood for _ in range(num_samples)]
        else:
            scaled_logits = logits / sample_temperature
            probs = F.softmax(scaled_logits, dim=-1)
            sampled_tokens = torch.multinomial(probs, num_samples=num_samples, replacement=True).transpose(0, 1)
            log_probs_for_sampled = torch.gather(
                ref_log_probs.unsqueeze(0).expand(num_samples, -1, -1),
                dim=-1,
                index=sampled_tokens.unsqueeze(-1)
            ).squeeze(-1)
            joint_log_likelihoods = log_probs_for_sampled.sum(dim=-1).cpu().tolist()
            all_sampled_tokens = [sampled_tokens[i].cpu().numpy() for i in range(num_samples)]
            all_sampled_log_likelihoods = joint_log_likelihoods

        return {
            'y_true': y_true.cpu().numpy(),
            'all_sampled_tokens': all_sampled_tokens,
            'all_sampled_log_likelihoods': all_sampled_log_likelihoods
        }