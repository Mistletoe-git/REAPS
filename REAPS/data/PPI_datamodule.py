import json
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import Dataset, DataLoader, Sampler
from tqdm import tqdm


def simple_collate_fn(batch):
    return batch


class PPI_Dataset(Dataset):
    """
    Loads a single pre-processed .pkl file from disk by its index.
    """
    def __init__(self, pkl_files: List[Path]):
        super().__init__()
        self.pkl_files = pkl_files

    def __len__(self) -> int:
        return len(self.pkl_files)

    def __getitem__(self, idx: int) -> dict:
        file_path = self.pkl_files[idx]
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data


class DynamicTokenBatchSampler(Sampler):
    """
    Sampler that supports both dynamic (per-cluster) and static sampling,
    as well as token-based batching.
    """
    def __init__(self,
                 sample_sizes: np.ndarray,
                 indices_to_use: List[int],
                 max_tokens: int,
                 shuffle: bool = True,
                 num_replicas: int = 1,
                 rank: int = 0,
                 seed: int = 42,
                 clusters: Optional[List[List[int]]] = None):
        super().__init__(data_source=None)
        if not isinstance(num_replicas, int) or num_replicas <= 0:
            raise ValueError("num_replicas must be a positive integer")
        if not isinstance(rank, int) or rank < 0 or rank >= num_replicas:
            raise ValueError("rank must be non-negative and less than num_replicas")

        self.rank = rank
        self.num_replicas = num_replicas
        self.sample_sizes = sample_sizes
        self.indices_to_use = indices_to_use
        self.max_tokens = max_tokens
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.clusters = clusters

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        active_indices = []
        if self.clusters is not None and self.shuffle:
            for cluster_members in self.clusters:
                if not cluster_members:
                    continue
                rand_idx = torch.randint(0, len(cluster_members), (1,), generator=g).item()
                active_indices.append(cluster_members[rand_idx])
        else:
            active_indices = self.indices_to_use

        current_epoch_sizes = self.sample_sizes[active_indices]
        sorted_order = np.argsort(current_epoch_sizes)
        sorted_active_indices = [active_indices[i] for i in sorted_order]

        batches, current_batch, current_batch_tokens = [], [], 0
        for idx in sorted_active_indices:
            size = self.sample_sizes[idx]
            if current_batch_tokens + size > self.max_tokens and current_batch:
                batches.append(current_batch)
                current_batch, current_batch_tokens = [], 0
            current_batch.append(idx)
            current_batch_tokens += size
        if current_batch:
            batches.append(current_batch)

        if self.shuffle:
            shuffled_batch_indices = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in shuffled_batch_indices]

        self.global_batches = batches
        num_global_batches_before_padding = len(self.global_batches)
        if num_global_batches_before_padding % self.num_replicas != 0:
            num_to_add = self.num_replicas - (num_global_batches_before_padding % self.num_replicas)
            padding_batches = self.global_batches[-num_to_add:] if num_to_add < len(self.global_batches) else self.global_batches
            self.global_batches.extend(padding_batches[:num_to_add])

        self.num_local_batches = len(self.global_batches) // self.num_replicas

        local_batches = self.global_batches[self.rank::self.num_replicas]
        return iter(local_batches)

    def __len__(self):
        if hasattr(self, 'num_local_batches'):
            return self.num_local_batches
        avg_size = self.sample_sizes[self.indices_to_use].mean()
        num_samples_per_batch = self.max_tokens / avg_size
        num_batches = len(self.indices_to_use) / num_samples_per_batch
        return int(num_batches // self.num_replicas)

    def set_epoch(self, epoch):
        self.epoch = epoch

class PPI_DataModule(LightningDataModule):
    def __init__(self,
                 data_dirs: List[str],
                 splits_json_path: str,
                 cluster_map_path:str,
                 mode: str,
                 max_tokens_per_batch: int,
                 num_workers: int,
                 pin_memory: bool,
                 max_length: int = 6000,
                 seed: int = 42):

        super().__init__()
        self.save_hyperparameters()
        self.consolidated_metadata_path = Path(self.hparams.splits_json_path).parent / "consolidated_metadata.csv"

        self.master_dataset: Optional[PPI_Dataset] = None
        self.samplers = {}
        self.datasets = {}

    def prepare_data(self):
        if self.consolidated_metadata_path.exists():
            print(f"Found existing metadata file: {self.consolidated_metadata_path}. Skipping scan.")
            return

        print(f"Metadata file not found. Scanning .pkl files to create: {self.consolidated_metadata_path}")
        metadata_list = []
        skipped_count = 0
        for data_dir in self.hparams.data_dirs:
            pkl_dir = Path(data_dir)
            if not pkl_dir.exists():
                raise FileNotFoundError(f"Data directory not found: {pkl_dir}")

            for pkl_file in tqdm(list(pkl_dir.glob("*.pkl")), desc=f"Scanning {pkl_dir.name}"):
                try:
                    with open(pkl_file, 'rb') as f: data = pickle.load(f)
                    total_length = sum(len(chain['seq']) for chain in data['chain_features'])
                    if total_length <= self.hparams.max_length:
                        complex_id = pkl_file.stem
                        metadata_list.append({
                            'complex_id': complex_id,
                            'total_length': total_length,
                            'path': str(pkl_file.resolve())
                        })
                    else:
                        skipped_count += 1
                except Exception as e:
                    print(f"Warning: Could not process file {pkl_file.name}. Error: {e}")

        df_meta = pd.DataFrame(metadata_list)
        df_meta.to_csv(self.consolidated_metadata_path, index=False)
        print(f"Consolidated metadata saved with {len(df_meta)} entries.")
        if skipped_count > 0:
            print(f"Skipped {skipped_count} samples with total length > {self.hparams.max_length}.")

    def setup(self, stage: Optional[str] = None):
        if self.master_dataset:
            return

        print("--- Running PPI_DataModule setup ---")
        df_meta = pd.read_csv(self.consolidated_metadata_path)
        complex_id_to_idx = {cid: i for i, cid in enumerate(df_meta['complex_id'])}

        with open(self.hparams.splits_json_path, 'r') as f:
            all_splits = json.load(f)

        print("Creating master dataset with all samples...")
        all_pkl_files = [Path(p) for p in df_meta['path']]
        self.master_dataset = PPI_Dataset(all_pkl_files)

        splits = all_splits[self.hparams.mode]

        train_cluster_map = None
        cluster_map_path = Path(self.hparams.cluster_map_path)
        if cluster_map_path.exists():
            with open(cluster_map_path, 'r') as f:
                train_cluster_map = json.load(f)

        is_main_process = not self.trainer or self.trainer.is_global_zero

        if is_main_process:
            print("\n" + "="*35)
            print("  Dataset Split Summary  ")
            print("="*35)
            valid_complex_ids_in_meta = set(df_meta['complex_id'])
            for split_mode, splits_in_mode in all_splits.items():
                print(f"Mode: {split_mode}")
                total_in_mode = 0
                for stage_name, cids in splits_in_mode.items():
                    valid_count = len([cid for cid in cids if cid in valid_complex_ids_in_meta])
                    print(f"  - {stage_name.capitalize():<12}: {valid_count} samples")
                    total_in_mode += valid_count
                print(f"  - Total for mode: {total_in_mode} samples")
            print("="*35 + "\n")

        for stage_name in ['train', 'validation', 'test']:
            if stage_name in splits:
                self.datasets[stage_name] = self.master_dataset

                complex_ids = splits[stage_name]
                current_stage_indices = [complex_id_to_idx[cid] for cid in complex_ids if cid in complex_id_to_idx]

                if is_main_process:
                    print(f"[{stage_name}] Found {len(current_stage_indices)} valid samples.")

                sampler_clusters = None
                if self.hparams.mode == 'pre-training' and stage_name == 'train' and train_cluster_map:
                    if is_main_process:
                        print(f"[{stage_name}] Activating Dynamic Sampling for pre-training.")
                    sampler_clusters = []
                    for rep, members in train_cluster_map.items():
                        member_indices = [complex_id_to_idx[cid] for cid in members if cid in complex_id_to_idx]
                        if member_indices:
                            sampler_clusters.append(member_indices)

                self.samplers[stage_name] = DynamicTokenBatchSampler(
                    sample_sizes=df_meta['total_length'].to_numpy(),
                    indices_to_use=current_stage_indices,
                    max_tokens=self.hparams.max_tokens_per_batch,
                    shuffle=(stage_name == 'train'),
                    clusters=sampler_clusters,
                    num_replicas=self.trainer.world_size if self.trainer else 1,
                    rank=self.trainer.global_rank if self.trainer else 0,
                    seed=self.hparams.seed
                )

    def _get_dataloader(self, stage: str) -> Optional[DataLoader]:
        if stage not in self.samplers:
            return None

        dataset = self.datasets.get(stage)
        print(f"--- Creating dataloader for stage: {stage}. Dataset object is: {type(dataset)} ---")
        if dataset is None:
            raise ValueError(f"Dataset for stage '{stage}' is None. This should not happen.")

        return DataLoader(
            dataset,
            batch_sampler=self.samplers[stage],
            num_workers=self.hparams.num_workers,
            collate_fn=simple_collate_fn,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.num_workers > 0
        )

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloader('train')

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloader('validation')

    def test_dataloader(self) -> DataLoader:
        return self._get_dataloader('test')
