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
from REAPS.data.PPI_datamodule import simple_collate_fn


class CPCore_Dataset(Dataset):
    """
    Dataset for CPCore samples.
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

        cyclic_type = data['metadata'].get('cyclic_type', 'UNKNOWN')

        rec = data['receptor_R']
        pep = data['peptide_L']

        rec_seq = rec['seq']
        pep_seq = pep['seq']
        rec_len = len(rec_seq)
        pep_len = len(pep_seq)

        rec_enc = rec.get('chain_encoding', np.zeros(rec_len, dtype=np.int64))
        pep_enc = pep.get('chain_encoding', np.ones(pep_len, dtype=np.int64))
        chain_encoding = np.concatenate([rec_enc, pep_enc])

        rec_indices = rec['R_idx'].astype(np.int64)
        pep_indices = np.arange(pep_len, dtype=np.int64)
        seq_indices = np.concatenate([rec_indices, pep_indices])

        xyz_37 = np.concatenate([rec['xyz_37'], pep['xyz_37']], axis=0)
        xyz_37_mask = np.concatenate([rec['xyz_37_mask'], pep['xyz_37_mask']], axis=0)
        backbone_mask = np.concatenate([rec['mask'], pep['mask']], axis=0)

        return {
            'complex_id': data['complex_id'],
            'cyclic_type': cyclic_type,
            'seq': rec_seq + pep_seq,
            'xyz_37': xyz_37,
            'xyz_37_mask': xyz_37_mask,
            'backbone_mask': backbone_mask,
            'chain_encoding': chain_encoding,
            'seq_indices': seq_indices,
            'rec_len': rec_len,
            'pep_len': pep_len
        }


class TokenBatchSampler(Sampler):
    def __init__(self,
                 sample_sizes: np.ndarray,
                 indices_to_use: List[int],
                 max_tokens: int,
                 shuffle: bool = True,
                 num_replicas: int = 1,
                 rank: int = 0,
                 seed: int = 42):
        super().__init__(data_source=None)

        self.rank = rank
        self.num_replicas = num_replicas
        self.sample_sizes = sample_sizes
        self.indices_to_use = indices_to_use
        self.max_tokens = max_tokens
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        active_indices = self.indices_to_use.copy()

        if self.shuffle:
            # Add small noise to sizes during shuffle to vary batching across epochs
            noise = torch.rand(len(active_indices), generator=g).numpy() * 20
            current_epoch_sizes = self.sample_sizes[active_indices] + noise
        else:
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
        num_global_batches = len(self.global_batches)

        # DDP Padding
        if num_global_batches % self.num_replicas != 0:
            num_to_add = self.num_replicas - (num_global_batches % self.num_replicas)
            self.global_batches.extend(self.global_batches[:num_to_add])

        self.num_local_batches = len(self.global_batches) // self.num_replicas
        local_batches = self.global_batches[self.rank::self.num_replicas]

        return iter(local_batches)

    def __len__(self):
        if hasattr(self, 'num_local_batches'):
            return self.num_local_batches
        return len(self.indices_to_use) // self.num_replicas # Rough estimate

    def set_epoch(self, epoch):
        self.epoch = epoch


class CPCore_DataModule(LightningDataModule):
    """
    DataModule for CPCore Fine-tuning and Testing.
    Now automatically routes data into Train, Val, and Test based on data_splits_purified.json.
    """
    def __init__(self,
                 data_dirs: List[str],
                 split_json_path: str,
                 max_tokens_per_batch: int = 8000,
                 num_workers: int = 8,
                 pin_memory: bool = True,
                 max_length: int = 3000,
                 seed: int = 3407):

        super().__init__()
        self.save_hyperparameters()
        self.metadata_path = Path(data_dirs[0]).resolve().parent / "CPCore_global_metadata.csv"

        self.train_dataset: Optional[CPCore_Dataset] = None
        self.train_sampler: Optional[TokenBatchSampler] = None

        self.val_dataset: Optional[CPCore_Dataset] = None
        self.val_sampler: Optional[TokenBatchSampler] = None

        self.test_dataset: Optional[CPCore_Dataset] = None
        self.test_sampler: Optional[TokenBatchSampler] = None

    def prepare_data(self):
        """Scans directories and creates a global metadata file if it doesn't exist."""
        if self.metadata_path.exists():
            print(f"[*] Found existing CPCore global metadata: {self.metadata_path}")
            return

        print(f"[*] Scanning CPCore .pkl files...")
        metadata_list = []
        for d in self.hparams.data_dirs:
            pkl_dir = Path(d)
            for pkl_file in tqdm(list(pkl_dir.glob("*.pkl")), desc=f"Scanning {pkl_dir.name}"):
                try:
                    with open(pkl_file, 'rb') as f:
                        data = pickle.load(f)
                    total_len = len(data['receptor_R']['seq']) + len(data['peptide_L']['seq'])

                    if total_len <= self.hparams.max_length:
                        metadata_list.append({
                            'complex_id': data['complex_id'],
                            'total_length': total_len,
                            'path': str(pkl_file.resolve())
                        })
                except Exception as e:
                    print(f"Warning: Failed to process {pkl_file.name}: {e}")

        df_meta = pd.DataFrame(metadata_list)
        df_meta.to_csv(self.metadata_path, index=False)
        print(f"[*] Global metadata saved: {len(df_meta)} total samples found.")

    def setup(self, stage: Optional[str] = None):
        df_meta = pd.read_csv(self.metadata_path)

        with open(self.hparams.split_json_path, 'r') as f:
            splits = json.load(f)

        train_ids = set(splits.get('train', []))
        val_ids = set(splits.get('validation', []))
        test_ids = set(splits.get('test', []))

        df_train = df_meta[df_meta['complex_id'].isin(train_ids)].reset_index(drop=True)
        df_val = df_meta[df_meta['complex_id'].isin(val_ids)].reset_index(drop=True)
        df_test = df_meta[df_meta['complex_id'].isin(test_ids)].reset_index(drop=True)

        self.train_dataset = CPCore_Dataset(pkl_files=[Path(p) for p in df_train['path']])
        self.val_dataset = CPCore_Dataset(pkl_files=[Path(p) for p in df_val['path']])
        self.test_dataset = CPCore_Dataset(pkl_files=[Path(p) for p in df_test['path']])

        world_size = self.trainer.world_size if self.trainer else 1
        global_rank = self.trainer.global_rank if self.trainer else 0

        self.train_sampler = TokenBatchSampler(
            sample_sizes=df_train['total_length'].to_numpy(),
            indices_to_use=list(range(len(df_train))),
            max_tokens=self.hparams.max_tokens_per_batch,
            shuffle=True,
            num_replicas=world_size,
            rank=global_rank,
            seed=self.hparams.seed
        )

        self.val_sampler = TokenBatchSampler(
            sample_sizes=df_val['total_length'].to_numpy(),
            indices_to_use=list(range(len(df_val))),
            max_tokens=self.hparams.max_tokens_per_batch,
            shuffle=False,
            num_replicas=world_size,
            rank=global_rank,
            seed=self.hparams.seed
        )

        self.test_sampler = TokenBatchSampler(
            sample_sizes=df_test['total_length'].to_numpy(),
            indices_to_use=list(range(len(df_test))),
            max_tokens=self.hparams.max_tokens_per_batch,
            shuffle=False,
            num_replicas=world_size,
            rank=global_rank,
            seed=self.hparams.seed
        )

        if not self.trainer or self.trainer.is_global_zero:
            print(f"[*] CPCore DataModule setup complete:")
            print(f"    - Training samples: {len(df_train)}")
            print(f"    - Validation samples: {len(df_val)}")
            print(f"    - Testing samples: {len(df_test)}")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_sampler=self.train_sampler,
            num_workers=self.hparams.num_workers,
            collate_fn=simple_collate_fn,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.num_workers > 0
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_sampler=self.val_sampler,
            num_workers=self.hparams.num_workers,
            collate_fn=simple_collate_fn,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.num_workers > 0
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_sampler=self.test_sampler,
            num_workers=self.hparams.num_workers,
            collate_fn=simple_collate_fn,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.num_workers > 0
        )