"""
Dataset loader for HDF5 EEG data.

Provides PyTorch-compatible dataset classes for loading EEG segments.
TODO: discuss about the current setting
"""

from enum import Enum
from pathlib import Path
from typing import Optional, Callable, Tuple
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:

    HAS_TORCH = False
    Dataset = object
    DataLoader = object

try:
    from .hdf5_io import HDF5Reader
    from .schema import EEGSegment
except ImportError:
    from hdf5_io import HDF5Reader
    from schema import EEGSegment


class SplitStrategy(Enum):
    """Split strategies for dataset partitioning."""
    RANDOM = "random"
    BY_SUBJECT = "by_subject"
    BY_SESSION = "by_session"  # Split sessions within each subject
    BY_TRIAL = "by_trial"  # Split trials within each session


class EEGDataset(Dataset):
    """PyTorch Dataset for loading EEG segments from HDF5 files."""

    def __init__(
        self,
        data_dir: str,
        indices: Optional[list] = None,
        transform: Optional[Callable] = None,
    ):
        """
        Initialize EEG dataset.

        Args:
            data_dir: Directory containing HDF5 files
            indices: Pre-computed indices (file_idx, trial_name, segment_name, session_id, trial_id)
            transform: Optional transform to apply to segments
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for EEGDataset")

        self.data_dir = Path(data_dir)
        self.transform = transform
        self.h5_files = sorted(self.data_dir.glob("sub_*.h5"))
        self.index = indices if indices is not None else []

    @staticmethod
    def _get_subject_id(filepath: Path) -> int | str:
        """Extract subject ID from filename."""
        stem = filepath.stem
        if stem.startswith("sub-"):
            return stem
        if stem.startswith("sub_"):
            suffix = stem.split("_", 1)[1]
            try:
                return int(suffix)
            except ValueError:
                return suffix
        return stem

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        file_idx, trial_name, segment_name, _, _ = self.index[idx]
        h5_file = self.h5_files[file_idx]

        with HDF5Reader(str(h5_file)) as reader:
            segment = reader.get_segment(trial_name, segment_name)

        item = {
            'data': torch.from_numpy(segment.data).float(),
            'label': torch.from_numpy(segment.segment.label),
            'subject_id': segment.subject.subject_id,
            'trial_id': segment.trial.trial_id,
            'session_id': segment.trial.session_id,
            'segment_id': segment.segment.segment_id,
            'start_time': segment.segment.start_time,
            'end_time': segment.segment.end_time,
        }

        if self.transform:
            item = self.transform(item)

        return item


def _build_full_index(data_dir: str) -> Tuple[list, list]:
    """Build full index of all segments and return h5_files list."""
    data_path = Path(data_dir)
    h5_files = sorted(data_path.glob("sub_*.h5"))
    index = []

    for file_idx, h5_file in enumerate(h5_files):
        with HDF5Reader(str(h5_file)) as reader:
            for trial_name in reader.get_trial_names():
                trial_attrs = reader.get_trial_attrs(trial_name)
                for segment_name in reader.get_segment_names(trial_name):
                    index.append((
                        file_idx, trial_name, segment_name,
                        trial_attrs.session_id, trial_attrs.trial_id
                    ))

    return index, h5_files


def _split_list(items: list, ratios: Tuple[float, float, float], rng) -> Tuple[list, list, list]:
    """Split a list into train/val/test based on ratios."""
    items = list(items)
    rng.shuffle(items)
    n = len(items)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    return items[:n_train], items[n_train:n_train + n_val], items[n_train + n_val:]


def split_dataset(
    data_dir: str,
    split_strategy: SplitStrategy = SplitStrategy.BY_SUBJECT,
    split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
    transform: Optional[Callable] = None,
) -> Tuple[EEGDataset, EEGDataset, EEGDataset]:
    """
    Split dataset into train/val/test sets.

    Split strategies:
    - BY_SUBJECT: Entire subjects go to train/val/test
    - BY_SESSION: Within each subject, sessions are split to train/val/test
    - BY_TRIAL: Within each session, trials are split to train/val/test
    - RANDOM: Random split of all segments

    Args:
        data_dir: Directory containing HDF5 files
        split_strategy: How to split the data
        split_ratio: (train, val, test) ratios
        seed: Random seed for reproducibility
        transform: Optional transform to apply

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch is required for split_dataset")

    rng = np.random.default_rng(seed)
    full_index, h5_files = _build_full_index(data_dir)

    train_indices, val_indices, test_indices = [], [], []

    if split_strategy == SplitStrategy.BY_SUBJECT:
        # Split by subject - entire subjects go to one split
        subject_to_indices = {}
        for entry in full_index:
            file_idx = entry[0]
            subject_id = EEGDataset._get_subject_id(h5_files[file_idx])
            subject_to_indices.setdefault(subject_id, []).append(entry)

        train_subs, val_subs, test_subs = _split_list(
            subject_to_indices.keys(), split_ratio, rng
        )

        for sub in train_subs:
            train_indices.extend(subject_to_indices[sub])
        for sub in val_subs:
            val_indices.extend(subject_to_indices[sub])
        for sub in test_subs:
            test_indices.extend(subject_to_indices[sub])

    elif split_strategy == SplitStrategy.BY_SESSION:
        # Split sessions within each subject
        # Group by (subject, session)
        subject_sessions = {}
        for entry in full_index:
            file_idx, _, _, session_id, _ = entry
            subject_id = EEGDataset._get_subject_id(h5_files[file_idx])
            key = (subject_id, session_id)
            subject_sessions.setdefault(key, []).append(entry)

        # Group sessions by subject
        subject_to_sessions = {}
        for (sub, sess), entries in subject_sessions.items():
            subject_to_sessions.setdefault(sub, {})[sess] = entries

        # For each subject, split their sessions
        for sub, sessions in subject_to_sessions.items():
            session_ids = list(sessions.keys())
            train_sess, val_sess, test_sess = _split_list(session_ids, split_ratio, rng)

            for sess in train_sess:
                train_indices.extend(sessions[sess])
            for sess in val_sess:
                val_indices.extend(sessions[sess])
            for sess in test_sess:
                test_indices.extend(sessions[sess])

    elif split_strategy == SplitStrategy.BY_TRIAL:
        ## TODO: NEED TO DISCUSS and specify the requirements
        # Split trials within each session
        # Group by (subject, session, trial)
        session_trials = {}
        for entry in full_index:
            file_idx, _, _, session_id, trial_id = entry
            subject_id = EEGDataset._get_subject_id(h5_files[file_idx])
            key = (subject_id, session_id, trial_id)
            session_trials.setdefault(key, []).append(entry)

        # Group trials by (subject, session)
        session_to_trials = {}
        for (sub, sess, trial), entries in session_trials.items():
            session_to_trials.setdefault((sub, sess), {})[trial] = entries

        # For each session, split their trials
        for (sub, sess), trials in session_to_trials.items():
            trial_ids = list(trials.keys())
            train_trials, val_trials, test_trials = _split_list(trial_ids, split_ratio, rng)

            for t in train_trials:
                train_indices.extend(trials[t])
            for t in val_trials:
                val_indices.extend(trials[t])
            for t in test_trials:
                test_indices.extend(trials[t])

    else:  # RANDOM
        train_indices, val_indices, test_indices = _split_list(
            full_index, split_ratio, rng
        )

    def make_dataset(indices):
        ds = EEGDataset(data_dir, indices=indices, transform=transform)
        ds.h5_files = h5_files
        return ds

    return make_dataset(train_indices), make_dataset(val_indices), make_dataset(test_indices)


class MultiDatasetLoader:
    ## TODO: need meta-dataset implementation
    """Load and combine multiple EEG datasets."""

    def __init__(
        self,
        data_root: str,
        dataset_names: list[str],
        batch_size: int = 32,
        num_workers: int = 4,
        shuffle: bool = True,
    ):
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for MultiDatasetLoader")

        self.data_root = Path(data_root)
        self.datasets = []

        for name in dataset_names:
            data_dir = self.data_root / name
            if data_dir.exists():
                full_index, h5_files = _build_full_index(str(data_dir))
                ds = EEGDataset(str(data_dir), indices=full_index)
                ds.h5_files = h5_files
                self.datasets.append(ds)

        self.combined = torch.utils.data.ConcatDataset(self.datasets)
        self.dataloader = DataLoader(
            self.combined,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            pin_memory=True,
        )

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader)


def load_dataset(
    data_dir: str,
    split: Optional[str] = None,
    split_strategy: SplitStrategy = SplitStrategy.BY_SUBJECT,
    split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
    transform: Optional[Callable] = None,
) -> DataLoader:
    """
    Load an EEG dataset with optional splitting.

    Args:
        data_dir: Directory containing HDF5 files
        split: Which split to return ("train", "val", "test", or None for all)
        split_strategy: How to split the data
        split_ratio: (train, val, test) ratios
        seed: Random seed for reproducibility
        batch_size: Batch size
        num_workers: Number of worker processes
        shuffle: Whether to shuffle
        transform: Optional transform to apply

    Returns:
        PyTorch DataLoader
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch is required for load_dataset")

    if split is not None:
        train_ds, val_ds, test_ds = split_dataset(
            data_dir, split_strategy, split_ratio, seed, transform
        )
        dataset = {"train": train_ds, "val": val_ds, "test": test_ds}[split]
    else:
        full_index, h5_files = _build_full_index(data_dir)
        dataset = EEGDataset(data_dir, indices=full_index, transform=transform)
        dataset.h5_files = h5_files

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test EEG data loading")
    parser.add_argument("data_dir", nargs="?", default="/mnt/dataset2/hdf5_datasets/SEEDIV")
    parser.add_argument("--strategy", default="by_subject",
                        choices=["random", "by_subject", "by_session", "by_trial"])
    args = parser.parse_args()

    strategy = SplitStrategy(args.strategy)
    train_ds, val_ds, test_ds = split_dataset(args.data_dir, strategy)
    print(f"Strategy: {strategy.value}")
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

