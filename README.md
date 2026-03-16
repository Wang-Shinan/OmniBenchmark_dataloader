# Benchmark Dataloader

HDF5-based data infrastructure for EEG Foundation Model Benchmarking.

## Features

- Unified HDF5 format for EEG datasets
- PyTorch-compatible data loading
- Multiple split strategies (by subject, session, trial, random)
- Preprocessing pipeline with MNE

## Supported Datasets

| Dataset | Task | Classes | Subjects |
|---------|------|---------|----------|
| SEED | Emotion | 3 | 15 |
| SEED-IV | Emotion | 4 | 15 |
| SEED-V | Emotion | 5 | 16 |
| BCIC-2A | Motor Imagery | 4 | 9 |

## Installation

```bash
pip install numpy h5py torch mne scipy
```

## Quick Start

### 1. Build HDF5 from raw data

```python
from benchmark_dataloader.datasets import build_seediv

# Convert raw .mat files to HDF5
build_seediv(
    raw_data_dir="/path/to/SEED_IV/eeg_raw_data",
    output_dir="/path/to/output",
)
```

Or via command line:
```bash
python -m benchmark_dataloader.datasets.seed_iv /path/to/raw/data --output_dir /path/to/output
```

### 2. Load data for training

```python
from benchmark_dataloader import load_dataset, split_dataset, SplitStrategy

# Split by subject (recommended for cross-subject evaluation)
train_ds, val_ds, test_ds = split_dataset(
    data_dir="/path/to/hdf5/SEEDIV",
    split_strategy=SplitStrategy.BY_SUBJECT,
    split_ratio=(0.7, 0.15, 0.15),
)

print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

# Get a batch
item = train_ds[0]
print(item['data'].shape)  # (n_channels, n_timepoints)
print(item['label'])       # class label
```

### 3. Use with PyTorch DataLoader

```python
from benchmark_dataloader import load_dataset, SplitStrategy

train_loader = load_dataset(
    data_dir="/path/to/hdf5/SEEDIV",
    split="train",
    split_strategy=SplitStrategy.BY_SUBJECT,
    batch_size=32,
    num_workers=4,
)

for batch in train_loader:
    data = batch['data']      # (batch, channels, timepoints)
    labels = batch['label']   # (batch,)
    # ... training code
```

## Split Strategies

- `BY_SUBJECT`: Entire subjects go to train/val/test (cross-subject evaluation)
- `BY_SESSION`: Sessions within each subject are split
- `BY_TRIAL`: Trials within each session are split
- `RANDOM`: Random split of all segments

## Project Structure

```
benchmark_dataloader/
├── __init__.py          # Public API
├── config.py            # Configuration classes
├── schema.py            # Data schema definitions
├── hdf5_io.py           # HDF5 read/write utilities
├── loader.py            # PyTorch dataset and loaders
├── datasets/            # Dataset builders
│   ├── seed.py
│   ├── seed_iv.py
│   ├── seedv.py
│   └── bcic_2a.py
└── docs/
    └── ADDING_NEW_DATASET.md  # Guide for adding datasets
```

## Adding New Datasets

See [docs/ADDING_NEW_DATASET.md](docs/ADDING_NEW_DATASET.md) for a guide on adding new EEG datasets.
