## testing 
## KL 2026-01-06

import sys
sys.path.insert(0, '.')

from pathlib import Path
from loader import load_dataset, split_dataset, SplitStrategy
from hdf5_io import HDF5Reader
import numpy as np

def detect_unit_and_convert_to_volts(data: np.ndarray) -> tuple[np.ndarray, str]:
    """Auto-detect data unit and convert to Volts for MNE."""
    max_amp = np.abs(data).max()
    if max_amp > 1e-2:  # likely µV
        return data / 1e6, "µV"
    elif max_amp > 1e-5:  # likely mV
        return data / 1e3, "mV"
    return data, "V"

data_dir = "/mnt/dataset2/Processed_datasets/EEG_Bench/SEEDIV"
# data_dir = '/mnt/dataset2/Processed_datasets/EEG_Bench/SEED'  # Alternative dataset
# data_dir = '/mnt/dataset2/Processed_datasets/EEG_Bench/SEEDV'  # Alternative dataset
batch_size = 32

loader = load_dataset(data_dir, batch_size=batch_size, num_workers=0)
print(f"Total samples: {len(loader.dataset)}")

# Show metadata from first HDF5 file
h5_files = sorted(Path(data_dir).glob("sub_*.h5"))
if h5_files:
    print("\n" + "="*50)
    print("METADATA FROM FIRST HDF5 FILE")
    print("="*50)
    with HDF5Reader(str(h5_files[0])) as reader:
        subj = reader.subject_attrs
        print(f"Dataset name: {subj.dataset_name}")
        print(f"Task type: {subj.task_type}")
        print(f"Downstream task: {subj.downstream_task_type}")
        print(f"Subject ID: {subj.subject_id}")
        print(f"Sampling rate: {subj.rsFreq} Hz")
        print(f"Montage: {subj.montage}")
        print(f"Channels ({len(subj.chn_name)}): {subj.chn_name[:5]}...")
        print(f"Num trials: {len(reader.get_trial_names())}")
        print(f"Num segments: {len(reader)}")

        trial_name = reader.get_trial_names()[0]
        seg_name = reader.get_segment_names(trial_name)[0]
        seg = reader.get_segment(trial_name, seg_name)
        print(f"\nFirst segment:")
        print(f"  Trial ID: {seg.trial.trial_id}, Session ID: {seg.trial.session_id}")
        print(f"  Segment ID: {seg.segment.segment_id}")
        print(f"  Time: {seg.segment.start_time:.2f}s - {seg.segment.end_time:.2f}s")
        print(f"  Duration: {seg.segment.time_length}s")
        print(f"  Label: {seg.segment.label}")
        print(f"  Data shape: {seg.data.shape}")

# Test loading batches
print("\n" + "="*50)
print("BATCH LOADING TEST")
print("="*50)
for i, batch in enumerate(loader):
    if i >= 3:
        break
    print(f"\nBatch {i}:")
    data_volts, unit = detect_unit_and_convert_to_volts(batch['data'].numpy())
    print(f"  Data shape: {data_volts.shape} (unit: {unit})")
    print(f"  Labels: {batch['label'].squeeze().tolist()[:5]}...")
    print(f"  Subject IDs: {batch['subject_id'].tolist()[:5]}...")

# Test split strategies
print("\n" + "="*50)
print("SPLIT STRATEGY TEST")
print("="*50)

for strategy in SplitStrategy:
    print(f"\n--- {strategy.value.upper()} ---")
    train_ds, val_ds, test_ds = split_dataset(data_dir, strategy)
    total = len(train_ds) + len(val_ds) + len(test_ds)
    print(f"Train: {len(train_ds)} ({100*len(train_ds)/total:.1f}%)")
    print(f"Val:   {len(val_ds)} ({100*len(val_ds)/total:.1f}%)")
    print(f"Test:  {len(test_ds)} ({100*len(test_ds)/total:.1f}%)")

    # Show which subjects/sessions are in each split
    train_subjs = set(idx[0] for idx in train_ds.index)
    val_subjs = set(idx[0] for idx in val_ds.index)
    test_subjs = set(idx[0] for idx in test_ds.index)
    print(f"Train subjects: {sorted(train_subjs)}")
    print(f"Val subjects:   {sorted(val_subjs)}")
    print(f"Test subjects:  {sorted(test_subjs)}")

    if strategy == SplitStrategy.BY_SUBJECT:
        # Verify no overlap
        assert train_subjs.isdisjoint(val_subjs), "Train/Val subjects overlap!"
        assert train_subjs.isdisjoint(test_subjs), "Train/Test subjects overlap!"
        assert val_subjs.isdisjoint(test_subjs), "Val/Test subjects overlap!"
        print("✓ No subject overlap between splits")

    elif strategy == SplitStrategy.BY_SESSION:
        # Check session distribution within subjects
        train_sessions = set((idx[0], idx[3]) for idx in train_ds.index)
        val_sessions = set((idx[0], idx[3]) for idx in val_ds.index)
        test_sessions = set((idx[0], idx[3]) for idx in test_ds.index)
        print(f"Train sessions (subj, sess): {len(train_sessions)} unique")
        print(f"Val sessions:   {len(val_sessions)} unique")
        print(f"Test sessions:  {len(test_sessions)} unique")
