
# EEG-FM-Bench Benchmark Dataloader Plan

## Goal
Data infrastructure for EEG benchmarking using HDF5 format with clean metadata schema.

## Key Design Decisions
- **Supervised only**: No unsupervised data support
- **Naming**: Concat dataset and experiment into one name (e.g., "SEED_3class")
- **Terminology**: Use "segment" instead of "sample" throughout
- **Segment attrs**: Include `start_time` and `end_time` (in seconds)
- **Flexible splits**: Dataset splitting via split function in eeg-fm-bench

---

## HDF5 Structure (per subject: `sub_{sub_id}.h5`)
```
sub_0.h5
├── attrs (subject-level):
│     subject_id: int
│     dataset_name: str          # e.g., "SEED_3class" (concat)
│     task_type: str
│     downstream_task_type: str  # NEW: downstream task type
│     rsFreq: float
│     chn_name, chn_pos, chn_ori, chn_type, montage
│
├── /trial0 (group)
│   ├── attrs: trial_id, session_id
│   ├── /segment0 (group)        # renamed from sample
│   │   └── eeg (dataset)
│   │       attrs:
│   │         segment_id: int
│   │         start_time: float  # NEW
│   │         end_time: float    # NEW
│   │         time_length: float
│   │         label: array
│   └── /segmentN ...
└── /trialM ...
```

---

## Changes Required

### 1. Schema (`schema.py`)
- `SampleAttrs` → `SegmentAttrs`
- Add `start_time`, `end_time` to `SegmentAttrs`
- Add `downstream_task_type` to `SubjectAttrs`

### 2. HDF5 I/O (`hdf5_io.py`)
- Rename sample → segment in group names
- Write/read new attrs

### 3. Loader (`loader.py`)
- Add split function for eeg-fm-bench
- Add split strategies: `random`, `by_subject`, `by_session`, `by_trial`
- Add `split_ratio`, `split_strategy` parameters

### 4. Dataset Builders (`datasets/*.py`)
- Update to segment naming
- Add downstream_task_type
- Calculate start_time/end_time

---

## Split Strategies
```python
class SplitStrategy(Enum):
    RANDOM = "random"
    BY_SUBJECT = "by_subject"
    BY_SESSION = "by_session"
    BY_TRIAL = "by_trial"

load_dataset(
    data_dir, split="train",
    split_strategy=SplitStrategy.BY_SUBJECT,
    split_ratio=(0.7, 0.15, 0.15),
    seed=42
)
```

---

## TODO
- [ ] Update schema.py: SampleAttrs → SegmentAttrs, add fields
- [ ] Update hdf5_io.py: sample → segment
- [ ] Update loader.py: add split function and strategies
- [ ] Update config.py: add downstream_task_type
- [ ] Update all dataset builders
