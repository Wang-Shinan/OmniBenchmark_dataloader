# EEG-FM-Bench Refactoring Plan

## Goal
repository to focus on data infrastructure only, using HDF5 format with a clean metadata schema. All new code goes in `benchmark_dataloader/` subfolder.

## Notes
- Unsupervised data: Include placeholder structure, implement later
- Supervised data: Full implementation as described below
- All new code in `benchmark_dataloader/` subfolder

## New Data Schema

### HDF5 Structure (per subject file: `sub_{sub_id}.h5`)
```
sub_0.h5
├── attrs (subject-level, shared across all samples):
│     subject_id: int
│     dataset_name: str
│     task_type: str
│     rsFreq: float
│     chn_name: str[]
│     chn_pos: array or 'None'
│     chn_ori: array or 'None'
│     chn_type: str
│     montage: str
│
├── /trial0 (group)
│   ├── attrs:
│   │     trial_id: int
│   │     session_id: int
│   │
│   ├── /sample0 (group)
│   │   └── eeg (dataset): float32[n_channels, n_timepoints]
│   │       attrs:
│   │         segment_id: int
│   │         time_length: float
│   │         label: array
│   ├── /sample1 ...
│   └── /sampleN ...
├── /trial1 ...
└── /trialM ...
```

### File Organization
```
hdf5/
  └── dataset_name/
      ├── sub_0.h5
      ├── sub_1.h5
      └── sub_N.h5
```

---

## Implementation Stages

### Stage 1: Core Infrastructure
**Files to create in `benchmark_dataloader/`:**
- `benchmark_dataloader/__init__.py` - Package init
- `benchmark_dataloader/schema.py` - Data classes (SubjectAttrs, TrialAttrs, SampleAttrs, EEGSample, DatasetConfig)
- `benchmark_dataloader/hdf5_io.py` - HDF5Writer and HDF5Reader classes
- `benchmark_dataloader/config.py` - Preprocessing config (from common/config.py)
- `benchmark_dataloader/utils.py` - ElectrodeSet, set_seed (from common/utils.py)

### Stage 2: Builder and Loader
**Files to create in `benchmark_dataloader/`:**
- `benchmark_dataloader/builder.py` - EEG dataset builder (simplified from data/processor/builder.py)
- `benchmark_dataloader/loader.py` - HDF5 dataset loading utilities

### Stage 3: Dataset Migration
**Files to create in `benchmark_dataloader/datasets/`:**
- Phase 1 (simple): bcic_2a, bcic_1a, seed, emobrain
- Phase 2 (medium): tuab, tusz, seed_iv, seed_v
- Phase 3 (complex): hbn, things_eeg

### Stage 4: Cleanup
**Delete entirely from root:**
- `baseline/` - All model implementations
- `baseline_main.py` - Model training entry point
- `plot/` - Visualization tools
- `visualize.py` - Visualization entry point
- `common/` - Moved to benchmark_dataloader
- `data/` - Moved to benchmark_dataloader
- `assets/conf/example/eegnet/`, `assets/conf/example/eegpt/` - Model configs

---

## Critical Files

| File | Action |
|------|--------|
| `benchmark_dataloader/schema.py` | Create - data classes |
| `benchmark_dataloader/hdf5_io.py` | Create - HDF5 read/write |
| `benchmark_dataloader/builder.py` | Create - preprocessing logic |
| `benchmark_dataloader/loader.py` | Create - dataset loading |
| `benchmark_dataloader/datasets/bcic_2a.py` | Create - reference implementation |

---

## New Folder Structure
```
EEG-FM-Bench/
├── benchmark_dataloader/
│   ├── __init__.py
│   ├── schema.py          # Data classes
│   ├── hdf5_io.py         # HDF5 read/write
│   ├── config.py          # Preprocessing config
│   ├── utils.py           # ElectrodeSet, utilities
│   ├── builder.py         # Dataset builder base
│   ├── loader.py          # Dataset loading
│   └── datasets/
│       ├── __init__.py
│       ├── bcic_2a.py
│       ├── bcic_1a.py
│       ├── seed.py
│       └── ...
├── hdf5/                  # Output data directory
│   └── dataset_name/
│       └── sub_*.h5
├── README.md
└── requirements.txt
```

---

## New Data Classes (benchmark_dataloader/schema.py)

```python
@dataclass
class SubjectAttrs:
    """Subject-level attributes (stored at file root)"""
    subject_id: int
    dataset_name: str
    task_type: str
    rsFreq: float
    chn_name: list[str]
    chn_pos: Optional[np.ndarray]  # None or (n_channels, 3)
    chn_ori: Optional[np.ndarray]  # None or (n_channels, 3)
    chn_type: str
    montage: str

@dataclass
class TrialAttrs:
    """Trial-level attributes"""
    trial_id: int
    session_id: int

@dataclass
class SampleAttrs:
    """Sample-level attributes"""
    segment_id: int
    time_length: float
    label: np.ndarray

@dataclass
class EEGSample:
    """Complete sample with inherited attributes from subject/trial"""
    data: np.ndarray  # (n_channels, n_timepoints)
    subject: SubjectAttrs
    trial: TrialAttrs
    sample: SampleAttrs

@dataclass
class DatasetConfig:
    dataset_name: str
    experiment_name: str
    num_labels: int
    category_list: list[str]
    sampling_rate: float = 256.0
```
