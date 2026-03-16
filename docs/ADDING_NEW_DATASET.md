# Adding a New Dataset

This guide walks you through adding a new EEG dataset processing file to the benchmark dataloader. We use SEED-IV as the primary example.

## 1. Pre-Implementation Questions

Before writing any code, answer these questions about your dataset:

### Dataset Overview
- [ ] **What is this dataset for?** (emotion recognition, motor imagery, sleep staging, seizure detection, etc.)
- [ ] **What is the downstream task?** (classification, regression, detection)
- [ ] **How many subjects?**
- [ ] **How many sessions per subject?**
- [ ] **How many trials per session?**
- [ ] **What are the class labels?**
- [ ] **Any bechmark (EEG-FM-Bench/adabrain-bench) include this dataset?**

### Signal Properties
- [ ] **What is the original sampling rate?**
- [ ] **What is the trial/segment length?**
- [ ] **What stimuli were used?** (video clips, audio, visual cues, motor cues, etc.)
- [ ] **If it is include in other benchmark, is there any exclusion, processing process for referece?**

### EEG Setup
- [ ] **What EEG montage?** (10-20, 10-10, custom)
- [ ] **What reference channels?** (which channels to remove)
- [ ] **Where was data collected?** (determines notch filter: 50Hz for Asia/Europe, 60Hz for Americas)

### Example: SEED-IV Answers
| Question | Answer |
|----------|--------|
| Purpose | Emotion recognition |
| Task | 4-class classification |
| Subjects | 15 |
| Sessions | 3 per subject |
| Trials | 24 per session |
| Labels | neutral, sad, fear, happy |
| Sampling rate | 200 Hz |
| Stimuli | Video clips |
| Montage | 10-10 (62 channels) |
| Reference | CB1, CB2 (removed) |
| Location | China → 50Hz notch |

---

## 2. Dataset Configuration

Define your dataset metadata using `DatasetInfo`:

```python
from ..config import DatasetInfo, DatasetTaskType, DownstreamTaskType

SEEDIV_INFO = DatasetInfo(
    dataset_name="SEEDIV_4class",
    task_type=DatasetTaskType.EMOTION,
    downstream_task_type=DownstreamTaskType.CLASSIFICATION,
    num_labels=4,
    category_list=["neutral", "sad", "fear", "happy"],
    sampling_rate=200.0,
    montage="10_10",
    channels=[
        'FP1', 'FPZ', 'FP2',
        'AF3', 'AF4',
        'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
        # ... (62 channels total)
    ],
)
```

### Available Task Types
From `config.py`:
- `DatasetTaskType.EMOTION` - Emotion recognition
- `DatasetTaskType.MOTOR_IMAGINARY` - Motor imagery
- `DatasetTaskType.SLEEP` - Sleep staging
- `DatasetTaskType.SEIZURE` - Seizure detection
- `DatasetTaskType.COGNITIVE` - Cognitive tasks
- `DatasetTaskType.RESTING_STATE` - Resting state
- `DatasetTaskType.OTHER` - Other tasks to be add/discuss

---

## 3. Builder Class Structure

Create a Builder class with these methods:

### 3.1 Constructor (`__init__`)

```python
class SEEDIVBuilder:
    def __init__(
        self,
        raw_data_dir: str,
        output_dir: str = "./hdf5",
        target_sfreq: float = 200.0,
        window_sec: float = 4.0,
        stride_sec: float = 4.0,
        filter_low: float = 0.1,
        filter_high: float = 75.0,
        filter_notch: float = 50.0,
    ):
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir) / "SEEDIV"
        self.target_sfreq = target_sfreq
        self.orig_sfreq = 200.0  # Original sampling rate
        self.window_sec = window_sec
        self.stride_sec = stride_sec
        self.filter_low = filter_low
        self.filter_high = filter_high
        self.filter_notch = filter_notch

        self.window_samples = int(window_sec * target_sfreq)
        self.stride_samples = int(stride_sec * target_sfreq)
```

### 3.2 Subject IDs (`get_subject_ids`)

```python
def get_subject_ids(self) -> list[int]:
    """Get list of subject IDs (1-15)."""
    return list(range(1, 16))
```

### 3.3 File Discovery (`_find_files`)

Locate raw data files for each subject:

```python
def _find_files(self, subject_id: int) -> dict[int, Path]:
    """Find all session files for a subject."""
    files = {}
    for session in range(1, 4):
        session_dir = self.raw_data_dir / str(session)
        if session_dir.exists():
            exact = session_dir / f"{subject_id}.mat"
            if exact.exists():
                files[session] = exact
            else:
                for f in session_dir.glob(f"{subject_id}_*.mat"):
                    files[session] = f
                    break
    return files
```

### 3.4 Data Loading (`_convert_to_mne` or `_read_raw`)

Convert raw data to MNE format. **MNE uses Volts internally**, so convert accordingly.

**Auto-detection helper function:**
```python
def detect_unit_and_convert_to_volts(data: np.ndarray) -> tuple[np.ndarray, str]:
    """Auto-detect data unit and convert to Volts for MNE.

    Returns:
        tuple: (data_in_volts, detected_unit)
    """
    max_amp = np.abs(data).max()

    if max_amp > 1e-2:  # > 0.01, likely microvolts
        return data / 1e6, "µV"
    elif max_amp > 1e-5:  # > 0.00001, likely millivolts
        return data / 1e3, "mV"
    else:  # likely already Volts
        return data, "V"
```

**Usage:**
```python
def _convert_to_mne(self, data: np.ndarray):
    """Convert numpy array to MNE Raw object."""
    # Auto-detect unit and convert to Volts
    data_volts, detected_unit = detect_unit_and_convert_to_volts(data)
    print(f"  Detected unit: {detected_unit}, max amplitude: {np.abs(data).max():.2e}")

    info = mne.create_info(
        ch_names=SEEDIV_ORIG_CHANNELS,
        sfreq=self.orig_sfreq,
        ch_types=['eeg'] * len(SEEDIV_ORIG_CHANNELS)
    )
    raw = mne.io.RawArray(data_volts, info, verbose=False)
    return raw
```

**Unit Convention:**
| Original Unit | Max Amplitude Range | Conversion for MNE | Output for HDF5 (µV) |
|---------------|--------------------|--------------------|---------------------|
| Volts (V) | < 1e-5 | No conversion | `* 1e6` |
| Millivolts (mV) | 1e-5 to 1e-2 | `/ 1e3` | `* 1e3` |
| Microvolts (µV) | > 1e-2 | `/ 1e6` | No conversion |

For standard file formats, use MNE's built-in readers:
```python
# EEGLAB .set files
raw = mne.io.read_raw_eeglab(str(file_path), preload=True, verbose=False)

# GDF files
raw = mne.io.read_raw_gdf(str(file_path), preload=True, verbose=False)

# EDF files
raw = mne.io.read_raw_edf(str(file_path), preload=True, verbose=False)
```

### 3.5 Preprocessing (`_preprocess`)

```python
def _preprocess(self, raw):
    """Apply preprocessing to raw data."""
    # Drop reference channels
    raw.drop_channels(SEEDIV_REMOVE_CHANNELS)

    # Notch filter
    if self.filter_notch > 0:
        raw.notch_filter(freqs=self.filter_notch, verbose=False)

    # Bandpass filter
    raw.filter(l_freq=self.filter_low, h_freq=self.filter_high, verbose=False)

    # Resample if needed
    if raw.info['sfreq'] != self.target_sfreq:
        raw.resample(self.target_sfreq, verbose=False)

    return raw
```

---

## 4. Processing Pipeline Details

### 4.1 Notch Filter Frequency

Choose based on power line frequency where data was collected:

| Region | Notch Frequency |
|--------|-----------------|
| Asia (China, Japan, Korea) | 50 Hz |
| Europe | 50 Hz |
| Americas (USA, Canada, Brazil) | 60 Hz |
| Australia | 50 Hz |

### 4.2 Bandpass Filter

Typical ranges by task type:

| Task Type | Low (Hz) | High (Hz) | Notes |
|-----------|----------|-----------|-------|
| Emotion | 0.1 | 75.0 | Wide band for emotion features |
| Motor Imagery | 0.5 | 45.0 | Focus on mu/beta rhythms |
| Sleep | 0.3 | 35.0 | Focus on delta/theta/alpha |
| Seizure | 0.5 | 70.0 | Wide band for spike detection |

### 4.3 Windowing

- **window_sec**: Length of each segment in seconds
- **stride_sec**: Step size between segments (same as window_sec for non-overlapping)

```python
# Non-overlapping windows
window_sec = 4.0
stride_sec = 4.0

# 50% overlapping windows
window_sec = 4.0
stride_sec = 2.0
```

### 4.4 Amplitude Validation (Trial Level)

**All output data should be in microvolts (µV).** Validate amplitude at the trial level before segmentation:

```python
# Default amplitude threshold (can be customized per dataset)
DEFAULT_MAX_AMPLITUDE_UV = 600.0

class YourDatasetBuilder:
    def __init__(
        self,
        ...
        max_amplitude_uv: float = DEFAULT_MAX_AMPLITUDE_UV,  # Configurable per dataset
    ):
        self.max_amplitude_uv = max_amplitude_uv
        # Track validation statistics
        self.total_trials = 0
        self.valid_trials = 0
        self.rejected_trials = 0

    def _validate_trial(self, trial_data: np.ndarray) -> bool:
        """Check if trial amplitude is within acceptable range."""
        return np.abs(trial_data).max() <= self.max_amplitude_uv

    def _report_validation_stats(self, subject_id: int):
        """Report trial validation statistics."""
        valid_pct = (self.valid_trials / self.total_trials * 100) if self.total_trials > 0 else 0
        print(f"Subject {subject_id} validation report:")
        print(f"  Amplitude threshold: {self.max_amplitude_uv} µV")
        print(f"  Total trials: {self.total_trials}")
        print(f"  Valid trials: {self.valid_trials} ({valid_pct:.1f}%)")
        print(f"  Rejected trials: {self.rejected_trials} ({100-valid_pct:.1f}%)")
```

**Usage in build_subject:**
```python
def build_subject(self, subject_id: int) -> str:
    # Reset counters for this subject
    self.total_trials = 0
    self.valid_trials = 0
    self.rejected_trials = 0

    for trial in all_trials:
        trial_data = trial['data']  # Should be in µV
        self.total_trials += 1

        # Skip trials with excessive amplitude (artifacts)
        if not self._validate_trial(trial_data):
            self.rejected_trials += 1
            print(f"  Skipping trial {trial['trial_id']}: amplitude {np.abs(trial_data).max():.1f} µV > {self.max_amplitude_uv} µV")
            continue

        self.valid_trials += 1
        # Process valid trial...

    # Report statistics after processing
    self._report_validation_stats(subject_id)
```

**Recommended thresholds by task:**
| Task Type | Threshold (µV) |
|-----------|---------------|
| Default | 600 |
| Emotion | 500 |
| Motor Imagery | 600 |
| Sleep | 800 |

---

## 5. Data Structure & Labels

### 5.1 Label Metadata

Define labels as arrays or mappings:

```python
# Array format (SEED-IV style)
SEEDIV_LABEL_META = np.array([
    [1, 2, 3, 0, 2, 0, 0, 1, ...],  # Session 1 labels
    [2, 1, 3, 0, 0, 2, 0, 2, ...],  # Session 2 labels
    [1, 2, 2, 1, 3, 3, 3, 1, ...],  # Session 3 labels
])

# Dictionary format (BCIC-2A style)
BCIC2A_LABEL_MAP = {
    '769': 0,  # left hand
    '770': 1,  # right hand
    '771': 2,  # feet
    '772': 3,  # tongue
}
```

### 5.2 Channel Configuration

```python
# Channels to remove (reference electrodes)
SEEDIV_REMOVE_CHANNELS = ['CB1', 'CB2']

# Original channel names (including reference)
SEEDIV_ORIG_CHANNELS = [
    'FP1', 'FPZ', 'FP2',
    # ... EEG channels ...
    'CB1', 'O1', 'OZ', 'O2', 'CB2'  # CB1, CB2 are reference
]
```

---

## 6. Output Format

### 6.1 Schema Classes

From `schema.py`:

```python
@dataclass
class SubjectAttrs:
    subject_id: int
    dataset_name: str
    task_type: str
    downstream_task_type: str
    rsFreq: float
    chn_name: list[str]
    num_labels: int = 0  # number of classes
    category_list: list[str] = []  # label names, index = label number (e.g., ["neutral", "sad", "fear", "happy"])
    chn_pos: Optional[np.ndarray] = None
    chn_ori: Optional[np.ndarray] = None
    chn_type: str = "EEG"
    montage: str = "10_20"

@dataclass
class TrialAttrs:
    trial_id: int
    session_id: int

@dataclass
class SegmentAttrs:
    segment_id: int
    start_time: float
    end_time: float
    time_length: float
    label: np.ndarray
```

### 6.2 Writing HDF5 Files

```python
from ..hdf5_io import HDF5Writer

# Create subject attributes (include label info from DatasetInfo)
subject_attrs = SubjectAttrs(
    subject_id=subject_id,
    dataset_name="SEEDIV_4class",
    task_type="emotion",
    downstream_task_type="classification",
    rsFreq=self.target_sfreq,
    chn_name=ch_names,
    num_labels=4,  # number of classes
    category_list=["neutral", "sad", "fear", "happy"],  # index = label number
    chn_pos=None,
    chn_ori=None,
    chn_type="EEG",
    montage="10_10",
)

# Write to HDF5
with HDF5Writer(str(output_path), subject_attrs) as writer:
    for trial in all_trials:
        trial_attrs = TrialAttrs(
            trial_id=trial['trial_id'],
            session_id=trial['session_id'],
        )
        trial_name = writer.add_trial(trial_attrs)

        # Segment into windows
        for i_slice, start in enumerate(range(0, n_samples - window_samples + 1, stride_samples)):
            end = start + window_samples
            slice_data = trial_data[:, start:end]

            segment_attrs = SegmentAttrs(
                segment_id=i_slice,
                start_time=start_time,
                end_time=end_time,
                time_length=self.window_sec,
                label=np.array([trial['label']]),
            )
            writer.add_segment(trial_name, segment_attrs, slice_data)
```

---

## 7. Template Skeleton

Copy and modify this template for your new dataset:

We also provide a ready-to-edit template at `datasets/template_dataset.py`. Search for `## TODO` markers to find all required changes.

```python
"""
YOUR_DATASET Dataset Builder.

YOUR_DATASET: Brief description.
- N subjects
- M sessions per subject
- K trials per session
- C classes: class1, class2, ...
"""

import os
from pathlib import Path
import numpy as np

try:
    import mne
    from scipy.io import loadmat
    HAS_MNE = True
except ImportError:
    HAS_MNE = False

try:
    from ..schema import SubjectAttrs, TrialAttrs, SegmentAttrs
    from ..hdf5_io import HDF5Writer
    from ..config import DatasetInfo, DatasetTaskType, DownstreamTaskType
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from schema import SubjectAttrs, TrialAttrs, SegmentAttrs
    from hdf5_io import HDF5Writer
    from config import DatasetInfo, DatasetTaskType, DownstreamTaskType


# Dataset Configuration
YOUR_DATASET_INFO = DatasetInfo(
    dataset_name="YOUR_DATASET_Nclass",
    task_type=DatasetTaskType.EMOTION,  # Change as needed
    downstream_task_type=DownstreamTaskType.CLASSIFICATION,
    num_labels=4,  # Number of classes
    category_list=["class1", "class2", "class3", "class4"],
    sampling_rate=200.0,
    montage="10_20",
    channels=[
        # List your channels here
    ],
)

# Label metadata
YOUR_DATASET_LABELS = {
    # Define your label mapping
}

# Channels to remove (if any)
YOUR_DATASET_REMOVE_CHANNELS = []

# Default amplitude threshold (µV)
DEFAULT_MAX_AMPLITUDE_UV = 600.0


def detect_unit_and_convert_to_volts(data: np.ndarray) -> tuple[np.ndarray, str]:
    """Auto-detect data unit and convert to Volts for MNE."""
    max_amp = np.abs(data).max()
    if max_amp > 1e-2:  # likely µV
        return data / 1e6, "µV"
    return data, "V"


class YourDatasetBuilder:
    """Builder for YOUR_DATASET dataset."""

    def __init__(
        self,
        raw_data_dir: str,
        output_dir: str = "./hdf5",
        target_sfreq: float = 200.0,
        window_sec: float = 4.0,
        stride_sec: float = 4.0,
        filter_low: float = 0.1,
        filter_high: float = 75.0,
        filter_notch: float = 50.0,  # 50Hz or 60Hz based on region
        max_amplitude_uv: float = DEFAULT_MAX_AMPLITUDE_UV,
    ):
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir) / "YOUR_DATASET"
        self.target_sfreq = target_sfreq
        self.orig_sfreq = 200.0  # Original sampling rate
        self.window_sec = window_sec
        self.stride_sec = stride_sec
        self.filter_low = filter_low
        self.filter_high = filter_high
        self.filter_notch = filter_notch

        self.window_samples = int(window_sec * target_sfreq)
        self.stride_samples = int(stride_sec * target_sfreq)
        self.max_amplitude_uv = max_amplitude_uv

        # Validation statistics
        self.total_trials = 0
        self.valid_trials = 0
        self.rejected_trials = 0

    def get_subject_ids(self) -> list[int]:
        """Get list of subject IDs."""
        return list(range(1, N + 1))  # Adjust N

    def _find_files(self, subject_id: int) -> list[Path]:
        """Find all files for a subject."""
        # Implement file discovery logic
        pass

    def _read_raw(self, file_path: Path):
        """Read raw EEG file and convert to MNE Raw object."""
        # Example for .mat files:
        # mat = loadmat(str(file_path))
        # data = mat['data']  # shape: (n_channels, n_samples)
        #
        # Auto-detect unit and convert to Volts
        # data_volts, detected_unit = detect_unit_and_convert_to_volts(data)
        # print(f"  Detected unit: {detected_unit}")
        #
        # info = mne.create_info(ch_names=..., sfreq=self.orig_sfreq, ch_types='eeg')
        # raw = mne.io.RawArray(data_volts, info, verbose=False)
        # return raw
        pass

    def _validate_trial(self, trial_data: np.ndarray) -> bool:
        """Check if trial amplitude is within acceptable range."""
        return np.abs(trial_data).max() <= self.max_amplitude_uv

    def _report_validation_stats(self, subject_id: int):
        """Report trial validation statistics."""
        valid_pct = (self.valid_trials / self.total_trials * 100) if self.total_trials > 0 else 0
        print(f"Subject {subject_id} validation report:")
        print(f"  Amplitude threshold: {self.max_amplitude_uv} µV")
        print(f"  Total trials: {self.total_trials}")
        print(f"  Valid trials: {self.valid_trials} ({valid_pct:.1f}%)")
        print(f"  Rejected trials: {self.rejected_trials} ({100-valid_pct:.1f}%)")

    def _preprocess(self, raw):
        """Apply preprocessing to raw data."""
        # Drop reference channels if needed
        if YOUR_DATASET_REMOVE_CHANNELS:
            raw.drop_channels(YOUR_DATASET_REMOVE_CHANNELS)

        # Notch filter
        if self.filter_notch > 0:
            raw.notch_filter(freqs=self.filter_notch, verbose=False)

        # Bandpass filter
        raw.filter(l_freq=self.filter_low, h_freq=self.filter_high, verbose=False)

        # Resample if needed
        if raw.info['sfreq'] != self.target_sfreq:
            raw.resample(self.target_sfreq, verbose=False)

        return raw

    def build_subject(self, subject_id: int) -> str:
        """Build HDF5 file for a single subject."""
        if not HAS_MNE:
            raise ImportError("MNE is required")

        # Reset validation counters
        self.total_trials = 0
        self.valid_trials = 0
        self.rejected_trials = 0

        # 1. Find files
        files = self._find_files(subject_id)
        if not files:
            raise FileNotFoundError(f"No files found for subject {subject_id}")

        all_trials = []
        ch_names = None

        # 2. Process each file/session
        for session_id, file_path in enumerate(files, 1):
            print(f"Reading {file_path}")

            raw = self._read_raw(file_path)
            raw = self._preprocess(raw)

            if ch_names is None:
                ch_names = raw.ch_names

            # 3. Extract trials and labels
            # Implement your trial extraction logic
            # Data should be in µV: raw.get_data() * 1e6
            pass

        # 4. Create subject attributes (use DatasetInfo values)
        subject_attrs = SubjectAttrs(
            subject_id=subject_id,
            dataset_name=YOUR_DATASET_INFO.dataset_name,
            task_type=YOUR_DATASET_INFO.task_type.value,
            downstream_task_type=YOUR_DATASET_INFO.downstream_task_type.value,
            rsFreq=self.target_sfreq,
            chn_name=ch_names,
            num_labels=YOUR_DATASET_INFO.num_labels,
            category_list=YOUR_DATASET_INFO.category_list,
            chn_pos=None,
            chn_ori=None,
            chn_type="EEG",
            montage=YOUR_DATASET_INFO.montage,
        )

        # 5. Write HDF5
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / f"sub_{subject_id}.h5"

        with HDF5Writer(str(output_path), subject_attrs) as writer:
            for trial in all_trials:
                trial_data = trial['data']  # Should be in µV
                self.total_trials += 1

                # Validate trial amplitude
                if not self._validate_trial(trial_data):
                    self.rejected_trials += 1
                    print(f"  Skipping trial {trial['trial_id']}: amplitude {np.abs(trial_data).max():.1f} µV > {self.max_amplitude_uv} µV")
                    continue

                self.valid_trials += 1
                trial_attrs = TrialAttrs(
                    trial_id=trial['trial_id'],
                    session_id=trial['session_id'],
                )
                trial_name = writer.add_trial(trial_attrs)

                # Segment into windows
                n_samples = trial_data.shape[1]

                for i_slice, start in enumerate(range(0, n_samples - self.window_samples + 1, self.stride_samples)):
                    end = start + self.window_samples
                    slice_data = trial_data[:, start:end]

                    segment_attrs = SegmentAttrs(
                        segment_id=i_slice,
                        start_time=start / self.target_sfreq,
                        end_time=end / self.target_sfreq,
                        time_length=self.window_sec,
                        label=np.array([trial['label']]),
                    )
                    writer.add_segment(trial_name, segment_attrs, slice_data)

        # Report validation statistics
        self._report_validation_stats(subject_id)
        print(f"Saved {output_path}")
        return str(output_path)

    def build_all(self, subject_ids: list[int] = None) -> list[str]:
        """Build HDF5 files for all subjects."""
        if subject_ids is None:
            subject_ids = self.get_subject_ids()

        output_paths = []
        for subject_id in subject_ids:
            try:
                output_path = self.build_subject(subject_id)
                output_paths.append(output_path)
            except Exception as e:
                print(f"Error processing subject {subject_id}: {e}")

        return output_paths


def build_your_dataset(
    raw_data_dir: str,
    output_dir: str = "./hdf5",
    subject_ids: list[int] = None,
    **kwargs,
) -> list[str]:
    """Convenience function to build YOUR_DATASET."""
    builder = YourDatasetBuilder(raw_data_dir, output_dir, **kwargs)
    return builder.build_all(subject_ids)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build YOUR_DATASET HDF5 dataset")
    parser.add_argument("raw_data_dir", help="Directory containing raw files")
    parser.add_argument("--output_dir", default="./hdf5", help="Output directory")
    parser.add_argument("--subjects", nargs="+", type=int, help="Subject IDs to process")
    args = parser.parse_args()

    build_your_dataset(args.raw_data_dir, args.output_dir, args.subjects)
```

---

## 8. Running & Testing
check the working dir for not found issues
### Run for a single subject (testing):
```bash
python -m benchmark_dataloader.datasets.your_dataset /path/to/raw/data --output_dir /path/to/output --subjects 1
```

### Run for all subjects:
```bash
python -m benchmark_dataloader.datasets.your_dataset /path/to/raw/data --output_dir /path/to/output
```

### Verify output:
```python
import h5py

with h5py.File("output/YOUR_DATASET/sub_1.h5", "r") as f:
    print("Keys:", list(f.keys()))
    print("Attrs:", dict(f.attrs))
```

---

## Checklist Before Submitting

- [ ] Dataset info defined with correct task type and labels
- [ ] Notch filter frequency matches data collection region
- [ ] Bandpass filter appropriate for task type
- [ ] Reference channels removed
- [ ] Channel names match standard montage
- [ ] Labels correctly mapped
- [ ] Data output in microvolts (µV)
- [ ] Amplitude threshold set appropriately (default: 600 µV)
- [ ] Validation report shows acceptable rejection rate
- [ ] Tested on at least one subject
- [ ] Added entry to `datasets/__init__.py`
