"""
PhysioNet MI Dataset Builder.

PhysioNet EEG Motor Movement/Imagery Database (EEGMMIDB)
- 109 subjects
- 14 runs per subject (R01-R14)
- 4 motor imagery classes: left hand, right hand, both feet, tongue
- 64 EEG channels (10-10 system)
- 160 Hz sampling rate
- https://physionet.org/content/eegmmidb/1.0.0/

Task Types:
- R01, R02: Baseline (eyes open/closed) - not used for MI classification
- R03, R04, R07, R08, R11, R12: Motor execution tasks - not used
- R05, R06, R09, R10, R13, R14: Motor imagery tasks (4 classes)
  - T0: Rest
  - T1: Left hand
  - T2: Right hand
  - T3: Both feet
  - T4: Tongue

"""

import os
import re
import json
import warnings
from pathlib import Path
from datetime import datetime
import numpy as np

try:
    import mne
    HAS_MNE = True
except ImportError:
    HAS_MNE = False

try:
    from ..schema import SubjectAttrs, TrialAttrs, SegmentAttrs
    from ..hdf5_io import HDF5Writer
    from ..config import DatasetInfo, DatasetTaskType, DownstreamTaskType
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from schema import SubjectAttrs, TrialAttrs, SegmentAttrs
    from hdf5_io import HDF5Writer
    from config import DatasetInfo, DatasetTaskType, DownstreamTaskType

# PhysioNet MI uses 64 channels (10-10 system)
# Standard 64-channel names (excluding EOG channels if present)
PHYSIONET_MI_CHANNELS = [
    'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6',
    'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6',
    'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6',
    'Fp1', 'Fpz', 'Fp2', 'AF7', 'AF3', 'AFZ', 'AF4', 'AF8',
    'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
    'FT7', 'FT8',
    'T7', 'T8', 'T9', 'T10',
    'TP7', 'TP8', 'TP9', 'TP10',
    'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
    'PO7', 'PO3', 'POZ', 'PO4', 'PO8',
    'O1', 'OZ', 'O2', 'Iz',
]

# Channels to remove (EOG, if present)
PHYSIONET_MI_REMOVE_CHANNELS = ['EOG-left', 'EOG-right', 'EOG-up', 'EOG-down']

# Task runs:
# R03, R07, R11: Task 1 (Real movement: left or right fist)
# R04, R08, R12: Task 2 (Imagined movement: left or right fist)
# R05, R09, R13: Task 3 (Real movement: both fists or both feet)
# R06, R10, R14: Task 4 (Imagined movement: both fists or both feet)

# Real movement runs (R03, R05, R07, R09, R11, R13)
REAL_MOVEMENT_RUNS = ['R03', 'R05', 'R07', 'R09', 'R11', 'R13']

# Imagined movement runs (R04, R06, R08, R10, R12, R14)
IMAGINED_MOVEMENT_RUNS = ['R04', 'R06', 'R08', 'R10', 'R12', 'R14']

# Left/Right fist runs (R03, R04, R07, R08, R11, R12)
FIST_RUNS = ['R03', 'R04', 'R07', 'R08', 'R11', 'R12']

# Both fists/feet runs (R05, R06, R09, R10, R13, R14)
FISTS_FEET_RUNS = ['R05', 'R06', 'R09', 'R10', 'R13', 'R14']

# All task runs (excluding baseline R01, R02)
ALL_TASK_RUNS = REAL_MOVEMENT_RUNS + IMAGINED_MOVEMENT_RUNS

# Label mapping system:
# Two-dimensional labeling:
# 1. Body part (body_part_label):
#    0: Left fist
#    1: Right fist
#    2: Both fists
#    3: Both feet
# 2. Movement type (movement_type_label):
#    0: Real movement
#    1: Imagined movement
# Combined 8-class label = body_part_label * 2 + movement_type_label
#    0: Left fist (real)      = 0*2 + 0 = 0
#    1: Left fist (imagined)  = 0*2 + 1 = 1
#    2: Right fist (real)     = 1*2 + 0 = 2
#    3: Right fist (imagined) = 1*2 + 1 = 3
#    4: Both fists (real)     = 2*2 + 0 = 4
#    5: Both fists (imagined) = 2*2 + 1 = 5
#    6: Both feet (real)      = 3*2 + 0 = 6
#    7: Both feet (imagined)  = 3*2 + 1 = 7
# T0: Rest - always skip

PHYSIONET_MI_INFO = DatasetInfo(
    dataset_name="Physionet_MI",
    task_type=DatasetTaskType.MOTOR_IMAGINARY,
    downstream_task_type=DownstreamTaskType.CLASSIFICATION,
    num_labels=8,  # 8-class: 4 body parts × 2 movement types
    category_list=[
        "left_real", "left_imagined",
        "right_real", "right_imagined",
        "fists_real", "fists_imagined",
        "feet_real", "feet_imagined"
    ],
    sampling_rate=160.0,
    montage="10_10",
    channels=PHYSIONET_MI_CHANNELS,
)

# Default amplitude threshold (µV)
DEFAULT_MAX_AMPLITUDE_UV = 800.0


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


class PhysionetMIBuilder:
    """Builder for PhysioNet MI dataset."""

    def __init__(
        self,
        raw_data_dir: str,
        output_dir: str = "./hdf5",
        target_sfreq: float = 200.0,
        window_sec: float = 1.0,
        stride_sec: float = 1.0,
        filter_low: float = 0.1,
        filter_high: float = 75.0,
        filter_notch: float = 60.0,  # 60 Hz for US data
        max_amplitude_uv: float = DEFAULT_MAX_AMPLITUDE_UV,
        imagined_only: bool = False,
    ):
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir) / "Physionet_MI"
        self.target_sfreq = target_sfreq
        self.orig_sfreq = 160.0
        self.window_sec = window_sec
        self.stride_sec = stride_sec
        self.filter_low = filter_low
        self.filter_high = filter_high
        self.filter_notch = filter_notch
        self.max_amplitude_uv = max_amplitude_uv
        self.imagined_only = imagined_only

        self.window_samples = int(window_sec * target_sfreq)
        self.stride_samples = int(stride_sec * target_sfreq)

        # Validation statistics
        self.total_trials = 0
        self.valid_trials = 0
        self.rejected_trials = 0

    def _validate_trial(self, trial_data: np.ndarray) -> bool:
        """Check if trial amplitude is within acceptable range."""
        return np.abs(trial_data).max() <= self.max_amplitude_uv

    def _report_validation_stats(self, subject_id: int):
        """Report validation statistics."""
        valid_pct = (self.valid_trials / self.total_trials * 100) if self.total_trials > 0 else 0
        print(f"Subject {subject_id} validation report:")
        print(f"  Amplitude threshold: {self.max_amplitude_uv} µV")
        print(f"  Total trials: {self.total_trials}")
        print(f"  Valid trials: {self.valid_trials} ({valid_pct:.1f}%)")
        print(f"  Rejected trials: {self.rejected_trials} ({100-valid_pct:.1f}%)")

    def _save_dataset_info(self, stats: dict):
        """Save dataset info and processing parameters to JSON."""
        info = {
            "dataset": {
                "name": PHYSIONET_MI_INFO.dataset_name,
                "description": "PhysioNet EEG Motor Movement/Imagery Database",
                "task_type": str(PHYSIONET_MI_INFO.task_type.value),
                "downstream_task": str(PHYSIONET_MI_INFO.downstream_task_type.value),
                "original_sampling_rate": PHYSIONET_MI_INFO.sampling_rate,
                "channels": PHYSIONET_MI_INFO.channels,
                "montage": PHYSIONET_MI_INFO.montage,
                "source_url": "https://physionet.org/content/eegmmidb/1.0.0/",
            },
            "labeling_system": {
                "description": "Two-dimensional labeling system with tuple format in label field",
                "label_format": "tuple",
                "label_structure": "label = [body_part_label, movement_type_label]",
                "default_reading": "When reading segments, label field contains tuple: (body_part, movement_type)",
                "label_dimensions": {
                    "body_part": {
                        "description": "Body part involved in movement/imagery",
                        "index": 0,
                        "num_labels": 4,
                        "labels": {
                            "0": "Left fist",
                            "1": "Right fist",
                            "2": "Both fists",
                            "3": "Both feet"
                        },
                        "mapping": {
                            "T1 in FIST_RUNS (R03, R04, R07, R08, R11, R12)": "Left fist (0)",
                            "T2 in FIST_RUNS (R03, R04, R07, R08, R11, R12)": "Right fist (1)",
                            "T1 in FISTS_FEET_RUNS (R05, R06, R09, R10, R13, R14)": "Both fists (2)",
                            "T2 in FISTS_FEET_RUNS (R05, R06, R09, R10, R13, R14)": "Both feet (3)"
                        }
                    },
                    "movement_type": {
                        "description": "Type of movement (real execution vs. imagined)",
                        "index": 1,
                        "num_labels": 2,
                        "labels": {
                            "0": "Real movement",
                            "1": "Imagined movement"
                        },
                        "mapping": {
                            "REAL_MOVEMENT_RUNS (R03, R05, R07, R09, R11, R13)": "Real movement (0)",
                            "IMAGINED_MOVEMENT_RUNS (R04, R06, R08, R10, R12, R14)": "Imagined movement (1)"
                        }
                    }
                },
                "combined_label_formula": "combined_label = body_part_label * 2 + movement_type_label",
                "combined_label_mapping": {
                    "0": "Left fist (real) - body_part=0, movement_type=0",
                    "1": "Left fist (imagined) - body_part=0, movement_type=1",
                    "2": "Right fist (real) - body_part=1, movement_type=0",
                    "3": "Right fist (imagined) - body_part=1, movement_type=1",
                    "4": "Both fists (real) - body_part=2, movement_type=0",
                    "5": "Both fists (imagined) - body_part=2, movement_type=1",
                    "6": "Both feet (real) - body_part=3, movement_type=0",
                    "7": "Both feet (imagined) - body_part=3, movement_type=1"
                },
                "usage_examples": {
                    "default_reading": "label = segment.label  # Returns [body_part, movement_type]",
                    "extract_body_part": "body_part = segment.label[0]  # 0-3",
                    "extract_movement_type": "movement_type = segment.label[1]  # 0-1",
                    "4_class_body_part": "Use label[0] for 4-class body part classification",
                    "2_class_movement_type": "Use label[1] for 2-class movement type classification",
                    "8_class_combined": "combined = label[0] * 2 + label[1]  # 0-7",
                    "8_class_from_metadata": "combined = trial.clinical_metadata['combined_label']  # Also available"
                },
                "note": "T0 events (rest) are excluded from all classifications. Tuple format allows flexible usage without accessing metadata."
            },
            "processing": {
                "target_sampling_rate": self.target_sfreq,
                "window_sec": self.window_sec,
                "stride_sec": self.stride_sec,
                "filter_low": self.filter_low,
                "filter_high": self.filter_high,
                "filter_notch": self.filter_notch,
                "removed_channels": PHYSIONET_MI_REMOVE_CHANNELS,
                "max_amplitude_uv": self.max_amplitude_uv,
                "real_movement_runs": REAL_MOVEMENT_RUNS,
                "imagined_movement_runs": IMAGINED_MOVEMENT_RUNS,
                "fist_runs": FIST_RUNS,
                "fists_feet_runs": FISTS_FEET_RUNS,
            },
            "statistics": stats,
            "generated_at": datetime.now().isoformat(),
        }

        json_path = self.output_dir / "dataset_info.json"
        with open(json_path, "w") as f:
            json.dump(info, f, indent=2)
        print(f"Saved dataset info to {json_path}")

    def get_subject_ids(self) -> list[int]:
        """Get list of subject IDs from directory structure."""
        # Look for S001, S002, ... directories
        base_path = self.raw_data_dir
        if (base_path / "files" / "eegmmidb" / "1.0.0").exists():
            base_path = base_path / "files" / "eegmmidb" / "1.0.0"
        elif (base_path / "eegmmidb" / "1.0.0").exists():
            base_path = base_path / "eegmmidb" / "1.0.0"

        subject_ids = []
        for item in base_path.iterdir():
            if item.is_dir() and item.name.startswith('S'):
                try:
                    sub_id = int(item.name[1:])
                    subject_ids.append(sub_id)
                except ValueError:
                    continue

        return sorted(subject_ids)

    def _find_files(self, subject_id: int) -> list[Path]:
        """Find all task files for a subject (excluding baseline R01, R02).
        
        If imagined_only=True, only returns imagined movement runs.
        """
        base_path = self.raw_data_dir
        if (base_path / "files" / "eegmmidb" / "1.0.0").exists():
            base_path = base_path / "files" / "eegmmidb" / "1.0.0"
        elif (base_path / "eegmmidb" / "1.0.0").exists():
            base_path = base_path / "eegmmidb" / "1.0.0"

        subject_dir = base_path / f"S{subject_id:03d}"
        if not subject_dir.exists():
            return []

        files = []
        # Use IMAGINED_MOVEMENT_RUNS if imagined_only, otherwise ALL_TASK_RUNS
        runs_to_process = IMAGINED_MOVEMENT_RUNS if self.imagined_only else ALL_TASK_RUNS
        for run in runs_to_process:
            edf_file = subject_dir / f"S{subject_id:03d}{run}.edf"
            if edf_file.exists():
                files.append(edf_file)

        return sorted(files)

    def _read_edf(self, file_path: Path):
        """Read EDF file and convert to MNE Raw object."""
        raw = mne.io.read_raw_edf(str(file_path), preload=True)

        # Auto-detect unit and convert to Volts
        max_amp = np.abs(raw._data).max()
        if max_amp > 1e-3:  # > 0.001, likely microvolts
            raw._data = raw._data / 1e6
            detected_unit = "µV"
        else:
            detected_unit = "V"
        print(f"  Detected unit: {detected_unit}, max amplitude: {max_amp:.2e}")

        return raw

    def _preprocess(self, raw):
        """Apply preprocessing to raw data."""
        # Drop EOG channels if present
        channels_to_drop = [ch for ch in PHYSIONET_MI_REMOVE_CHANNELS if ch in raw.ch_names]
        if channels_to_drop:
            raw.drop_channels(channels_to_drop)
            print(f"  Dropped EOG channels: {channels_to_drop}")

        # Keep only EEG channels (remove any non-EEG)
        eeg_chs = [ch for ch in raw.ch_names if ch.startswith(('EEG', 'FC', 'C', 'CP', 'F', 'AF', 'FT', 'T', 'TP', 'P', 'PO', 'O', 'I', 'Fp'))]
        if len(eeg_chs) < len(raw.ch_names):
            non_eeg = [ch for ch in raw.ch_names if ch not in eeg_chs]
            if non_eeg:
                raw.drop_channels(non_eeg)
                print(f"  Dropped non-EEG channels: {non_eeg}")

        # If we have more channels than expected, try to pick standard ones
        if len(raw.ch_names) > len(PHYSIONET_MI_CHANNELS):
            # Try to pick standard channels by name
            available_std = [ch for ch in PHYSIONET_MI_CHANNELS if ch in raw.ch_names]
            if len(available_std) >= 60:  # Keep at least 60 standard channels
                raw.pick(available_std)
                print(f"  Picked {len(available_std)} standard EEG channels")
            else:
                # Keep first N channels (assuming they're EEG)
                n_keep = min(len(PHYSIONET_MI_CHANNELS), len(raw.ch_names))
                raw.pick(raw.ch_names[:n_keep])
                print(f"  Kept first {n_keep} channels")

        # Notch filter (60 Hz for US data)
        if self.filter_notch > 0:
            raw.notch_filter(freqs=self.filter_notch)

        # Bandpass filter
        raw.filter(l_freq=self.filter_low, h_freq=self.filter_high)

        # Resample if needed
        if raw.info['sfreq'] != self.target_sfreq:
            raw.resample(self.target_sfreq)

        return raw

    def _extract_trials(self, raw, run_name: str) -> list[dict]:
        """Extract trials from annotations with two-dimensional labeling.
        
        Args:
            raw: MNE Raw object
            run_name: Run identifier (e.g., 'R03', 'R05')
        
        Returns:
            List of trial dictionaries with:
            - body_part_label: 0=left, 1=right, 2=fists, 3=feet
            - movement_type_label: 0=real, 1=imagined
            - combined_label: 8-class label (body_part * 2 + movement_type)
        """
        trials = []
        anno = raw.annotations

        # Determine run characteristics
        is_fist_run = run_name in FIST_RUNS
        is_fists_feet_run = run_name in FISTS_FEET_RUNS
        is_real_movement = run_name in REAL_MOVEMENT_RUNS
        is_imagined_movement = run_name in IMAGINED_MOVEMENT_RUNS

        for onset, duration, desc in zip(anno.onset, anno.duration, anno.description):
            # Skip T0 (rest)
            if desc == 'T0':
                continue
            
            # Determine body part label based on run type and event code
            if desc == 'T1':
                if is_fist_run:
                    body_part_label = 0  # Left fist
                elif is_fists_feet_run:
                    body_part_label = 2  # Both fists
                else:
                    continue  # Unknown run type
            elif desc == 'T2':
                if is_fist_run:
                    body_part_label = 1  # Right fist
                elif is_fists_feet_run:
                    body_part_label = 3  # Both feet
                else:
                    continue  # Unknown run type
            else:
                continue  # Unknown event code
            
            # Determine movement type label
            if is_real_movement:
                movement_type_label = 0  # Real movement
            elif is_imagined_movement:
                movement_type_label = 1  # Imagined movement
            else:
                continue  # Unknown movement type
            
            # Combined 8-class label
            combined_label = body_part_label * 2 + movement_type_label
            
            # If imagined_only, use body_part_label as the main label (4-class)
            # Otherwise use combined_label (8-class)
            main_label = body_part_label if self.imagined_only else combined_label
            
            trials.append({
                'onset': onset,
                'duration': duration,
                'body_part_label': body_part_label,
                'movement_type_label': movement_type_label,
                'combined_label': combined_label,  # 8-class label for compatibility
                'label': main_label,  # Main label: 4-class if imagined_only, 8-class otherwise
            })

        return trials

    def build_subject(self, subject_id: int) -> str:
        """
        Build HDF5 file for a single subject.

        Args:
            subject_id: Subject identifier (1-109)

        Returns:
            Path to output HDF5 file
        """
        if not HAS_MNE:
            raise ImportError("MNE is required for building PhysioNet MI dataset")

        # Reset validation counters
        self.total_trials = 0
        self.valid_trials = 0
        self.rejected_trials = 0

        # Find files
        files = self._find_files(subject_id)
        if not files:
            raise FileNotFoundError(f"No MI task files found for subject {subject_id}")

        print(f"Processing subject {subject_id}, found {len(files)} task file(s)")

        all_trials = []
        ch_names = None
        trial_counter = 0

        # Process each file (run)
        for session_id, file_path in enumerate(files, 1):
            print(f"  Reading {file_path.name}")
            
            # Extract run name from filename (e.g., "S002R03.edf" -> "R03")
            run_match = re.search(r'(R\d+)', file_path.name)
            run_name = run_match.group(1) if run_match else None

            try:
                raw = self._read_edf(file_path)
                raw = self._preprocess(raw)

                if ch_names is None:
                    ch_names = raw.ch_names

                # Extract trials (pass run_name for correct label mapping)
                if run_name:
                    trials = self._extract_trials(raw, run_name)
                else:
                    print(f"  Warning: Could not extract run name from {file_path.name}, skipping")
                    continue
                    
                data = raw.get_data()  # shape: (n_channels, n_samples) in Volts

                for trial in trials:
                    onset_sample = int(trial['onset'] * self.target_sfreq)
                    # Use trial duration or window_sec, whichever is smaller
                    trial_duration = min(trial['duration'], self.window_sec)
                    end_sample = onset_sample + int(trial_duration * self.target_sfreq)

                    if end_sample <= data.shape[1]:
                        # Convert to µV
                        trial_data_v = data[:, onset_sample:end_sample]
                        trial_data_uv = trial_data_v * 1e6

                        # Validate trial amplitude
                        self.total_trials += 1
                        if not self._validate_trial(trial_data_uv):
                            max_amp = np.abs(trial_data_uv).max()
                            print(f"  Skipping trial {trial_counter}: amplitude {max_amp:.1f} µV > {self.max_amplitude_uv} µV")
                            self.rejected_trials += 1
                            continue

                        self.valid_trials += 1
                        all_trials.append({
                            'data': trial_data_uv,
                            'label': trial['label'],  # 4-class if imagined_only, 8-class otherwise
                            'body_part_label': trial['body_part_label'],
                            'movement_type_label': trial['movement_type_label'],
                            'session_id': session_id,
                            'trial_id': trial_counter,
                            'onset_time': trial['onset'],
                        })
                        trial_counter += 1

            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                import traceback
                traceback.print_exc()

        if not all_trials:
            raise ValueError(f"No valid trials extracted for subject {subject_id}")

        # Create dataset info based on imagined_only flag
        if self.imagined_only:
            dataset_info = DatasetInfo(
                dataset_name="Physionet_MI_imagined",
                task_type=PHYSIONET_MI_INFO.task_type,
                downstream_task_type=PHYSIONET_MI_INFO.downstream_task_type,
                num_labels=4,  # 4-class: left, right, fists, feet
                category_list=["left", "right", "fists", "feet"],
                sampling_rate=PHYSIONET_MI_INFO.sampling_rate,
                montage=PHYSIONET_MI_INFO.montage,
                channels=PHYSIONET_MI_INFO.channels,
            )
        else:
            dataset_info = PHYSIONET_MI_INFO

        # Create subject attributes
        subject_attrs = SubjectAttrs(
            subject_id=subject_id,
            dataset_name=dataset_info.dataset_name,
            task_type=dataset_info.task_type.value,
            downstream_task_type=dataset_info.downstream_task_type.value,
            rsFreq=self.target_sfreq,
            chn_name=ch_names,
            num_labels=dataset_info.num_labels,
            category_list=dataset_info.category_list,
            chn_pos=None,
            chn_ori=None,
            chn_type="EEG",
            montage=dataset_info.montage,
        )

        # Create output file
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / f"sub_{subject_id}.h5"

        with HDF5Writer(str(output_path), subject_attrs) as writer:
            for trial in all_trials:
                # Store additional label information in trial metadata
                trial_attrs = TrialAttrs(
                    trial_id=trial['trial_id'],
                    session_id=trial['session_id'],
                )
                # Add label metadata to trial attributes (will be stored in HDF5)
                trial_attrs.clinical_metadata = {
                    'body_part_label': int(trial['body_part_label']),
                    'movement_type_label': int(trial['movement_type_label']),
                    'combined_label': int(trial['label']),
                    'body_part_name': ['left', 'right', 'fists', 'feet'][trial['body_part_label']],
                    'movement_type_name': ['real', 'imagined'][trial['movement_type_label']],
                }
                trial_name = writer.add_trial(trial_attrs)

                # Segment into windows
                data = trial['data']  # shape = (n_chs, n_times) in µV
                n_chs, total_samples = data.shape

                start_sample = 0
                seg_id = 0

                while start_sample + self.window_samples <= total_samples:
                    end_sample = start_sample + self.window_samples
                    seg_data = data[:, start_sample:end_sample]

                    seg_start_time = trial['onset_time'] + start_sample / self.target_sfreq
                    seg_end_time = seg_start_time + self.window_sec

                    # Label format depends on imagined_only flag
                    if self.imagined_only:
                        # 4-class: only body_part_label
                        segment_label = np.array([trial['body_part_label']])
                    else:
                        # 8-class: tuple (body_part_label, movement_type_label)
                        segment_label = np.array([trial['body_part_label'], trial['movement_type_label']])
                    
                    segment_attrs = SegmentAttrs(
                        segment_id=seg_id,
                        start_time=seg_start_time,
                        end_time=seg_end_time,
                        time_length=self.window_sec,
                        label=segment_label,
                    )
                    writer.add_segment(trial_name, segment_attrs, seg_data)

                    start_sample += self.stride_samples
                    seg_id += 1

        # Report validation statistics
        self._report_validation_stats(subject_id)
        print(f"Saved {output_path} ({self.valid_trials} valid trials)")
        return str(output_path)

    def build_all(self, subject_ids: list[int] = None) -> list[str]:
        """
        Build HDF5 files for all subjects.

        Args:
            subject_ids: List of subject IDs to process (None = all)

        Returns:
            List of output file paths
        """
        if subject_ids is None:
            subject_ids = self.get_subject_ids()

        output_paths = []
        failed_subjects = []
        all_total_trials = 0
        all_valid_trials = 0
        all_rejected_trials = 0

        for subject_id in subject_ids:
            try:
                output_path = self.build_subject(subject_id)
                output_paths.append(output_path)
                all_total_trials += self.total_trials
                all_valid_trials += self.valid_trials
                all_rejected_trials += self.rejected_trials
            except Exception as e:
                print(f"Error processing subject {subject_id}: {e}")
                failed_subjects.append(subject_id)
                import traceback
                traceback.print_exc()

        # Summary report
        print("\n" + "=" * 50)
        print("PROCESSING SUMMARY")
        print("=" * 50)
        print(f"Total subjects: {len(subject_ids)}")
        print(f"Successful: {len(output_paths)}")
        print(f"Failed: {len(failed_subjects)}")
        if failed_subjects:
            print(f"Failed subject IDs: {failed_subjects}")
        print(f"\nTotal trials: {all_total_trials}")
        print(f"Valid trials: {all_valid_trials}")
        print(f"Rejected trials: {all_rejected_trials}")
        if all_total_trials > 0:
            print(f"Rejection rate: {all_rejected_trials / all_total_trials * 100:.1f}%")
        print("=" * 50)

        # Save dataset info JSON
        stats = {
            "total_subjects": len(subject_ids),
            "successful_subjects": len(output_paths),
            "failed_subjects": failed_subjects,
            "total_trials": all_total_trials,
            "valid_trials": all_valid_trials,
            "rejected_trials": all_rejected_trials,
            "rejection_rate": f"{all_rejected_trials / all_total_trials * 100:.1f}%" if all_total_trials > 0 else "0%",
        }
        self._save_dataset_info(stats)

        return output_paths


def build_physionet_mi(
    raw_data_dir: str,
    output_dir: str = "./hdf5",
    subject_ids: list[int] = None,
    imagined_only: bool = False,
    **kwargs,
) -> list[str]:
    """
    Convenience function to build PhysioNet MI dataset.

    Args:
        raw_data_dir: Directory containing raw files
        output_dir: Output directory for HDF5 files
        subject_ids: List of subject IDs to process (None = all)
        imagined_only: If True, only process imagined movement tasks (4-class labels)
        **kwargs: Additional arguments for PhysionetMIBuilder

    Returns:
        List of output file paths
    """
    builder = PhysionetMIBuilder(raw_data_dir, output_dir, imagined_only=imagined_only, **kwargs)
    return builder.build_all(subject_ids)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build PhysioNet MI HDF5 dataset")
    parser.add_argument("raw_data_dir", help="Directory containing raw files")
    parser.add_argument("--output_dir", default="./hdf5", help="Output directory")
    parser.add_argument("--subjects", nargs="+", type=int, help="Subject IDs to process")
    parser.add_argument("--target_sfreq", type=float, default=200.0, help="Target sampling rate")
    parser.add_argument("--window_sec", type=float, default=4.0, help="Window length in seconds")
    parser.add_argument("--stride_sec", type=float, default=4.0, help="Stride length in seconds")
    parser.add_argument("--filter_notch", type=float, default=60.0, help="Notch filter frequency")
    parser.add_argument("-i", "--imagined_only", action="store_true", 
                        help="Only process imagined movement tasks (4-class labels: left, right, fists, feet)")
    args = parser.parse_args()

    build_physionet_mi(
        args.raw_data_dir,
        args.output_dir,
        args.subjects,
        imagined_only=args.imagined_only,
        target_sfreq=args.target_sfreq,
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        filter_notch=args.filter_notch,
    )
