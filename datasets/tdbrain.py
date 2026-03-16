"""
TDBRAIN Dataset Builder.

TDBRAIN Dataset - Treatment-resistant Depression Brain Dataset
- 1274 subjects
- 200 Hz sampling rate
- 33 channels (27 EEG + 6 non-EEG)
- Resting state: eyes closed (restEC) and eyes open (restEO)
- Multiple diagnostic categories
- https://brainclinics.com/resources/tdbrain-dataset/introduction/downloads

"""

import os
import re
import json
import warnings
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

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

# Standard EEG channels (10-20 system)
TDBRAIN_EEG_CHANNELS = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
    'FC3', 'FCz', 'FC4',
    'T7', 'C3', 'Cz', 'C4', 'T8',
    'CP3', 'CPz', 'CP4',
    'P7', 'P3', 'Pz', 'P4', 'P8',
    'O1', 'Oz', 'O2'
]

# Non-EEG channels to remove
TDBRAIN_REMOVE_CHANNELS = ['VPVA', 'VNVB', 'HPHL', 'HNHR', 'Erbs', 'OrbOcc', 'Mass']

# All channels in CSV files
TDBRAIN_ALL_CHANNELS = TDBRAIN_EEG_CHANNELS + TDBRAIN_REMOVE_CHANNELS

TDBRAIN_INFO = DatasetInfo(
    dataset_name="TDBRAIN",
    task_type=DatasetTaskType.RESTING_STATE,
    downstream_task_type=DownstreamTaskType.CLASSIFICATION,
    num_labels=0,  # Will be determined from participants file
    category_list=[],  # Will be determined from participants file
    sampling_rate=200.0,
    montage="10_20",
    channels=TDBRAIN_EEG_CHANNELS,
)

# Default amplitude threshold (µV)
DEFAULT_MAX_AMPLITUDE_UV = 600.0


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


class TDBRAINBuilder:
    """Builder for TDBRAIN dataset."""

    def __init__(
        self,
        raw_data_dir: str,
        output_dir: str = "./hdf5",
        target_sfreq: float = 200.0,
        window_sec: float = 2.0,
        stride_sec: float = 2.0,
        filter_low: float = 0.1,
        filter_high: float = 75.0,
        filter_notch: float = 50.0,
        participants_file: str = None,
        max_amplitude_uv: float = DEFAULT_MAX_AMPLITUDE_UV,
        use_diagnosis_labels: bool = True,
    ):
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir) / "TDBRAIN"
        self.target_sfreq = target_sfreq
        self.orig_sfreq = 200.0
        self.window_sec = window_sec
        self.stride_sec = stride_sec
        self.filter_low = filter_low
        self.filter_high = filter_high
        self.filter_notch = filter_notch
        self.max_amplitude_uv = max_amplitude_uv
        self.use_diagnosis_labels = use_diagnosis_labels

        self.window_samples = int(window_sec * target_sfreq)
        self.stride_samples = int(stride_sec * target_sfreq)

        # Load participants file if provided
        self.participants_df = None
        self.label_map = {}
        if participants_file:
            self._load_participants(participants_file)

        # Validation statistics
        self.total_trials = 0
        self.valid_trials = 0
        self.rejected_trials = 0

    def _load_participants(self, participants_file: str):
        """Load participants TSV file and create label mapping."""
        try:
            if participants_file.endswith('.zip'):
                import zipfile
                z = zipfile.ZipFile(participants_file)
                files = [f for f in z.namelist() if f.endswith('.tsv') and not f.startswith('__MACOSX')]
                if files:
                    self.participants_df = pd.read_csv(z.open(files[0]), sep='\t')
            else:
                self.participants_df = pd.read_csv(participants_file, sep='\t')

            if 'participants_ID' in self.participants_df.columns and 'indication' in self.participants_df.columns:
                # Create label mapping from indication
                unique_indications = self.participants_df['indication'].dropna().unique()
                unique_indications = sorted([ind for ind in unique_indications if ind != 'REPLICATION'])
                
                # Create label map: indication -> label_id
                for label_id, indication in enumerate(unique_indications):
                    self.label_map[indication] = label_id
                
                # Map subject IDs to labels
                self.subject_labels = {}
                for _, row in self.participants_df.iterrows():
                    sub_id = row['participants_ID']
                    indication = row['indication']
                    if pd.notna(indication) and indication in self.label_map:
                        self.subject_labels[sub_id] = self.label_map[indication]
                
                print(f"Loaded {len(self.subject_labels)} subjects with labels")
                print(f"Label mapping: {dict(list(self.label_map.items())[:10])}")
        except Exception as e:
            print(f"Warning: Could not load participants file: {e}")
            self.participants_df = None
            self.label_map = {}
            self.subject_labels = {}

    def _validate_trial(self, trial_data: np.ndarray) -> bool:
        """Check if trial amplitude is within acceptable range."""
        return np.abs(trial_data).max() <= self.max_amplitude_uv

    def _report_validation_stats(self, subject_id: int, is_resting: bool = True):
        """Report validation statistics."""
        unit = "segments" if is_resting else "trials"
        valid_pct = (self.valid_trials / self.total_trials * 100) if self.total_trials > 0 else 0
        print(f"Subject {subject_id} validation report:")
        print(f"  Amplitude threshold: {self.max_amplitude_uv} µV")
        print(f"  Total {unit}: {self.total_trials}")
        print(f"  Valid {unit}: {self.valid_trials} ({valid_pct:.1f}%)")
        print(f"  Rejected {unit}: {self.rejected_trials} ({100-valid_pct:.1f}%)")

    def _save_dataset_info(self, stats: dict):
        """Save dataset info and processing parameters to JSON."""
        info = {
            "dataset": {
                "name": TDBRAIN_INFO.dataset_name,
                "description": "TDBRAIN Dataset - Treatment-resistant Depression Brain Dataset",
                "task_type": str(TDBRAIN_INFO.task_type.value),
                "downstream_task": str(TDBRAIN_INFO.downstream_task_type.value),
                "num_labels": len(self.label_map) if self.label_map else 0,
                "category_list": list(self.label_map.keys()) if self.label_map else [],
                "original_sampling_rate": TDBRAIN_INFO.sampling_rate,
                "channels": TDBRAIN_INFO.channels,
                "montage": TDBRAIN_INFO.montage,
                "source_url": "https://brainclinics.com/resources/tdbrain-dataset/introduction/downloads",
            },
            "processing": {
                "target_sampling_rate": self.target_sfreq,
                "window_sec": self.window_sec,
                "stride_sec": self.stride_sec,
                "filter_low": self.filter_low,
                "filter_high": self.filter_high,
                "filter_notch": self.filter_notch,
                "removed_channels": TDBRAIN_REMOVE_CHANNELS,
                "max_amplitude_uv": self.max_amplitude_uv,
            },
            "statistics": stats,
            "generated_at": datetime.now().isoformat(),
        }

        json_path = self.output_dir / "dataset_info.json"
        with open(json_path, "w") as f:
            json.dump(info, f, indent=2)
        print(f"Saved dataset info to {json_path}")

    def get_subject_ids(self) -> list[str]:
        """Get list of subject IDs from directory structure."""
        subject_dirs = [d for d in self.raw_data_dir.iterdir() if d.is_dir() and d.name.startswith('sub-')]
        subject_ids = [d.name.replace('sub-', '') for d in subject_dirs]
        return sorted(subject_ids)

    def _find_files(self, subject_id: str) -> dict[str, Path]:
        """
        Find CSV files for a subject.
        Returns dict with task names as keys and file paths as values.
        """
        subject_dir = self.raw_data_dir / f"sub-{subject_id}"
        if not subject_dir.exists():
            raise FileNotFoundError(f"Subject directory not found: {subject_dir}")

        files = {}
        # Look for ses-1/eeg/ directory
        ses_dir = subject_dir / "ses-1" / "eeg"
        if ses_dir.exists():
            for task in ['restEC', 'restEO']:
                pattern = f"sub-{subject_id}_ses-1_task-{task}_eeg.csv"
                file_path = ses_dir / pattern
                if file_path.exists():
                    files[task] = file_path

        if not files:
            raise FileNotFoundError(f"No CSV files found for subject {subject_id} in {subject_dir}")

        return files

    def _read_csv(self, file_path: Path):
        """Read CSV file and convert to MNE Raw object."""
        # Read CSV (try C engine first, fallback to Python engine)
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"  Warning: C engine failed, trying Python engine: {e}")
            df = pd.read_csv(file_path, engine='python')
        
        # Get channel names from header
        ch_names = df.columns.tolist()
        
        # Reorder channels to match expected order (if possible)
        ordered_ch_names = []
        for ch in TDBRAIN_ALL_CHANNELS:
            if ch in ch_names:
                ordered_ch_names.append(ch)
        
        # Add any remaining channels
        for ch in ch_names:
            if ch not in ordered_ch_names:
                ordered_ch_names.append(ch)
        
        # Extract data (transpose to channels x time)
        data = df[ordered_ch_names].values.T
        
        # Auto-detect unit and convert to Volts for MNE
        data_volts, detected_unit = detect_unit_and_convert_to_volts(data)
        print(f"  Detected unit: {detected_unit}, max amplitude: {np.abs(data).max():.2e}")

        # Create MNE info
        info = mne.create_info(
            ch_names=ordered_ch_names,
            sfreq=self.orig_sfreq,
            ch_types=['eeg'] * len(ordered_ch_names)
        )
        
        raw = mne.io.RawArray(data_volts, info, verbose=False)
        return raw

    def _preprocess(self, raw):
        """Apply preprocessing to raw data."""
        # Drop non-EEG channels
        channels_to_drop = [ch for ch in TDBRAIN_REMOVE_CHANNELS if ch in raw.ch_names]
        if channels_to_drop:
            raw.drop_channels(channels_to_drop)
            print(f"  Dropped non-EEG channels: {channels_to_drop}")

        # Set common average reference
        raw.set_eeg_reference('average')

        # Notch filter
        if self.filter_notch > 0:
            raw.notch_filter(freqs=self.filter_notch)

        # Bandpass filter
        raw.filter(l_freq=self.filter_low, h_freq=self.filter_high)

        # Resample if needed
        if raw.info['sfreq'] != self.target_sfreq:
            raw.resample(self.target_sfreq)

        return raw

    def _get_label(self, subject_id: str) -> int:
        """Get label for a subject."""
        sub_id_str = f"sub-{subject_id}"
        if self.use_diagnosis_labels and hasattr(self, 'subject_labels'):
            return self.subject_labels.get(sub_id_str, 0)
        return 0  # Default label if no mapping available

    def build_subject(self, subject_id: str) -> str:
        """
        Build HDF5 file for a single subject.

        Args:
            subject_id: Subject identifier (e.g., "88079017")

        Returns:
            Path to output HDF5 file
        """
        if not HAS_MNE:
            raise ImportError("MNE is required for building TDBRAIN dataset")

        # Reset validation counters
        self.total_trials = 0
        self.valid_trials = 0
        self.rejected_trials = 0

        # Find files
        files = self._find_files(subject_id)
        print(f"Processing subject {subject_id}, found {len(files)} task(s)")

        all_trials = []
        ch_names = None
        trial_counter = 0
        session_time_offset = 0.0

        # Process each task (restEC, restEO)
        for task_name, file_path in files.items():
            print(f"  Reading {file_path.name}")

            try:
                raw = self._read_csv(file_path)
                raw = self._preprocess(raw)

                if ch_names is None:
                    ch_names = raw.ch_names

                # Get data (in Volts)
                data = raw.get_data()

                # Create a trial for this task
                trial_data = data  # shape: (n_channels, n_samples)

                # Get label for this subject
                label = self._get_label(subject_id)

                all_trials.append({
                    'data': trial_data,
                    'label': label,
                    'trial_id': trial_counter,
                    'task': task_name,
                    'onset_time': session_time_offset,
                })
                trial_counter += 1

                # Update session time offset
                file_duration = data.shape[1] / self.target_sfreq
                session_time_offset += file_duration

            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                import traceback
                traceback.print_exc()

        if not all_trials:
            raise ValueError(f"No valid trials extracted for subject {subject_id}")

        # Determine number of labels and category list
        num_labels = len(self.label_map) if self.label_map else 1
        category_list = list(self.label_map.keys()) if self.label_map else ["unknown"]

        # Create subject attributes
        subject_attrs = SubjectAttrs(
            subject_id=int(subject_id) if subject_id.isdigit() else 0,
            dataset_name=TDBRAIN_INFO.dataset_name,
            task_type=TDBRAIN_INFO.task_type.value,
            downstream_task_type=TDBRAIN_INFO.downstream_task_type.value,
            rsFreq=self.target_sfreq,
            chn_name=ch_names,
            num_labels=num_labels,
            category_list=category_list,
            chn_pos=None,
            chn_ori=None,
            chn_type="EEG",
            montage=TDBRAIN_INFO.montage,
        )

        # Create output file
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / f"sub_{subject_id}.h5"

        with HDF5Writer(str(output_path), subject_attrs) as writer:
            for trial in all_trials:
                trial_attrs = TrialAttrs(
                    trial_id=trial['trial_id'],
                    session_id=0,
                )
                trial_name = writer.add_trial(trial_attrs)

                # Segment into windows
                data = trial['data']  # shape = (n_chs, n_times) in Volts
                n_chs, total_samples = data.shape

                start_sample = 0
                seg_id = 0

                while start_sample + self.window_samples <= total_samples:
                    end_sample = start_sample + self.window_samples
                    seg_data = data[:, start_sample:end_sample]

                    # Convert from V back to μV for validation and export
                    seg_data_uv = seg_data * 1e6

                    # Validate at segment level (resting state)
                    self.total_trials += 1
                    if not self._validate_trial(seg_data_uv):
                        self.rejected_trials += 1
                        start_sample += self.stride_samples
                        continue
                    self.valid_trials += 1

                    # Calculate absolute time
                    seg_start_time = trial['onset_time'] + start_sample / self.target_sfreq
                    seg_end_time = seg_start_time + self.window_sec

                    segment_attrs = SegmentAttrs(
                        segment_id=seg_id,
                        start_time=seg_start_time,
                        end_time=seg_end_time,
                        time_length=self.window_sec,
                        label=np.array([trial['label']]),
                    )
                    writer.add_segment(trial_name, segment_attrs, seg_data_uv)

                    start_sample += self.stride_samples
                    seg_id += 1

        # Report validation statistics
        is_resting = TDBRAIN_INFO.task_type == DatasetTaskType.RESTING_STATE
        self._report_validation_stats(subject_id, is_resting)
        unit = "segments" if is_resting else "trials"
        print(f"Saved {output_path} ({self.valid_trials} valid {unit})")
        return str(output_path)

    def build_all(self, subject_ids: list[str] = None) -> list[str]:
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
        is_resting = TDBRAIN_INFO.task_type == DatasetTaskType.RESTING_STATE
        unit = "segments" if is_resting else "trials"
        print("\n" + "=" * 50)
        print("PROCESSING SUMMARY")
        print("=" * 50)
        print(f"Total subjects: {len(subject_ids)}")
        print(f"Successful: {len(output_paths)}")
        print(f"Failed: {len(failed_subjects)}")
        if failed_subjects:
            print(f"Failed subject IDs: {failed_subjects[:10]}...")  # Show first 10
        print(f"\nTotal {unit}: {all_total_trials}")
        print(f"Valid {unit}: {all_valid_trials}")
        print(f"Rejected {unit}: {all_rejected_trials}")
        if all_total_trials > 0:
            print(f"Rejection rate: {all_rejected_trials / all_total_trials * 100:.1f}%")
        print("=" * 50)

        # Save dataset info JSON
        stats = {
            "total_subjects": len(subject_ids),
            "successful_subjects": len(output_paths),
            "failed_subjects": failed_subjects,
            f"total_{unit}": all_total_trials,
            f"valid_{unit}": all_valid_trials,
            f"rejected_{unit}": all_rejected_trials,
            "rejection_rate": f"{all_rejected_trials / all_total_trials * 100:.1f}%" if all_total_trials > 0 else "0%",
        }
        self._save_dataset_info(stats)

        return output_paths


def build_tdbrain(
    raw_data_dir: str,
    output_dir: str = "./hdf5",
    subject_ids: list[str] = None,
    participants_file: str = None,
    **kwargs,
) -> list[str]:
    """
    Convenience function to build TDBRAIN dataset.

    Args:
        raw_data_dir: Directory containing raw files (derivatives folder)
        output_dir: Output directory for HDF5 files
        subject_ids: List of subject IDs to process (None = all)
        participants_file: Path to participants TSV file or ZIP containing it
        **kwargs: Additional arguments for TDBRAINBuilder

    Returns:
        List of output file paths
    """
    builder = TDBRAINBuilder(raw_data_dir, output_dir, participants_file=participants_file, **kwargs)
    return builder.build_all(subject_ids)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build TDBRAIN HDF5 dataset")
    parser.add_argument("raw_data_dir", help="Directory containing raw files (derivatives folder)")
    parser.add_argument("--output_dir", default="./hdf5", help="Output directory")
    parser.add_argument("--subjects", nargs="+", type=str, help="Subject IDs to process")
    parser.add_argument("--participants_file", help="Path to participants TSV file or ZIP")
    parser.add_argument("--target_sfreq", type=float, default=200.0, help="Target sampling rate")
    parser.add_argument("--window_sec", type=float, default=2.0, help="Window length in seconds")
    parser.add_argument("--stride_sec", type=float, default=2.0, help="Stride length in seconds")
    parser.add_argument("--filter_notch", type=float, default=50.0, help="Notch filter frequency")
    args = parser.parse_args()

    build_tdbrain(
        args.raw_data_dir,
        args.output_dir,
        args.subjects,
        participants_file=args.participants_file,
        target_sfreq=args.target_sfreq,
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        filter_notch=args.filter_notch,
    )
