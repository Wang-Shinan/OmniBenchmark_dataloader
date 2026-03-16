"""
PD31 Dataset Builder.

UC San Diego Resting State EEG Data from Patients with Parkinson's Disease
- 31 subjects (16 PD patients, 15 healthy controls)
- 41 channels (32 EEG after removing EXG1-8 and Status)
- 512 Hz sampling rate
- Resting state task
- Multiple sessions per subject:
  - PD patients: ses-on (on medication), ses-off (off medication)
  - Healthy controls: ses-hc (healthy control)
- https://openneuro.org/datasets/ds002778

Data Unit Handling:
- MNE's read_raw_bdf automatically converts BDF data to Volts
- Convert to µV when writing to HDF5 (multiply by 1e6)
- Default amplitude validation threshold: 600 µV
"""

import os
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


# PD31 Dataset Configuration
PD31_INFO = DatasetInfo(
    dataset_name="PD31_2class",
    task_type=DatasetTaskType.RESTING_STATE,
    downstream_task_type=DownstreamTaskType.CLASSIFICATION,
    num_labels=2,
    category_list=["HC", "PD"],  # 0=Healthy Control, 1=Parkinson's Disease
    sampling_rate=512.0,  # Original sampling rate
    montage="10_10",
    channels=[],  # Will be populated from Raw.ch_names at runtime
)

# Label mapping based on subject type
LABEL_MAP = {
    "HC": 0,  # Healthy Control
    "PD": 1,  # Parkinson's Disease
}

# Session name mapping (raw session -> descriptive name for trial attrs)
SESSION_NAME_MAP = {
    "hc": "hc",
    "on": "on_medication",
    "off": "off_medication",
}

# Default amplitude threshold (µV) for validation
DEFAULT_MAX_AMPLITUDE_UV = 600.0

REMOVED_CHANNELS = ["EXG1", "EXG2", "EXG3", "EXG4", "EXG5", "EXG6", "EXG7", "EXG8", "Status"]


def detect_unit_and_convert_to_volts(data: np.ndarray) -> tuple[np.ndarray, str]:
    """
    Auto-detect data unit and convert to Volts for MNE.

    MNE uses Volts internally, so we need to ensure data is in V before processing.

    Args:
        data: Input data array (shape: n_channels x n_samples)

    Returns:
        tuple: (data_in_volts, detected_unit)
    """
    max_amp = np.abs(data).max()
    if max_amp > 1e-3:  # > 0.001, likely microvolts (µV)
        return data / 1e6, "µV"
    return data, "V"


class PD31Builder:
    """Builder for PD31 Resting-state EEG dataset."""

    def __init__(
        self,
        raw_data_dir: str,
        output_dir: str = "./hdf5",
        target_sfreq: float = 200.0,
        window_sec: float = 1.0,
        stride_sec: float = 1.0,
        filter_low: float = 0.1,
        filter_high: float = 75.0,
        filter_notch: float = 60.0,  # 60 Hz for US
        max_amplitude_uv: float = DEFAULT_MAX_AMPLITUDE_UV,
    ):
        """
        Initialize PD31 builder.

        Args:
            raw_data_dir: Directory containing raw files
            output_dir: Output directory for HDF5 files
            target_sfreq: Target sampling frequency
            window_sec: Window length in seconds
            stride_sec: Stride length in seconds
            filter_low: Low cutoff frequency for bandpass filter
            filter_high: High cutoff frequency for bandpass filter
            filter_notch: Notch filter frequency (60 Hz for US)
            max_amplitude_uv: Maximum amplitude threshold in µV for validation
        """
        self.raw_data_dir = Path(raw_data_dir)
        output_path = Path(output_dir)
        if output_path.name == "PD31":
            self.output_dir = output_path
        else:
            self.output_dir = output_path / "PD31"
        self.target_sfreq = target_sfreq
        self.orig_sfreq = 512.0  # Original sampling rate
        self.window_sec = window_sec
        self.stride_sec = stride_sec
        self.filter_low = filter_low
        self.filter_high = filter_high
        self.filter_notch = filter_notch
        self.max_amplitude_uv = max_amplitude_uv

        self.window_samples = int(window_sec * target_sfreq)
        self.stride_samples = int(stride_sec * target_sfreq)

        # Track validation statistics
        self.total_segments = 0
        self.valid_segments = 0
        self.rejected_segments = 0

    def get_subject_ids(self) -> list[str]:
        """Get list of subject IDs from BDF files."""
        bdf_files = sorted(self.raw_data_dir.glob("sub-*_ses-*_task-rest_eeg.bdf"))
        subject_ids = set()
        
        for bdf_file in bdf_files:
            # Extract subject ID (e.g., sub-pd3_ses-on_task-rest_eeg.bdf -> pd3)
            parts = bdf_file.stem.split('_')
            if len(parts) >= 1:
                sub_id = parts[0].replace('sub-', '')
                subject_ids.add(sub_id)
        
        return sorted(list(subject_ids))

    def _find_files(self, subject_id: str) -> list[dict]:
        """
        Find all BDF files for a subject.

        Returns:
            List of dicts with keys: 'bdf', 'session', 'session_name', 'subject_type', 'label'
        """
        # Find all BDF files for this subject
        pattern = f"sub-{subject_id}_ses-*_task-rest_eeg.bdf"
        bdf_files = sorted(self.raw_data_dir.glob(pattern))

        files = []
        for bdf_file in bdf_files:
            # Extract session ID (e.g., ses-on, ses-off, ses-hc)
            parts = bdf_file.stem.split('_')
            session = None
            for part in parts:
                if part.startswith('ses-'):
                    session = part.replace('ses-', '')
                    break

            # Determine subject type (PD or HC)
            subject_type = 'PD' if subject_id.startswith('pd') else 'HC'

            # Get label based on subject type
            label = LABEL_MAP.get(subject_type, 0)

            # Get descriptive session name
            session_name = SESSION_NAME_MAP.get(session, session)

            file_info = {
                'bdf': bdf_file,
                'session': session,
                'session_name': session_name,
                'subject_type': subject_type,
                'label': label,
            }
            files.append(file_info)

        if not files:
            raise FileNotFoundError(f"No BDF files found for subject {subject_id}")

        return files

    def _read_raw(self, bdf_file: Path):
        """
        Read BDF file and return MNE Raw object.

        Note: MNE's read_raw_bdf automatically converts data to Volts using
        the calibration factors in the BDF header. No manual unit conversion needed.
        """
        if not HAS_MNE:
            raise ImportError("MNE is required for reading BDF files")

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                raw = mne.io.read_raw_bdf(str(bdf_file), preload=True, verbose=False)

            # MNE reads BDF files in Volts - just report the amplitude
            max_amp = np.abs(raw._data).max()
            print(f"  Data loaded, max amplitude: {max_amp:.2e} V")

            return raw
        except Exception as e:
            print(f"  Warning: Failed to read {bdf_file.name}: {e}")
            return None

    def _preprocess(self, raw):
        """Apply preprocessing to raw data."""
        # Drop non-EEG channels (EXG1-8, Status)
        channels_to_drop = [ch for ch in REMOVED_CHANNELS if ch in raw.ch_names]
        if channels_to_drop:
            raw.drop_channels(channels_to_drop)
            print(f"  Dropped {len(channels_to_drop)} channels: {channels_to_drop}")

        # Apply average reference
        raw.set_eeg_reference('average', verbose=False)

        # Notch filter (60 Hz for US)
        if self.filter_notch > 0:
            raw.notch_filter(freqs=self.filter_notch, verbose=False)

        # Bandpass filter
        raw.filter(l_freq=self.filter_low, h_freq=self.filter_high, verbose=False)

        # Resample if needed
        if raw.info['sfreq'] != self.target_sfreq:
            raw.resample(self.target_sfreq, verbose=False)

        return raw

    def _extract_segments(self, raw, label: int, session_name: str) -> list[dict]:
        """
        Extract segments from raw data using sliding window.

        Args:
            raw: MNE Raw object
            label: Class label (0=HC, 1=PD)
            session_name: Session name (hc, on_medication, off_medication)

        Returns:
            List of segment dicts with keys: 'data', 'start_time', 'end_time', 'label', 'session_name'
        """
        segments = []

        # Get data (in Volts)
        data = raw.get_data()  # shape: (n_channels, n_samples)
        n_channels, n_samples = data.shape
        sfreq = raw.info['sfreq']

        # Convert to µV
        data_uv = data * 1e6

        # Extract segments using sliding window
        start_sample = 0
        segment_counter = 0

        while start_sample + self.window_samples <= n_samples:
            end_sample = start_sample + self.window_samples

            # Extract segment data
            segment_data_uv = data_uv[:, start_sample:end_sample]

            # Validate segment amplitude
            self.total_segments += 1
            if not self._validate_segment(segment_data_uv):
                self.rejected_segments += 1
                start_sample += self.stride_samples
                continue

            self.valid_segments += 1

            start_time = start_sample / sfreq
            end_time = end_sample / sfreq

            segments.append({
                'data': segment_data_uv,  # Store in µV
                'start_time': start_time,
                'end_time': end_time,
                'segment_id': segment_counter,
                'label': label,
                'session_name': session_name,
            })

            segment_counter += 1
            start_sample += self.stride_samples

        return segments

    def _validate_segment(self, segment_data: np.ndarray) -> bool:
        """
        Validate segment amplitude.

        Args:
            segment_data: Segment data in µV (shape: n_channels x n_samples)

        Returns:
            True if amplitude is within threshold, False otherwise
        """
        return np.abs(segment_data).max() <= self.max_amplitude_uv

    def _report_validation_stats(self, subject_id: str):
        """Report validation statistics."""
        valid_pct = (self.valid_segments / self.total_segments * 100) if self.total_segments > 0 else 0
        print(f"Subject {subject_id} validation report:")
        print(f"  Amplitude threshold: {self.max_amplitude_uv} µV")
        print(f"  Total segments: {self.total_segments}")
        print(f"  Valid segments: {self.valid_segments} ({valid_pct:.1f}%)")
        print(f"  Rejected segments: {self.rejected_segments} ({100-valid_pct:.1f}%)")

    def build_subject(self, subject_id: str) -> str:
        """
        Build HDF5 file for a single subject.

        Args:
            subject_id: Subject identifier (e.g., "pd3", "hc1")

        Returns:
            Path to output HDF5 file
        """
        if not HAS_MNE:
            raise ImportError("MNE is required for building PD31 dataset")

        # Reset validation counters
        self.total_segments = 0
        self.valid_segments = 0
        self.rejected_segments = 0

        # Find files
        files = self._find_files(subject_id)
        
        all_segments = []
        ch_names = None
        trial_counter = 0

        # Process each BDF file
        for file_info in files:
            bdf_file = file_info['bdf']
            session = file_info['session']
            session_name = file_info['session_name']
            subject_type = file_info['subject_type']
            label = file_info['label']

            print(f"Reading {bdf_file.name} (Session: {session_name}, Type: {subject_type}, Label: {label})")
            raw = self._read_raw(bdf_file)
            if raw is None:
                continue

            raw = self._preprocess(raw)

            if ch_names is None:
                ch_names = raw.ch_names

            # Extract segments with label and session_name
            segments = self._extract_segments(raw, label, session_name)
            print(f"  Extracted {len(segments)} valid segments")
            
            for segment in segments:
                segment['trial_id'] = trial_counter
                segment['session'] = session
                segment['subject_type'] = subject_type
                trial_counter += 1
                all_segments.append(segment)

        if not all_segments:
            raise ValueError(f"No valid segments extracted for subject {subject_id}")

        # Create subject attributes
        subject_attrs = SubjectAttrs(
            subject_id=subject_id,
            dataset_name=PD31_INFO.dataset_name,
            task_type=PD31_INFO.task_type.value,
            downstream_task_type=PD31_INFO.downstream_task_type.value,
            rsFreq=self.target_sfreq,
            chn_name=ch_names,
            num_labels=PD31_INFO.num_labels,
            category_list=PD31_INFO.category_list,
            chn_pos=None,
            chn_ori=None,
            chn_type="EEG",
            montage="10_10",
        )

        # Create output file
        if self.output_dir.exists() and not self.output_dir.is_dir():
            raise FileExistsError(
                f"Output directory path exists as a file: {self.output_dir}. "
                f"Please remove it or choose a different output directory."
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / f"sub_{subject_id}.h5"

        with HDF5Writer(str(output_path), subject_attrs) as writer:
            for segment in all_segments:
                trial_attrs = TrialAttrs(
                    trial_id=segment['trial_id'],
                    session_id=0,
                    task_name=segment['session_name'],  # "hc", "on_medication", or "off_medication"
                )
                trial_name = writer.add_trial(trial_attrs)

                segment_attrs = SegmentAttrs(
                    segment_id=segment['segment_id'],
                    start_time=segment['start_time'],
                    end_time=segment['end_time'],
                    time_length=self.window_sec,
                    label=np.array([segment['label']]),
                )
                writer.add_segment(trial_name, segment_attrs, segment['data'])

        # Report validation statistics
        self._report_validation_stats(subject_id)
        print(f"Saved {output_path}")
        return str(output_path)

    def _save_dataset_info(self, stats: dict):
        """Save dataset info and processing parameters to JSON."""
        info = {
            "dataset": {
                "name": PD31_INFO.dataset_name,
                "description": "UC San Diego Resting State EEG Data from Patients with Parkinson's Disease",
                "task_type": str(PD31_INFO.task_type.value),
                "downstream_task": str(PD31_INFO.downstream_task_type.value),
                "num_labels": PD31_INFO.num_labels,
                "category_list": PD31_INFO.category_list,
                "original_sampling_rate": self.orig_sfreq,
                "channels": PD31_INFO.channels,
                "montage": PD31_INFO.montage,
                "source_url": "https://openneuro.org/datasets/ds002778",
                "label_map": LABEL_MAP,
                "removed_channels": REMOVED_CHANNELS,
            },
            "processing": {
                "target_sampling_rate": self.target_sfreq,
                "window_sec": self.window_sec,
                "stride_sec": self.stride_sec,
                "filter_low": self.filter_low,
                "filter_high": self.filter_high,
                "filter_notch": self.filter_notch,
                "max_amplitude_uv": self.max_amplitude_uv,
            },
            "statistics": stats,
            "generated_at": datetime.now().isoformat(),
        }

        if self.output_dir.exists() and not self.output_dir.is_dir():
            raise FileExistsError(
                f"Output directory path exists as a file: {self.output_dir}. "
                f"Please remove it or choose a different output directory."
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        json_path = self.output_dir / "dataset_info.json"
        with open(json_path, "w") as f:
            json.dump(info, f, indent=2)
        print(f"Saved dataset info to {json_path}")

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
        all_total_segments = 0
        all_valid_segments = 0
        all_rejected_segments = 0

        for subject_id in subject_ids:
            try:
                output_path = self.build_subject(subject_id)
                output_paths.append(output_path)
                all_total_segments += self.total_segments
                all_valid_segments += self.valid_segments
                all_rejected_segments += self.rejected_segments
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
        print(f"\nTotal segments: {all_total_segments}")
        print(f"Valid segments: {all_valid_segments}")
        print(f"Rejected segments: {all_rejected_segments}")
        if all_total_segments > 0:
            print(f"Rejection rate: {all_rejected_segments / all_total_segments * 100:.1f}%")
        print("=" * 50)

        # Save dataset info JSON
        stats = {
            "total_subjects": len(subject_ids),
            "successful_subjects": len(output_paths),
            "failed_subjects": failed_subjects,
            "total_segments": all_total_segments,
            "valid_segments": all_valid_segments,
            "rejected_segments": all_rejected_segments,
            "rejection_rate": f"{all_rejected_segments / all_total_segments * 100:.1f}%" if all_total_segments > 0 else "0%",
        }
        self._save_dataset_info(stats)

        return output_paths


def build_pd31(
    raw_data_dir: str,
    output_dir: str = "./hdf5",
    subject_ids: list[str] = None,
    **kwargs,
) -> list[str]:
    """
    Convenience function to build PD31 dataset.

    Args:
        raw_data_dir: Directory containing raw files
        output_dir: Output directory for HDF5 files
        subject_ids: List of subject IDs to process (None = all)
        **kwargs: Additional arguments for PD31Builder

    Returns:
        List of output file paths
    """
    builder = PD31Builder(raw_data_dir, output_dir, **kwargs)
    return builder.build_all(subject_ids)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build PD31 HDF5 dataset")
    parser.add_argument("raw_data_dir", help="Directory containing raw files")
    parser.add_argument("--output_dir", default="./hdf5", help="Output directory")
    parser.add_argument("--subjects", nargs="+", help="Subject IDs to process (e.g., pd3 hc1)")
    args = parser.parse_args()

    build_pd31(args.raw_data_dir, args.output_dir, args.subjects)

