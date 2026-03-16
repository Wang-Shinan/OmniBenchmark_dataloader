"""
RestCog Dataset Builder.

A test-retest resting and cognitive state EEG dataset
- 60 subjects
- 3 sessions per subject
- 5 tasks: eyesclosed, eyesopen, mathematic, memory, music
- 64 channels (variable: 61-64)
- 500 Hz sampling rate
- 300 seconds (5 minutes) per task
- https://openneuro.org/datasets/ds004148

Data Unit Handling:
- MNE internally uses Volts (V) as the unit
- Automatically detect unit (V/mV/µV) when reading files and convert to V for MNE processing
- Automatically convert to microvolts (µV) when writing to HDF5, i.e., multiply by 1e6
- Default amplitude validation threshold: 600 µV (adjustable via max_amplitude_uv parameter)
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


# RestCog Dataset Configuration
# For now, we'll process all tasks as separate datasets or combine them
RESTCOG_INFO = DatasetInfo(
    dataset_name="RestCog",
    task_type=DatasetTaskType.COGNITIVE,
    downstream_task_type=DownstreamTaskType.CLASSIFICATION,
    num_labels=5,  # 5 tasks: eyesclosed, eyesopen, mathematic, memory, music
    category_list=["eyesclosed", "eyesopen", "mathematic", "memory", "music"],
    sampling_rate=250.0,  # Target sampling rate (downsampled from 500 Hz)
    montage="10_10",
    channels=[],  # Will be populated from Raw.ch_names at runtime
)

# Task name mapping
TASK_NAME_MAP = {
    "eyesclosed": 0,
    "eyesopen": 1,
    "mathematic": 2,
    "memory": 3,
    "music": 4,
}

# Default amplitude threshold (µV) for validation
DEFAULT_MAX_AMPLITUDE_UV = 600.0


def detect_unit_and_convert_to_volts(data: np.ndarray) -> tuple[np.ndarray, str]:
    """
    Auto-detect data unit and convert to Volts for MNE.

    MNE uses Volts internally, so we need to ensure data is in V before processing.
    This function detects the unit based on amplitude range and converts accordingly.
    
    Uses robust statistics (percentile) instead of max to avoid noise/artifact interference.

    Args:
        data: Input data array (shape: n_channels x n_samples)

    Returns:
        tuple: (data_in_volts, detected_unit)
    """
    # Use 99th percentile instead of max to be robust against noise/artifacts
    abs_data = np.abs(data)
    robust_max = np.percentile(abs_data, 99.0)
    
    # Fallback to max if percentile is too small (edge case)
    max_amp = max(robust_max, np.percentile(abs_data, 95.0))
    
    # Also check median absolute deviation (MAD) as a sanity check
    mad = np.median(abs_data)
    if mad > 0:
        mad_based_estimate = 3 * mad
        max_amp = max(max_amp, mad_based_estimate)

    if max_amp > 1e-2:  # > 0.01, likely microvolts (µV)
        return data / 1e6, "µV"
    elif max_amp > 1e-5:  # > 0.00001, likely millivolts (mV)
        return data / 1e3, "mV"
    else:  # likely already Volts (V)
        return data, "V"


class RestCogBuilder:
    """Builder for RestCog dataset."""

    def __init__(
        self,
        raw_data_dir: str,
        output_dir: str = "./hdf5",
        target_sfreq: float = 200.0,
        window_sec: float = 1.0,
        stride_sec: float = 1.0,
        filter_low: float = 0.1,
        filter_high: float = 75.0,
        filter_notch: float = 50.0,  # 50 Hz for China
        max_amplitude_uv: float = DEFAULT_MAX_AMPLITUDE_UV,
        task_filter: list[str] = None,  # Filter specific tasks (None = all tasks)
    ):
        """
        Initialize RestCog builder.

        Args:
            raw_data_dir: Directory containing raw files (ds004148-download)
            output_dir: Output directory for HDF5 files
            target_sfreq: Target sampling frequency
            window_sec: Window length in seconds
            stride_sec: Stride length in seconds
            filter_low: Low cutoff frequency for bandpass filter
            filter_high: High cutoff frequency for bandpass filter
            filter_notch: Notch filter frequency (50 Hz for China)
            max_amplitude_uv: Maximum amplitude threshold in µV for validation
            task_filter: List of task names to process (None = all tasks)
        """
        self.raw_data_dir = Path(raw_data_dir)
        output_path = Path(output_dir)
        if output_path.name == "RestCog":
            self.output_dir = output_path
        else:
            self.output_dir = output_path / "RestCog"
        self.target_sfreq = target_sfreq
        self.orig_sfreq = 500.0  # Original sampling rate
        self.window_sec = window_sec
        self.stride_sec = stride_sec
        self.filter_low = filter_low
        self.filter_high = filter_high
        self.filter_notch = filter_notch
        self.max_amplitude_uv = max_amplitude_uv
        self.task_filter = task_filter if task_filter else list(TASK_NAME_MAP.keys())

        self.window_samples = int(window_sec * target_sfreq)
        self.stride_samples = int(stride_sec * target_sfreq)

        # Track validation statistics
        self.total_segments = 0
        self.valid_segments = 0
        self.rejected_segments = 0

    def get_subject_ids(self) -> list[int]:
        """Get list of subject IDs (1-60)."""
        subject_dirs = sorted(self.raw_data_dir.glob("sub-*"))
        subject_ids = []
        for sub_dir in subject_dirs:
            try:
                sub_id = int(sub_dir.name.split("-")[1])
                # Check if any session has task files
                for ses_dir in sub_dir.glob("ses-*"):
                    eeg_dir = ses_dir / "eeg"
                    if eeg_dir.exists():
                        vhdr_files = list(eeg_dir.glob("*_task-*_eeg.vhdr"))
                        if vhdr_files:
                            subject_ids.append(sub_id)
                            break
            except (ValueError, IndexError):
                continue
        return sorted(subject_ids)

    def _find_files(self, subject_id: int) -> list[dict]:
        """
        Find all task files for a subject across all sessions.
        
        Returns:
            List of dicts with keys: 'vhdr', 'task', 'session_id'
        """
        sub_dir = self.raw_data_dir / f"sub-{subject_id:02d}"
        files = []
        
        # Find all sessions
        for ses_dir in sorted(sub_dir.glob("ses-*")):
            ses_name = ses_dir.name  # e.g., "ses-session1"
            ses_id = int(ses_name.split("session")[1])  # Extract number from "session1" -> 1
            
            eeg_dir = ses_dir / "eeg"
            if not eeg_dir.exists():
                continue
            
            # Find all task files
            for task_name in self.task_filter:
                vhdr_file = eeg_dir / f"sub-{subject_id:02d}_{ses_name}_task-{task_name}_eeg.vhdr"
                
                if vhdr_file.exists():
                    files.append({
                        'vhdr': vhdr_file,
                        'task': task_name,
                        'session_id': ses_id,
                        'session_name': ses_name,
                    })
        
        if not files:
            raise FileNotFoundError(f"No task files found for subject {subject_id}")
        
        return files

    def _read_raw(self, vhdr_file: Path):
        """
        Read BrainVision file and convert to MNE Raw object.
        
        Note: MNE's BrainVision reader typically handles unit conversion automatically,
        so we trust MNE's unit handling and don't perform additional conversion.
        """
        if not HAS_MNE:
            raise ImportError("MNE is required for reading BrainVision files")
        
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                raw = mne.io.read_raw_brainvision(str(vhdr_file), preload=True, verbose=False)
            
            # MNE's BrainVision reader typically returns data in Volts already
            # Check if data is in reasonable range for Volts (0.001 to 1 V)
            # If data is much larger (> 10 V), it might be in µV and needs conversion
            if hasattr(raw, '_data') and raw._data is not None:
                max_amp = np.abs(raw._data).max()
                # If max amplitude > 10 V, likely in µV (typical EEG is 10-100 µV = 1e-5 to 1e-4 V)
                if max_amp > 10.0:
                    raw._data = raw._data / 1e6
                    print(f"  Detected unit: µV (max={max_amp:.2e}), converted to V")
                elif max_amp > 1.0:
                    # Between 1-10 V, could be mV
                    raw._data = raw._data / 1e3
                    print(f"  Detected unit: mV (max={max_amp:.2e}), converted to V")
                # Otherwise, assume already in Volts (typical range: 0.001-1 V)
            
            return raw
        except Exception as e:
            print(f"  Warning: Failed to read {vhdr_file.name}: {e}")
            return None

    def _preprocess(self, raw):
        """Apply preprocessing to raw data."""
        # Notch filter (50 Hz for China)
        if self.filter_notch > 0:
            raw.notch_filter(freqs=self.filter_notch, verbose=False)

        # Bandpass filter
        raw.filter(l_freq=self.filter_low, h_freq=self.filter_high, verbose=False)

        # Resample if needed
        if raw.info['sfreq'] != self.target_sfreq:
            raw.resample(self.target_sfreq, verbose=False)

        return raw

    def _validate_segment(self, segment_data: np.ndarray) -> bool:
        """
        Validate segment amplitude.

        Args:
            segment_data: Segment data in µV (shape: n_channels x n_samples)

        Returns:
            True if amplitude is within threshold, False otherwise
        """
        return np.abs(segment_data).max() <= self.max_amplitude_uv

    def _report_validation_stats(self, subject_id: int):
        """Report validation statistics."""
        valid_pct = (self.valid_segments / self.total_segments * 100) if self.total_segments > 0 else 0
        print(f"Subject {subject_id} validation report:")
        print(f"  Amplitude threshold: {self.max_amplitude_uv} µV")
        print(f"  Total segments: {self.total_segments}")
        print(f"  Valid segments: {self.valid_segments} ({valid_pct:.1f}%)")
        print(f"  Rejected segments: {self.rejected_segments} ({100-valid_pct:.1f}%)")

    def build_subject(self, subject_id: int) -> str:
        """
        Build HDF5 file for a single subject.

        Args:
            subject_id: Subject identifier

        Returns:
            Path to output HDF5 file
        """
        if not HAS_MNE:
            raise ImportError("MNE is required for building RestCog dataset")

        # Reset validation counters
        self.total_segments = 0
        self.valid_segments = 0
        self.rejected_segments = 0

        # Find files
        files = self._find_files(subject_id)
        
        all_segments = []
        ch_names = None
        trial_counter = 0

        # Process each task file
        for file_info in files:
            vhdr_file = file_info['vhdr']
            task_name = file_info['task']
            session_id = file_info['session_id']
            
            print(f"Reading {vhdr_file.name} (Task: {task_name}, Session: {session_id})")
            raw = self._read_raw(vhdr_file)
            if raw is None:
                continue
            
            raw = self._preprocess(raw)
            
            if ch_names is None:
                ch_names = raw.ch_names
            
            # Get data (in Volts)
            data = raw.get_data()  # shape: (n_channels, n_samples)
            n_channels, n_samples = data.shape
            
            # Convert to µV
            data_uv = data * 1e6
            
            # Segment into windows (sliding window)
            # Each segment becomes a separate trial
            # Only create segments that fit exactly within the data
            for start_sample in range(0, n_samples - self.window_samples + 1, self.stride_samples):
                end_sample = start_sample + self.window_samples
                
                # Ensure we don't exceed data bounds
                if end_sample > n_samples:
                    break
                
                segment_data_uv = data_uv[:, start_sample:end_sample]
                
                # Validate segment shape (must match window_samples exactly)
                if segment_data_uv.shape[1] != self.window_samples:
                    print(f"  Warning: Skipping incomplete segment at {start_sample} (shape: {segment_data_uv.shape[1]} != {self.window_samples})")
                    continue
                
                # Validate segment amplitude
                self.total_segments += 1
                if not self._validate_segment(segment_data_uv):
                    self.rejected_segments += 1
                    continue
                
                self.valid_segments += 1
                
                # Calculate time
                start_time = start_sample / self.target_sfreq
                end_time = start_time + self.window_sec
                
                # Get task label
                task_label = TASK_NAME_MAP[task_name]
                
                all_segments.append({
                    'data': segment_data_uv,  # Store in µV, shape: (n_channels, window_samples)
                    'trial_id': trial_counter,  # Each segment is a separate trial
                    'session_id': session_id,
                    'task': task_name,
                    'label': task_label,
                    'start_time': start_time,
                    'end_time': end_time,
                })
                
                trial_counter += 1  # Increment for each segment
            
            # Report if there's remaining data that couldn't form a complete window
            # Find the last segment start position for this task
            last_segment_start = None
            for seg in reversed(all_segments):
                if seg.get('task') == task_name:
                    last_segment_start = int(seg['start_time'] * self.target_sfreq)
                    break
            
            if last_segment_start is not None:
                last_segment_end = last_segment_start + self.window_samples
                remaining_samples = n_samples - last_segment_end
                if remaining_samples > 0:
                    if remaining_samples < self.window_samples:
                        print(f"  Info: {remaining_samples} samples remaining (less than window size {self.window_samples}, skipped)")
                    else:
                        print(f"  Warning: {remaining_samples} samples remaining but not processed (check window logic)")

        if not all_segments:
            raise ValueError(f"No valid segments extracted for subject {subject_id}")

        # Create subject attributes
        subject_attrs = SubjectAttrs(
            subject_id=subject_id,
            dataset_name=RESTCOG_INFO.dataset_name,
            task_type=RESTCOG_INFO.task_type.value,
            downstream_task_type=RESTCOG_INFO.downstream_task_type.value,
            rsFreq=self.target_sfreq,
            chn_name=ch_names,
            num_labels=RESTCOG_INFO.num_labels,
            category_list=RESTCOG_INFO.category_list,
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
            for seg_idx, segment in enumerate(all_segments):
                trial_attrs = TrialAttrs(
                    trial_id=segment['trial_id'],
                    session_id=segment['session_id'],
                )
                trial_name = writer.add_trial(trial_attrs)

                segment_attrs = SegmentAttrs(
                    segment_id=0,  # Single segment per trial
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

    def _save_dataset_info(self, stats: dict, channel_names: list = None):
        """Save dataset info and processing parameters to JSON."""
        # Use provided channel names or fall back to default
        channels_to_save = channel_names if channel_names else RESTCOG_INFO.channels
        
        info = {
            "dataset": {
                "name": RESTCOG_INFO.dataset_name,
                "description": "A test-retest resting and cognitive state EEG dataset",
                "task_type": str(RESTCOG_INFO.task_type.value),
                "downstream_task": str(RESTCOG_INFO.downstream_task_type.value),
                "num_labels": RESTCOG_INFO.num_labels,
                "category_list": RESTCOG_INFO.category_list,
                "original_sampling_rate": self.orig_sfreq,
                "channels": channels_to_save,
                "channel_count": len(channels_to_save) if channels_to_save else 0,
                "montage": RESTCOG_INFO.montage,
                "source_url": "https://openneuro.org/datasets/ds004148",
            },
            "processing": {
                "target_sampling_rate": self.target_sfreq,
                "window_sec": self.window_sec,
                "stride_sec": self.stride_sec,
                "filter_low": self.filter_low,
                "filter_high": self.filter_high,
                "filter_notch": self.filter_notch,
                "task_filter": self.task_filter,
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
        all_total_segments = 0
        all_valid_segments = 0
        all_rejected_segments = 0
        
        # Store channel names from first successfully processed subject
        first_ch_names = None

        for subject_id in subject_ids:
            try:
                output_path = self.build_subject(subject_id)
                output_paths.append(output_path)
                all_total_segments += self.total_segments
                all_valid_segments += self.valid_segments
                all_rejected_segments += self.rejected_segments
                
                # Extract channel names from first successful subject's HDF5 file
                if first_ch_names is None:
                    try:
                        import h5py
                        with h5py.File(output_path, 'r') as f:
                            if 'chn_name' in f.attrs:
                                first_ch_names = [name.decode('utf-8') if isinstance(name, bytes) else name 
                                                 for name in f.attrs['chn_name']]
                    except Exception as e:
                        print(f"  Warning: Could not extract channel names from {output_path}: {e}")
                        
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
        self._save_dataset_info(stats, first_ch_names)

        return output_paths


def build_restcog(
    raw_data_dir: str,
    output_dir: str = "./hdf5",
    subject_ids: list[int] = None,
    **kwargs,
) -> list[str]:
    """
    Convenience function to build RestCog dataset.

    Args:
        raw_data_dir: Directory containing raw files (ds004148-download)
        output_dir: Output directory for HDF5 files
        subject_ids: List of subject IDs to process (None = all)
        **kwargs: Additional arguments for RestCogBuilder

    Returns:
        List of output file paths
    """
    builder = RestCogBuilder(raw_data_dir, output_dir, **kwargs)
    return builder.build_all(subject_ids)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build RestCog HDF5 dataset")
    parser.add_argument("raw_data_dir", help="Directory containing raw files (ds004148-download)")
    parser.add_argument("--output_dir", default="./hdf5", help="Output directory")
    parser.add_argument("--subjects", nargs="+", type=int, help="Subject IDs to process")
    parser.add_argument("--tasks", nargs="+", choices=list(TASK_NAME_MAP.keys()), help="Tasks to process (default: all)")
    args = parser.parse_args()

    build_restcog(args.raw_data_dir, args.output_dir, args.subjects, task_filter=args.tasks)

