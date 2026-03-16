"""
CHB-MIT Scalp EEG Database Dataset Builder.

CHB-MIT Scalp EEG Database
- 24 subjects (chb01-chb24)
- Multiple EDF files per subject
- 23 channels
- 256 Hz sampling rate
- Seizure detection task (2 classes: seizure, non-seizure)
- https://physionet.org/content/chbmit/1.0.0/

Data Unit Handling:
- MNE internally uses Volts (V) as the unit
- Automatically detect unit (V/mV/µV) when reading files and convert to V for MNE processing
- Automatically convert to microvolts (µV) when writing to HDF5, i.e., multiply by 1e6
- Default amplitude validation threshold: 600 µV (adjustable via max_amplitude_uv parameter)
"""

import os
import json
import re
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


# CHB-MIT Dataset Configuration
CHBMIT_INFO = DatasetInfo(
    dataset_name="CHBMIT_2class",
    task_type=DatasetTaskType.SEIZURE,
    downstream_task_type=DownstreamTaskType.CLASSIFICATION,
    num_labels=2,
    category_list=["non_seizure", "seizure"],
    sampling_rate=200.0,  # Target sampling rate (downsampled from 256 Hz)
    montage="10_20",
    channels=[],  # Will be populated from Raw.ch_names at runtime
)

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


class CHBMITBuilder:
    """Builder for CHB-MIT dataset."""

    def __init__(
        self,
        raw_data_dir: str,
        output_dir: str = "./hdf5",
        target_sfreq: float = 200.0,
        window_sec: float = 1.0,
        stride_sec: float = 1.0,
        filter_low: float = 0.5,
        filter_high: float = 70.0,
        filter_notch: float = 60.0,  # 60 Hz for US
        max_amplitude_uv: float = DEFAULT_MAX_AMPLITUDE_UV,
    ):
        """
        Initialize CHB-MIT builder.

        Args:
            raw_data_dir: Directory containing raw files (should point to chbmit/1.0.0/)
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
        if output_path.name == "CHBMIT":
            self.output_dir = output_path
        else:
            self.output_dir = output_path / "CHBMIT"
        self.target_sfreq = target_sfreq
        self.orig_sfreq = 256.0  # Original sampling rate
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
        """Get list of subject IDs (chb01-chb24)."""
        # Find chbmit directory
        chbmit_dir = None
        if (self.raw_data_dir / "chbmit" / "1.0.0").exists():
            chbmit_dir = self.raw_data_dir / "chbmit" / "1.0.0"
        elif (self.raw_data_dir / "physionet.org" / "files" / "chbmit" / "1.0.0").exists():
            chbmit_dir = self.raw_data_dir / "physionet.org" / "files" / "chbmit" / "1.0.0"
        else:
            # Try to find chbmit directory recursively
            for path in self.raw_data_dir.rglob("chbmit/1.0.0"):
                chbmit_dir = path
                break
        
        if chbmit_dir is None:
            raise FileNotFoundError(f"Could not find chbmit/1.0.0 directory in {self.raw_data_dir}")
        
        # Find all subject directories
        subject_dirs = sorted(chbmit_dir.glob("chb*"))
        subject_ids = []
        for sub_dir in subject_dirs:
            if sub_dir.is_dir() and re.match(r'chb\d+', sub_dir.name):
                subject_ids.append(sub_dir.name)
        
        return sorted(subject_ids)

    def _parse_summary_file(self, summary_file: Path) -> dict:
        """
        Parse summary file to extract seizure information.
        
        Returns:
            dict: {filename: [(start_time, end_time), ...]}
        """
        seizures_info = {}
        
        if not summary_file.exists():
            return seizures_info
        
        try:
            with open(summary_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            current_file = None
            current_start = None
            
            for line in content.split('\n'):
                # Match file name
                file_match = re.search(r'File Name: (.+\.edf)', line)
                if file_match:
                    current_file = file_match.group(1)
                    seizures_info[current_file] = []
                    current_start = None
                    continue
                
                # Match seizure times
                if current_file:
                    start_match = re.search(r'Seizure Start Time: (\d+) seconds', line)
                    end_match = re.search(r'Seizure End Time: (\d+) seconds', line)
                    
                    if start_match:
                        current_start = float(start_match.group(1))
                    elif end_match:
                        end_time = float(end_match.group(1))
                        if current_start is not None:
                            seizures_info[current_file].append((current_start, end_time))
                            current_start = None
                        else:
                            # If no start time found, use 0 as start
                            seizures_info[current_file].append((0.0, end_time))
        
        except Exception as e:
            print(f"  Warning: Failed to parse summary file {summary_file.name}: {e}")
        
        return seizures_info

    def _find_files(self, subject_id: str) -> list[dict]:
        """
        Find all EDF files for a subject.
        
        Returns:
            List of dicts with keys: 'edf', 'seizures'
        """
        # Find subject directory
        chbmit_dir = None
        if (self.raw_data_dir / "chbmit" / "1.0.0").exists():
            chbmit_dir = self.raw_data_dir / "chbmit" / "1.0.0"
        elif (self.raw_data_dir / "physionet.org" / "files" / "chbmit" / "1.0.0").exists():
            chbmit_dir = self.raw_data_dir / "physionet.org" / "files" / "chbmit" / "1.0.0"
        else:
            for path in self.raw_data_dir.rglob("chbmit/1.0.0"):
                chbmit_dir = path
                break
        
        if chbmit_dir is None:
            raise FileNotFoundError(f"Could not find chbmit/1.0.0 directory")
        
        sub_dir = chbmit_dir / subject_id
        if not sub_dir.exists():
            raise FileNotFoundError(f"Subject directory not found: {sub_dir}")
        
        # Parse summary file
        summary_file = sub_dir / f"{subject_id}-summary.txt"
        seizures_info = self._parse_summary_file(summary_file)
        
        # Find all EDF files
        edf_files = sorted(sub_dir.glob("*.edf"))
        files = []
        
        for edf_file in edf_files:
            # Skip files with '+' in name (continuation files)
            if '+' in edf_file.name:
                continue
            
            file_info = {
                'edf': edf_file,
                'seizures': seizures_info.get(edf_file.name, []),
            }
            files.append(file_info)
        
        if not files:
            raise FileNotFoundError(f"No EDF files found for subject {subject_id}")
        
        return files

    def _read_raw(self, edf_file: Path):
        """
        Read EDF file and convert to MNE Raw object.
        """
        if not HAS_MNE:
            raise ImportError("MNE is required for reading EDF files")
        
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                raw = mne.io.read_raw_edf(str(edf_file), preload=True, verbose=False)
            
            # MNE's EDF reader typically returns data in Volts already
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
            print(f"  Warning: Failed to read {edf_file.name}: {e}")
            return None

    def _preprocess(self, raw):
        """Apply preprocessing to raw data."""
        # Notch filter (60 Hz for US)
        if self.filter_notch > 0:
            raw.notch_filter(freqs=self.filter_notch, verbose=False)

        # Bandpass filter
        raw.filter(l_freq=self.filter_low, h_freq=self.filter_high, verbose=False)

        # Resample if needed
        if raw.info['sfreq'] != self.target_sfreq:
            raw.resample(self.target_sfreq, verbose=False)

        return raw

    def _get_segment_label(self, start_time: float, end_time: float, seizures: list) -> int:
        """
        Determine label for a segment based on seizure times.
        
        Args:
            start_time: Segment start time in seconds
            end_time: Segment end time in seconds
            seizures: List of (start, end) tuples for seizures
        
        Returns:
            0 for non-seizure, 1 for seizure
        """
        segment_center = (start_time + end_time) / 2.0
        
        for seizure_start, seizure_end in seizures:
            # Check if segment overlaps with seizure
            if start_time < seizure_end and end_time > seizure_start:
                return 1  # Seizure
        
        return 0  # Non-seizure

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
            subject_id: Subject identifier (e.g., "chb01")

        Returns:
            Path to output HDF5 file
        """
        if not HAS_MNE:
            raise ImportError("MNE is required for building CHB-MIT dataset")

        # Reset validation counters
        self.total_segments = 0
        self.valid_segments = 0
        self.rejected_segments = 0

        # Find files
        files = self._find_files(subject_id)
        
        all_segments = []
        ch_names = None
        trial_counter = 0

        # Process each EDF file
        for file_info in files:
            edf_file = file_info['edf']
            seizures = file_info['seizures']
            
            print(f"Reading {edf_file.name} ({len(seizures)} seizures)")
            raw = self._read_raw(edf_file)
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
            for start_sample in range(0, n_samples - self.window_samples + 1, self.stride_samples):
                end_sample = start_sample + self.window_samples
                segment_data_uv = data_uv[:, start_sample:end_sample]
                
                # Validate segment amplitude
                self.total_segments += 1
                if not self._validate_segment(segment_data_uv):
                    self.rejected_segments += 1
                    continue
                
                self.valid_segments += 1
                
                # Calculate time
                start_time = start_sample / self.target_sfreq
                end_time = start_time + self.window_sec
                
                # Get label (seizure or non-seizure)
                label = self._get_segment_label(start_time, end_time, seizures)
                
                all_segments.append({
                    'data': segment_data_uv,  # Store in µV
                    'trial_id': trial_counter,
                    'file_name': edf_file.name,
                    'label': label,
                    'start_time': start_time,
                    'end_time': end_time,
                })
                
                trial_counter += 1  # Increment for each segment

        if not all_segments:
            raise ValueError(f"No valid segments extracted for subject {subject_id}")

        # Create subject attributes
        subject_attrs = SubjectAttrs(
            subject_id=subject_id,
            dataset_name=CHBMIT_INFO.dataset_name,
            task_type=CHBMIT_INFO.task_type.value,
            downstream_task_type=CHBMIT_INFO.downstream_task_type.value,
            rsFreq=self.target_sfreq,
            chn_name=ch_names,
            num_labels=CHBMIT_INFO.num_labels,
            category_list=CHBMIT_INFO.category_list,
            chn_pos=None,
            chn_ori=None,
            chn_type="EEG",
            montage="10_20",
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
                    session_id=0,  # CHB-MIT doesn't have explicit sessions
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
        channels_to_save = channel_names if channel_names else CHBMIT_INFO.channels
        
        info = {
            "dataset": {
                "name": CHBMIT_INFO.dataset_name,
                "description": "CHB-MIT Scalp EEG Database - Seizure Detection",
                "task_type": str(CHBMIT_INFO.task_type.value),
                "downstream_task": str(CHBMIT_INFO.downstream_task_type.value),
                "num_labels": CHBMIT_INFO.num_labels,
                "category_list": CHBMIT_INFO.category_list,
                "original_sampling_rate": self.orig_sfreq,
                "channels": channels_to_save,
                "channel_count": len(channels_to_save) if channels_to_save else 0,
                "montage": CHBMIT_INFO.montage,
                "source_url": "https://physionet.org/content/chbmit/1.0.0/",
                "note": "Channel count varies across subjects (22-31 channels). Listed channels represent the most common configuration (23 channels).",
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
        all_seizure_segments = 0
        all_non_seizure_segments = 0
        
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
                
                # Count seizure vs non-seizure segments (would need to track this)
                # For now, just report total segments
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


def build_chbmit(
    raw_data_dir: str,
    output_dir: str = "./hdf5",
    subject_ids: list[str] = None,
    **kwargs,
) -> list[str]:
    """
    Convenience function to build CHB-MIT dataset.

    Args:
        raw_data_dir: Directory containing raw files (should point to chbmit/1.0.0/ or parent)
        output_dir: Output directory for HDF5 files
        subject_ids: List of subject IDs to process (None = all)
        **kwargs: Additional arguments for CHBMITBuilder

    Returns:
        List of output file paths
    """
    builder = CHBMITBuilder(raw_data_dir, output_dir, **kwargs)
    return builder.build_all(subject_ids)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build CHB-MIT HDF5 dataset")
    parser.add_argument("raw_data_dir", help="Directory containing raw files (chbmit/1.0.0/ or parent)")
    parser.add_argument("--output_dir", default="./hdf5", help="Output directory")
    parser.add_argument("--subjects", nargs="+", help="Subject IDs to process (e.g., chb01 chb02)")
    args = parser.parse_args()

    build_chbmit(args.raw_data_dir, args.output_dir, args.subjects)

