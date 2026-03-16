"""
AD65 Dataset Builder.

AD65 Dataset
- 88 subjects (ID: 1-88)
- 500 Hz sampling rate
- 19 channels
- https://openneuro.org/datasets/ds004504/versions/1.0.2/file-display/README

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
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from schema import SubjectAttrs, TrialAttrs, SegmentAttrs
    from hdf5_io import HDF5Writer
    from config import DatasetInfo, DatasetTaskType, DownstreamTaskType

AD65_INFO = DatasetInfo(
    dataset_name="AD65",
    task_type=DatasetTaskType.RESTING_STATE,
    downstream_task_type=DownstreamTaskType.CLASSIFICATION,
    num_labels=3,
    category_list=["AD", "FTD", "HC"],  # 0=Alzheimer, 1=Frontotemporal Dementia, 2=Healthy
    sampling_rate=500.0,
    montage="10_20",
    channels=["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz", "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2"],
)

# Channels to remove (EOG)
REMOVE_CHANNELS = []

REFERENCE_CHANNELS = ['A1', 'A2']  # Mastoid references

# Default amplitude threshold (µV)
DEFAULT_MAX_AMPLITUDE_UV = 600.0

class AD65Builder:
    """Builder for AD65 dataset."""

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
        file_format: str = "auto",
        max_amplitude_uv: float = DEFAULT_MAX_AMPLITUDE_UV,
    ):
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir) / "AD65"
        self.target_sfreq = target_sfreq
        self.orig_sfreq = 500.0
        self.window_sec = window_sec
        self.stride_sec = stride_sec
        self.filter_low = filter_low
        self.filter_high = filter_high
        self.filter_notch = filter_notch
        self.file_format = file_format
        self.max_amplitude_uv = max_amplitude_uv

        self.window_samples = int(window_sec * target_sfreq)
        self.stride_samples = int(stride_sec * target_sfreq)

        # Validation statistics
        self.total_trials = 0
        self.valid_trials = 0
        self.rejected_trials = 0

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
                "name": AD65_INFO.dataset_name,
                "description": "AD65 Dataset - Alzheimer's, FTD, and Healthy Controls EEG",
                "task_type": str(AD65_INFO.task_type.value),
                "downstream_task": str(AD65_INFO.downstream_task_type.value),
                "num_labels": AD65_INFO.num_labels,
                "category_list": AD65_INFO.category_list,
                "original_sampling_rate": AD65_INFO.sampling_rate,
                "channels": AD65_INFO.channels,
                "montage": AD65_INFO.montage,
                "source_url": "https://openneuro.org/datasets/ds004504/versions/1.0.2",
            },
            "processing": {
                "target_sampling_rate": self.target_sfreq,
                "window_sec": self.window_sec,
                "stride_sec": self.stride_sec,
                "filter_low": self.filter_low,
                "filter_high": self.filter_high,
                "filter_notch": self.filter_notch,
                "reference_channels": REFERENCE_CHANNELS,
                "max_amplitude_uv": self.max_amplitude_uv,
            },
            "statistics": stats,
            "generated_at": datetime.now().isoformat(),
        }

        json_path = self.output_dir / "dataset_info.json"
        with open(json_path, "w") as f:
            json.dump(info, f, indent=2)
        print(f"Saved dataset info to {json_path}")

    def get_subject_ids(self) -> list[int]:
        """Get list of subject IDs (1-88)."""
        return list(range(1, 89))

    def _detect_file_format(self) -> str:
        """Auto-detect file format from directory structure."""
        gdf_files = list(self.raw_data_dir.rglob("*.gdf"))
        mat_files = list(self.raw_data_dir.rglob("*.mat"))
        edf_files = list(self.raw_data_dir.rglob("*.edf"))
        set_files = list(self.raw_data_dir.rglob("*.set"))
        if gdf_files and mat_files and edf_files:
            print("Warning: Both GDF, MAT and EDF files found. Using EDF format.")
            return "edf"
        elif gdf_files:
            print("Warning: GDF files found. Using GDF format.")
            return "gdf"
        elif mat_files:
            print("Warning: MAT files found. Using MAT format.")
            return "mat"
        elif set_files:
            return "set"
        elif edf_files:
            print("Warning: EDF files found. Using EDF format.")
            return "edf"
        else:
            raise FileNotFoundError("No files found in directory.")

    def _find_files(self, subject_id: int) -> dict[str, list[Path]]:
        """
        仅返回单个被试的set文件。
        目录结构假设：
            <raw_data_dir>/sub-001_task-eyesclosed_eeg.set
        """
        ext = '.set'
        pattern = f'sub-{subject_id:03d}_task-eyesclosed_eeg{ext}'
        data = next(self.raw_data_dir.glob(pattern), None)   # Path 对象 or None
        if data is None:
            raise FileNotFoundError(f'{pattern} not found in {self.raw_data_dir}')
        return data   
        
    def _preprocess(self, raw):
        """Apply preprocessing to raw data."""
        # # Drop EOG channels if present
        # channels_to_drop = [ch for ch in REMOVE_CHANNELS if ch in raw.ch_names]
        # if channels_to_drop:
        #     raw.drop_channels(channels_to_drop)
        
        # Notch filter
        if self.filter_notch > 0:
            raw.notch_filter(freqs=self.filter_notch, verbose=False)
        
        # Bandpass filter
        raw.filter(l_freq=self.filter_low, h_freq=self.filter_high, verbose=False)
        
        # Resample if needed
        if raw.info['sfreq'] != self.target_sfreq:
            raw.resample(self.target_sfreq, verbose=False)
        
        return raw

    def _read_set(self, file_path: Path):
        """Read raw set file."""
        raw = mne.io.read_raw_eeglab(file_path, preload=True)

        # Print channel info
        print(f"  Channels ({len(raw.ch_names)}): {raw.ch_names}")

        # Check for reference channels
        ref_chs = [ch for ch in REFERENCE_CHANNELS if ch in raw.ch_names]
        if ref_chs:
            print(f"  Reference channels found: {ref_chs}")
            # Re-reference to mastoid channels and drop them
            raw.set_eeg_reference(ref_channels=ref_chs, verbose=False)
            raw.drop_channels(ref_chs)
            print(f"  After re-referencing, channels ({len(raw.ch_names)}): {raw.ch_names}")
        else:
            print(f"  No reference channels found, using common average reference")
            raw.set_eeg_reference('average', verbose=False)

        # Auto-detect unit and convert to Volts for MNE
        max_amp = np.abs(raw._data).max()
        if max_amp > 1e-3:  # > 0.001, likely microvolts
            raw._data = raw._data / 1e6
            detected_unit = "µV"
        else:  # likely already Volts
            detected_unit = "V"
        print(f"  Detected unit: {detected_unit}, max amplitude: {max_amp:.2e}")

        return raw

    def _extract_trials(self, raw) -> list[dict]:
        """Extract trials from annotations. for resting state, return entire recording as single trial.""" 
        trials = []
        
        trials.append({
            'onset': 0.0,
            'label': 0,
        })
        
        return trials

    def _get_label(self,subject_id:int):
        if subject_id<=37:
            label=1
        elif subject_id<=66:
            label=2
        else:
            label=3
        return label;

    def build_subject(self, subject_id: int) -> str:
        """
        Build HDF5 file for a single subject.

        Args:
            subject_id: Subject identifier (1-39)

        Returns:
            Path to output HDF5 file
        """
        if not HAS_MNE:
            raise ImportError("MNE and scipy are required for building CIRE dataset")

        # Reset validation counters
        self.total_trials = 0
        self.valid_trials = 0
        self.rejected_trials = 0

        subdata = self._find_files(subject_id)
        print(subdata)

        all_trials = []
        ch_names = None
        trial_counter = 0
        
        session_time_offset = 0.0
                
        try:
            raw = self._read_set(subdata)
            raw = self._preprocess(raw)


            if ch_names is None:
                ch_names = raw.ch_names
                    
            # Extract trials from this file
            trials = self._extract_trials(raw)
            data = raw.get_data()
                    
            # Process each trial
            for trial in trials:
                onset_sample = int(trial['onset'] * self.target_sfreq)
                end_sample = onset_sample + self.window_samples
                        
                trial_data = data
                            
                # Calculate absolute time within session
                trial_start_time = session_time_offset + trial['onset']
                label=self._get_label(subject_id)
                            
                all_trials.append({
                    'data': trial_data,
                    'label': label,
                    'trial_id': trial_counter,
                    'onset_time': trial_start_time,
                })
                trial_counter += 1
                    
            # Update session time offset (add file duration)
            file_duration = data.shape[1] / self.target_sfreq
            session_time_offset += file_duration
                    
        except Exception as e:
            print(f"Error processing {subdata}: {e}")
            import traceback
            traceback.print_exc()

        if not all_trials:
            raise ValueError(f"No valid trials extracted for subject {subject_id}")

        # Create subject attributes
        subject_attrs = SubjectAttrs(
            subject_id=subject_id,
            dataset_name="AD65_3class",
            task_type="resting_state",
            downstream_task_type="classification",
            num_labels=3,
            category_list=["AD", "FTD", "HC"],
            rsFreq=self.target_sfreq,
            chn_name=ch_names,
            chn_pos=None,
            chn_ori=None,
            chn_type="EEG",
            montage="10_20",
        )

        # Create output file
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / f"sub_{subject_id}.h5"

        with HDF5Writer(str(output_path), subject_attrs) as writer:
            for trial in all_trials:
                # For non-resting-state tasks, validate at trial level
                if AD65_INFO.task_type != DatasetTaskType.RESTING_STATE:
                    trial_data_uv = trial['data'] * 1e6
                    self.total_trials += 1
                    if not self._validate_trial(trial_data_uv):
                        self.rejected_trials += 1
                        print(f"  Skipping trial {trial['trial_id']}: amplitude {np.abs(trial_data_uv).max():.1f} µV > {self.max_amplitude_uv} µV")
                        continue
                    self.valid_trials += 1

                trial_attrs = TrialAttrs(
                    trial_id=trial['trial_id'],
                    session_id=0,
                )
                trial_name = writer.add_trial(trial_attrs)

                # 计算采样点
                sfreq = self.target_sfreq
                window_samples = int(self.window_sec * sfreq)
                step_samples   = window_samples
                # 当前 trial 的整条数据
                data = trial['data']              # shape = (n_chs, n_times)
                n_chs, total_samples = data.shape

                start_sample = 0
                seg_id = 0

                while start_sample + window_samples <= total_samples:
                    end_sample = start_sample + window_samples
                    seg_data = data[:, start_sample:end_sample]

                    # Convert from V back to μV for validation and export
                    seg_data_uv = seg_data * 1e6

                    # For resting state, validate at segment level
                    if AD65_INFO.task_type == DatasetTaskType.RESTING_STATE:
                        self.total_trials += 1  # counting segments
                        if not self._validate_trial(seg_data_uv):
                            self.rejected_trials += 1
                            start_sample += step_samples
                            continue
                        self.valid_trials += 1

                    # 对应绝对时间
                    seg_start_time = trial['onset_time'] + start_sample / sfreq
                    seg_end_time   = seg_start_time + self.window_sec

                    segment_attrs = SegmentAttrs(
                        segment_id=seg_id,
                        start_time=seg_start_time,
                        end_time=seg_end_time,
                        time_length=self.window_sec,
                        label=np.array([trial['label']]),
                    )
                    writer.add_segment(trial_name, segment_attrs, seg_data_uv)

                    start_sample += step_samples
                    seg_id += 1

        # Report validation statistics
        is_resting = AD65_INFO.task_type == DatasetTaskType.RESTING_STATE
        self._report_validation_stats(subject_id, is_resting)
        unit = "segments" if is_resting else "trials"
        print(f"Saved {output_path} ({self.valid_trials} valid {unit})")
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
        is_resting = AD65_INFO.task_type == DatasetTaskType.RESTING_STATE
        unit = "segments" if is_resting else "trials"
        print("\n" + "=" * 50)
        print("PROCESSING SUMMARY")
        print("=" * 50)
        print(f"Total subjects: {len(subject_ids)}")
        print(f"Successful: {len(output_paths)}")
        print(f"Failed: {len(failed_subjects)}")
        if failed_subjects:
            print(f"Failed subject IDs: {failed_subjects}")
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


def build_ad65(
    raw_data_dir: str,
    output_dir: str = "./hdf5",
    subject_ids: list[int] = None,
    **kwargs,
) -> list[str]:
    """
    Convenience function to build CIRE dataset.

    Args:
        raw_data_dir: Directory containing raw files
        output_dir: Output directory for HDF5 files
        subject_ids: List of subject IDs to process (None = all)
        **kwargs: Additional arguments for CIREBuilder

    Returns:
        List of output file paths
    """
    builder = AD65Builder(raw_data_dir, output_dir, **kwargs)
    return builder.build_all(subject_ids)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build AD65 HDF5 dataset")
    parser.add_argument("raw_data_dir",help="Directory containing raw files ")
    parser.add_argument("--output_dir", default="/mnt/dataset2/Processed_datasets/EEG_Bench/AD65", help="Output directory")
    parser.add_argument("--subjects", nargs="+", type=int, help="Subject IDs to process")
    parser.add_argument("--format", default="set", help="File format (auto-detect if not specified)")
    args = parser.parse_args()

    build_ad65(args.raw_data_dir, args.output_dir, args.subjects, file_format=args.format)