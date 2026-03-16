"""
HFO Dataset Builder.

HFO Dataset
- 30 subjects (ID: 1-30)
- 1024 Hz sampling rate
- 23 channels
- https://openneuro.org/datasets/ds003555/versions/1.0.1
"""

import os
import re
import json 
from datetime import datetime
import warnings
import pandas as pd
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
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from schema import SubjectAttrs, TrialAttrs, SegmentAttrs
    from hdf5_io import HDF5Writer
    from config import DatasetInfo, DatasetTaskType, DownstreamTaskType

HFO_INFO = DatasetInfo(
    dataset_name="HFO",
    task_type=DatasetTaskType.RESTING_STATE,
    downstream_task_type=DownstreamTaskType.DETECTION,
    num_labels=5,
    category_list=["Sleep Stage:","1:N3","2:N2","3:Awake","4:REM","0:other"],
    sampling_rate=1024.0,
    montage="10_20",
    channels=['Fp1','A2','Fp2','F7','F3',
              'Fz','F4','F8','T3','C3',	
              'Cz','C4','T4','T5','P3',
              'Pz','P4','T6','O1','A1',	
              'O2','T1','T2'],
)

# Channels to remove (EOG)
REMOVE_CHANNELS = []

DEFAULT_MAX_AMPLITUDE_UV = 600.0

def detect_unit_and_convert_to_volts(data: np.ndarray) -> tuple[np.ndarray, str]:
    """Auto-detect data unit and convert to Volts for MNE."""
    max_amp = np.abs(data).max()
    if max_amp > 1e-3:  # likely uV
        return data / 1e6  #turn to V

    return data
class HFOBuilder:
    """Builder for HFO dataset."""

    def __init__(
        self,
        raw_data_dir: str,
        output_dir: str = "./hdf5",
        target_sfreq: float = 200.0,
        window_sec: float = 1.0,
        stride_sec: float = 1.0,
        filter_low: float = 0.1,
        filter_high: float = 75.0,
        filter_notch: float = 50.0,
        max_amplitude_uv: float = DEFAULT_MAX_AMPLITUDE_UV,
    ):
        """
        Initialize HFO builder.

        Args:
            raw_data_dir: Directory containing raw files
            output_dir: Output directory for HDF5 files
            target_sfreq: Target sampling frequency
            window_sec: Window length in seconds
            stride_sec: Stride length in seconds
            filter_low: Low cutoff frequency for bandpass filter
            filter_high: High cutoff frequency for bandpass filter
            filter_notch: Notch filter frequency (50Hz for Europe)
            file_format: File format ('auto', 'gdf', or 'mat')
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir) / "HFO"
        self.target_sfreq = target_sfreq
        self.orig_sfreq = 1024.0  # Original sampling rate
        self.window_sec = window_sec
        self.stride_sec = stride_sec
        self.filter_low = filter_low
        self.filter_high = filter_high
        self.filter_notch = filter_notch
        self.max_amplitude_uv = max_amplitude_uv

        self.window_samples = int(window_sec * target_sfreq)
        self.stride_samples = int(stride_sec * target_sfreq)

        # Validation statistics
        self.total_trials = 0
        self.valid_trials = 0
        self.rejected_trials = 0

    def get_subject_ids(self) -> list[int]:
        """Get list of subject IDs (1-30)."""
        return list(range(1, 31))

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
            <raw_data_dir>/sub-01/ses-01/eeg/sub-01_ses-01_task-hfo_eeg.edf
        """
        ext = '.edf'
        filedir = self.raw_data_dir/f'sub-{subject_id:02d}'/'ses-01'/'eeg'
        if not filedir.is_dir():
            raise FileNotFoundError(f'Subject directory not found: {filedir}')

        file=next(filedir.glob('*.edf'),None)
        if file is None:
            raise FileNotFoundError(f'{file}not found')

        return file
        
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

    # def _read_set(self, file_path: Path):
    #     """Read raw set file."""
    #     raw = mne.io.read_raw_eeglab(file_path,preload=True)
    #     return raw

    def _read_edf(self,file_path:Path):
        "Read raw edf file."
        raw = mne.io.read_raw_edf(file_path, preload=True)
        channels_to_pick = [ch for ch in HFO_INFO.channels if ch in raw.ch_names]
    
        if channels_to_pick:
            raw.pick_channels(channels_to_pick)
        else:
            raise ValueError(f"No matching channels found. EDF channels: {raw.ch_names}")
        return raw
    def _validate_trial(self, trial_data: np.ndarray) -> bool:
        """Check if trial amplitude is within acceptable range, unit is uv."""
        max_amplitude=np.abs(trial_data).max()
        return max_amplitude <= self.max_amplitude_uv

    def _report_validation_stats(self, subject_id: int, is_resting: bool = True):
        """Report validation statistics."""
        unit = "segments" if is_resting else "trials"
        valid_pct = (self.valid_trials / self.total_trials * 100) if self.total_trials > 0 else 0
        print(f"Subject {subject_id} validation report:")
        print(f"  Amplitude threshold: {self.max_amplitude_uv} uV")
        print(f"  Total {unit}: {self.total_trials}")
        print(f"  Valid {unit}: {self.valid_trials} ({valid_pct:.1f}%)")
        print(f"  Rejected {unit}: {self.rejected_trials} ({100 - valid_pct:.1f}%)")

    def _save_dataset_info(self, stats: dict) -> None:
        """Save dataset info and processing parameters to JSON."""
        info = {
            "dataset": {
                "name": HFO_INFO.dataset_name,
                "task_type": str(HFO_INFO.task_type.value),
                "downstream_task": str(HFO_INFO.downstream_task_type.value),
                "num_labels": HFO_INFO.num_labels,
                "category_list": HFO_INFO.category_list,
                "session_list":[],
                "original_sampling_rate": HFO_INFO.sampling_rate,
                "channels": HFO_INFO.channels,
                "montage": HFO_INFO.montage,
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

        json_path = self.output_dir / "dataset_info.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)
        print(f"Saved dataset info to {json_path}")

    def _extract_trials(self,subject_id:int) -> list[dict]:
        """Extract trials from DataIntervals.tsv."""
        sub = f'sub-{subject_id:02d}'
        interval_file = (Path(self.raw_data_dir) /'derivatives' / sub / 'ses-01' / 'eeg' / 'DataIntervals.tsv')
        if not interval_file.exists():
            raise FileNotFoundError(f'Cann`t find DataIntervals.tsv:{interval_file}')
        df = pd.read_csv(interval_file, sep='\t')

        stage2label = {'N3': 1, 'N2': 2, 'Awake': 3, 'REM': 4}

        trials = []
        for _, row in df.iterrows():
            onset = row['StartInd']/self.orig_sfreq
            offset= row['EndInd']/self.orig_sfreq         
            label = stage2label.get(row['SleepStage'], 0)
            trials.append({'onset': onset,'offset':offset,'label': label})

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
            subject_id: Subject identifier (1-30)

        Returns:
            Path to output HDF5 file
        """
        if not HAS_MNE:
            raise ImportError("MNE and scipy are required for building HFO dataset")

        subdata = self._find_files(subject_id)
        print(subdata)

        all_trials = []
        ch_names = None
        trial_counter = 0
        
        session_time_offset = 0.0
                
        try:
            raw = self._read_edf(subdata)
            raw = self._preprocess(raw)


            if ch_names is None:
                ch_names = raw.ch_names
                    
            # Extract trials from this file
            trials = self._extract_trials(subject_id)
            data = raw.get_data()
                    
            # Process each trial
            for trial in trials:
                onset_sample = int(trial['onset']*self.target_sfreq)
                offset_sample = int(trial['offset']*self.target_sfreq)
                        
                trial_data = data[:, onset_sample:offset_sample]
                            
                # Calculate absolute time within session
                trial_start_time =trial['onset']/self.target_sfreq
                            
                all_trials.append({
                    'data': trial_data,
                    'label': trial['label'],
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
            dataset_name="HFO",
            task_type="resting_state",
            downstream_task_type="classification",
            rsFreq=self.target_sfreq,
            chn_name=ch_names,
            num_labels=HFO_INFO.num_labels,
            category_list=HFO_INFO.category_list,
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
                trial_data = trial["data"]

                if HFO_INFO.task_type != DatasetTaskType.RESTING_STATE:
                    self.total_trials += 1
                    if not self._validate_trial(trial_data):
                        self.rejected_trials += 1
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
                    seg_data = data[:, start_sample:end_sample]  # 形状保持 (n_chs, window_samples)
                    if HFO_INFO.task_type == DatasetTaskType.RESTING_STATE:
                        self.total_trials += 1
                        if not self._validate_trial(seg_data):
                            self.rejected_trials += 1
                            start_sample += step_samples
                            seg_id += 1
                            continue
                        self.valid_trials += 1

                    seg_start_time = trial['onset_time'] + start_sample / sfreq
                    seg_end_time   = seg_start_time + self.window_sec

                    segment_attrs = SegmentAttrs(
                        segment_id=seg_id,
                        start_time=seg_start_time,
                        end_time=seg_end_time,
                        time_length=self.window_sec,
                        label=[trial['label']],
                    )
                    writer.add_segment(trial_name, segment_attrs, seg_data)

                    start_sample += step_samples
                    seg_id += 1

        print(f"Saved {output_path} ({len(all_trials)} trials)")
        return str(output_path)

    def build_all(self, subject_ids: list[int] = None) -> list[str]:
        """Build HDF5 files for all subjects."""
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
            except Exception as exc:
                print(f"Error processing subject {subject_id}: {exc}")
                failed_subjects.append(subject_id)

        stats = {
            "total_subjects": len(subject_ids),
            "successful": len(output_paths),
            "failed": len(failed_subjects),
            "failed_subject_ids": failed_subjects,
            "total_trials_or_segments": all_total_trials,
            "valid_trials_or_segments": all_valid_trials,
            "rejected_trials_or_segments": all_rejected_trials,
        }
        self._save_dataset_info(stats)

        return output_paths


def build_hfo(
    raw_data_dir: str,
    output_dir: str = "./hdf5",
    subject_ids: list[int] = None,
    **kwargs,
) -> list[str]:
    """
    Convenience function to build HFO dataset.

    Args:
        raw_data_dir: Directory containing raw files
        output_dir: Output directory for HDF5 files
        subject_ids: List of subject IDs to process (None = all)
        **kwargs: Additional arguments for CIREBuilder

    Returns:
        List of output file paths
    """
    builder = HFOBuilder(raw_data_dir, output_dir, **kwargs)
    return builder.build_all(subject_ids)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build HFO HDF5 dataset")
    parser.add_argument("raw_data_dir",help="Directory containing raw files ")
    parser.add_argument("--output_dir", default="/mnt/dataset2/Processed_datasets/EEG_Bench", help="Output directory")
    parser.add_argument("--subjects", nargs="+", type=int, help="Subject IDs to process")
    parser.add_argument("--format", default="set", help="File format (auto-detect if not specified)")
    args = parser.parse_args()

    build_hfo(args.raw_data_dir, args.output_dir, args.subjects)