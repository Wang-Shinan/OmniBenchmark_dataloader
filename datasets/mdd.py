"""
MDD Dataset Builder.

MDD Dataset
- 64 subjects (healthy: 1-30;MDD :31-64)
- 256 Hz sampling rate
- 22 channels
- https://figshare.com/articles/dataset/EEG_Data_New/4244171
"""

import os
import re
import warnings
import json
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
MDD_INFO = DatasetInfo(
    dataset_name="MDD_2Class",
    task_type=DatasetTaskType.RESTING_STATE,
    downstream_task_type=DownstreamTaskType.CLASSIFICATION,
    num_labels=2,
    category_list=["Healthy Controls","Major Depressive Disorder"],
    sampling_rate=256.0,
    montage="10_20",
    channels=['Fp1','F3','C3','P3','O1',
              'F7','T3','T5','Fz','Fp2',
              'F4','C4','P4','O2','F8',
              'T4','T6','Cz','Pz','A2-A1'],
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

class MDDBuilder:
    """Builder for MDD dataset."""

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
        Initialize MDD builder.

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
        self.output_dir = Path(output_dir) / "MDD"
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

        # Validation statistics
        self.total_trials = 0
        self.valid_trials = 0
        self.rejected_trials = 0

    def get_subject_ids(self) -> list[int]:
        """Get list of subject IDs (1-64)."""
        return list(range(1, 65))

    def _find_files(self, subject_id: int) -> dict[str, list[Path]]:
        """
        仅返回单个被试的set文件。
        目录结构假设：
            <raw_data_dir>/H S1 EC.edf
            <raw_data_dir>/H S1 EO.edf
            <raw_data_dir>/H S1 TASK.edf
        """
        if subject_id<=30:
            patterns = {
                'EC': f'H S{subject_id} EC.edf',
                'EO': f'H S{subject_id} EO.edf',
                'TASK': f'H S{subject_id} TASK.edf'
            }
        else:
            patterns = {
                'EC': f'MDD S{subject_id-30} EC.edf',
                'EO': f'MDD S{subject_id-30} EO.edf',
                'TASK': f'MDD S{subject_id-30} TASK.edf'
            }

        files = {}
        for task, name in patterns.items():
            fp = self.raw_data_dir / name
            if fp.is_file():
                files[task] = [fp]      
            else:
                files[task] = []         

        return files
        
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

    def _read_edf(self, files: dict[str, list[Path]], subject_id: int):
        """
        input files = {'EC': [Path], 'EO': [Path], 'TASK': [Path]}
        output {'EC': raw_ec, 'EO': raw_eo, 'TASK': raw_task}
        """
        raws: dict[str, mne.io.Raw] = {}
        trials: list[dict] = []
        
        for task, f_list in files.items():
            if not f_list:  # empty->lost file
                continue
            file_path = f_list[0]               
            raw = mne.io.read_raw_edf(file_path, preload=True)
            if len(raw.ch_names) > 20:
                raw.pick_channels(raw.ch_names[:20])
            raw = self._preprocess(raw)
            raws[task] = raw
            trials.append({
                'task': task,
                'onset': 0.0,
                'label': self._get_label(subject_id),
            })
        return raws, trials
        
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

    # def _extract_trials(self, raw ) -> list[dict]:
    #     """Extract trials from annotations."""
    #     return trials
    def _save_dataset_info(self, stats: dict) -> None:
        """Save dataset info and processing parameters to JSON."""
        info = {
            "dataset": {
                "name": MDD_INFO.dataset_name,
                "task_type": str(MDD_INFO.task_type.value),
                "downstream_task": str(MDD_INFO.downstream_task_type.value),
                "num_labels": MDD_INFO.num_labels,
                "category_list": MDD_INFO.category_list,
                "session_list":["1:eyes closed","2:eyes opened","P300 task data"],
                "original_sampling_rate": MDD_INFO.sampling_rate,
                "channels": MDD_INFO.channels,
                "montage": MDD_INFO.montage,
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

    def _get_label(self,subject_id:int):
        label=0
        if subject_id<=30:
            label=1
        else:
            label=2
        return label;
    def _get_session_id(self,task:str):
        if task=='EC':
            session_id=1
        elif task=='EO':
            session_id=2
        else:
            session_id=3

        return session_id

    def build_subject(self, subject_id: int) -> str:
        """
        Build HDF5 file for a single subject.

        Args:
            subject_id: Subject identifier (1-64)

        Returns:
            Path to output HDF5 file
        """
        if not HAS_MNE:
            raise ImportError("MNE and scipy are required for building MDD dataset")

        files = self._find_files(subject_id)

        all_trials = []
        ch_names = None
        trial_counter = 0
        self.total_trials = 0
        self.valid_trials = 0
        self.rejected_trials = 0
        session_time_offset = 0.0
                
        try:
            raws,trials = self._read_edf(files,subject_id)

            ch_names = ['Fp1','F3','C3','P3','O1',
              'F7','T3','T5','Fz','Fp2',
              'F4','C4','P4','O2','F8',
              'T4','T6','Cz','Pz','A2-A1']

            # Process each task as one trial (EC/EO/TASK)
            for trial in trials:
                task = trial['task']
                if task not in raws:
                    continue
                raw = raws[task]
                data = raw.get_data()  # shape = (n_chs, n_times) in V

                trial_start_time = session_time_offset + trial['onset']
                all_trials.append({
                    'task':trial['task'],
                    'data': data,
                    'label': trial['label'],
                    'trial_id': trial_counter,
                    'onset_time': 0,
                })
                trial_counter += 1

                # Update session time offset per file
                file_duration = data.shape[1] / self.target_sfreq
                session_time_offset += file_duration
                    
        except Exception as e:
            print(f"Error processing {files}: {e}")
            import traceback
            traceback.print_exc()

        if not all_trials:
            raise ValueError(f"No valid trials extracted for subject {subject_id}")

        # Create subject attributes
        subject_attrs = SubjectAttrs(
            subject_id=subject_id,
            dataset_name="MDD",
            task_type="resting_state",
            downstream_task_type="classification",
            rsFreq=self.target_sfreq,
            chn_name=ch_names,
            num_labels=MDD_INFO.num_labels,
            category_list=MDD_INFO.category_list,
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
                trial_data_uv=trial_data*1e6
                if MDD_INFO.task_type != DatasetTaskType.RESTING_STATE:
                    self.total_trials += 1
                    if not self._validate_trial(trial_data_uv):
                        self.rejected_trials += 1
                        continue
                    self.valid_trials += 1

                trial_attrs = TrialAttrs(
                    trial_id=trial['trial_id'],
                    session_id=self._get_session_id(trial['task']),
                )
                trial_name = writer.add_trial(trial_attrs)

                # sampling point
                sfreq = self.target_sfreq
                window_samples = int(self.window_sec * sfreq)
                step_samples   = window_samples
                
                data = trial['data']              # shape = (n_chs, n_times)
                n_chs, total_samples = data.shape

                start_sample = 0
                seg_id = 0

                while start_sample + window_samples <= total_samples:
                    end_sample = start_sample + window_samples
                    seg_data = data[:, start_sample:end_sample]  # shape (n_chs, window_samples)
                    # Convert from V back to μV for export/display
                    seg_data_uv =seg_data*1e6
                    
                    if MDD_INFO.task_type == DatasetTaskType.RESTING_STATE:
                        self.total_trials += 1
                        if not self._validate_trial(seg_data_uv):
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
                        label=np.array([trial['label']]),
                    )
                    writer.add_segment(trial_name, segment_attrs, seg_data_uv)

                    start_sample += step_samples
                    seg_id += 1
        is_resting = True            
        self._report_validation_stats(subject_id, is_resting)
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


def build_mdd(
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
    builder = MDDBuilder(raw_data_dir, output_dir, **kwargs)
    return builder.build_all(subject_ids)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build MDD HDF5 dataset")
    parser.add_argument("raw_data_dir",help="Directory containing raw files ")
    parser.add_argument("--output_dir", default="/mnt/dataset2/Processed_datasets/EEG_Bench", help="Output directory")
    parser.add_argument("--subjects", nargs="+", type=int, help="Subject IDs to process")
    parser.add_argument("--format", default="set", help="File format (auto-detect if not specified)")
    args = parser.parse_args()

    build_mdd(args.raw_data_dir, args.output_dir, args.subjects)