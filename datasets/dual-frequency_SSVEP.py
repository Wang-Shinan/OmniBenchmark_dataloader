"""
dual-frequency_SSVEP Dataset Builder.

YOUR_DATASET: Brief description.
- 13 subjects for 1 target and 14 subject for 40 targets
- 80 trials per subject
- 8 classes for 1 target and 40 classes for 40 targets
- link to dataset: https://bci.med.tsinghua.edu.cn/
"""

from pathlib import Path
from datetime import datetime
import json
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


# Dataset configuration
SSVEP_INFO = DatasetInfo(
    dataset_name="dual_frequency_SSVEP",
    task_type=DatasetTaskType.COGNITIVE,
    downstream_task_type=DownstreamTaskType.CLASSIFICATION,
    num_labels=8,                         #8 for 1 target and 40 for 40 target
    category_list=["check the file code of 8Target.xlsx"],   # different files 
    sampling_rate=1000.0,
    montage="10_10",
    channels=['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 
              'F7', 'F5', 'F3', 'F1', 'FZ', 
              'F2', 'F4', 'F6', 'F8', 'FT7', 
              'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 
              'FC4', 'FC6', 'FT8', 'T7', 'C5', 
              'C3', 'C1', 'Cz', 'C2', 'C4', 
              'C6', 'T8', 'M1', 'TP7', 'CP5', 
              'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 
              'CP6', 'TP8', 'M2', 'P7', 'P5', 
              'P3', 'P1', 'Pz', 'P2', 'P4', 
              'P6', 'P8', 'PO7', 'PO5', 'PO3', 
              'POz', 'PO4', 'PO6', 'PO8', 'CB1', 
              'O1', 'Oz', 'O2', 'CB2'],
)

# ## TODO: define label mapping or metadata if needed
YOUR_DATASET_LABELS = {}

# ## TODO: list reference channels to remove, empty list if none
REMOVE_CHANNELS = []

# Default amplitude threshold (uV), check the final report after building dataset, if rejected segments are too many, let's DISCUSS and adjust this value
DEFAULT_MAX_AMPLITUDE_UV = 600.0


def detect_unit_and_convert_to_volts(data: np.ndarray) -> tuple[np.ndarray, str]:
    """Auto-detect data unit and convert to Volts for MNE."""
    max_amp = np.abs(data).max()
    if max_amp > 1e-3:  # likely uV
        return data / 1e6, "uV"

    return data, "V"


class YourDatasetBuilder:
    """Builder for YOUR_DATASET."""

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
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir) / "dual-frequency_SSVEP_1_target"   # change path for different targets
        self.target_sfreq = target_sfreq
        # ## TODO: set original sampling rate
        self.orig_sfreq = 1000.0
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
        """Get list of subject IDs."""
        return list(range(1, 14))   #13 subjects for 1 target and 14 subjects for 40 targets

    def _find_files(self, subject_id: int):
        """Find all files for a subject."""
        file = Path(self.raw_data_dir) / f'Subject {subject_id:02d}.cnt'
        return file

    def _read_raw(self, file_path: Path):
        """Read raw EEG file and convert to MNE Raw object."""
        raw = mne.io.read_raw_cnt(file_path, preload=True)
        return raw

    def _extract_trials(self,raw):
        """
        annotation.onset is the start of trial 
        label annotation.description。
        """
        fs = raw.info['sfreq']
        win_len_s = 5.0
        win_sample = int(win_len_s * fs)

        df_ann = raw.annotations.to_data_frame(time_format= None)      # onset, duration, description
        df_ann = df_ann.rename(columns={'onset': 'time_sec'})
        trials = []
        for idx, row in df_ann.iterrows():
            onset_sec = row['time_sec']
            onset_samp = int(onset_sec * fs)
            offset_samp = onset_samp + win_sample
            if(int(row['description'])==253 or int(row['description'])==254):
                continue
            if offset_samp > raw.n_times:
                continue

            trial_data = raw.get_data(start=onset_samp, stop=offset_samp)

            trials.append({
                'data': trial_data,
                'label': int(row['description']),      
                'trial_id': idx,
                'onset_time': onset_sec,
                'offset_time': onset_sec + win_len_s,
            })

        return trials

    def _validate_trial(self, trial_data: np.ndarray) -> bool:
        """Check if trial amplitude is within acceptable range."""
        return np.abs(trial_data).max() <= self.max_amplitude_uv

    def _report_validation_stats(self, subject_id: int, is_resting: bool = True):
        """Report validation statistics."""
        unit = "segments" if is_resting else "trials"
        valid_pct = (self.valid_trials / self.total_trials * 100) if self.total_trials > 0 else 0
        print(f"Subject {subject_id} validation report:")
        print(f"  Amplitude threshold: {self.max_amplitude_uv} uV")
        print(f"  Total {unit}: {self.total_trials}")
        print(f"  Valid {unit}: {self.valid_trials} ({valid_pct:.1f}%)")
        print(f"  Rejected {unit}: {self.rejected_trials} ({100 - valid_pct:.1f}%)")

    def _preprocess(self, raw):
        """Apply preprocessing to raw data."""
        # Drop target remove channels if needed, check with the dataset documentation
        if REMOVE_CHANNELS:
            raw.drop_channels(REMOVE_CHANNELS)
        

        # Notch filter
        if self.filter_notch > 0:
            raw.notch_filter(freqs=self.filter_notch, verbose=False)

        # Bandpass filter
        raw.filter(l_freq=self.filter_low, h_freq=self.filter_high, verbose=False)

        # Resample if needed
        if raw.info["sfreq"] != self.target_sfreq:
            raw.resample(self.target_sfreq, verbose=False)

        return raw

    def _save_dataset_info(self, stats: dict) -> None:
        """Save dataset info and processing parameters to JSON."""
        info = {
            "dataset": {
                "name": SSVEP_INFO.dataset_name,
                "task_type": str(SSVEP_INFO.task_type.value),
                "downstream_task": str(SSVEP_INFO.downstream_task_type.value),
                "num_labels": SSVEP_INFO.num_labels,
                "category_list": SSVEP_INFO.category_list,
                "original_sampling_rate": SSVEP_INFO.sampling_rate,
                "channels": SSVEP_INFO.channels,
                "montage": SSVEP_INFO.montage,
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

    def build_subject(self, subject_id: int) -> str:
        """Build HDF5 file for a single subject."""
        if not HAS_MNE:
            raise ImportError("MNE is required to build this dataset")

        self.total_trials = 0
        self.valid_trials = 0
        self.rejected_trials = 0

        files = self._find_files(subject_id)
        if not files:
            raise FileNotFoundError(f"No files found for subject {subject_id}")

        all_trials = []
        ch_names = SSVEP_INFO.channels

        print(f"Reading {files}")
        raw = self._read_raw(files)
        raw = self._preprocess(raw)

        if ch_names is None:
            ch_names = raw.ch_names

        # Example placeholder: replace with your trial extraction, here let's change voltage unit back to uV, for further processing
        data_uv = raw.get_data() * 1e6
        all_trials=self._extract_trials(raw)
        print(f"Subject{subject_id} has {len(all_trials)} trials")

        if not all_trials:
            raise ValueError(f"No valid trials extracted for subject {subject_id}")

        subject_attrs = SubjectAttrs(
            subject_id=subject_id,
            dataset_name=SSVEP_INFO.dataset_name,
            task_type=SSVEP_INFO.task_type.value,
            downstream_task_type=SSVEP_INFO.downstream_task_type.value,
            rsFreq=self.target_sfreq,
            chn_name=ch_names,
            num_labels=SSVEP_INFO.num_labels,
            category_list=SSVEP_INFO.category_list,
            chn_pos=None,
            chn_ori=None,
            chn_type="EEG",
            montage=SSVEP_INFO.montage,
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / f"sub_{subject_id}.h5"

        with HDF5Writer(str(output_path), subject_attrs) as writer:
            for trial in all_trials:
                trial_data = trial["data"]
                trial_data= trial_data*1e6

                # For non-resting-state tasks, validate at trial level
                if SSVEP_INFO.task_type != DatasetTaskType.RESTING_STATE:
                    self.total_trials += 1
                    if not self._validate_trial(trial_data):
                        self.rejected_trials += 1
                        continue
                    self.valid_trials += 1

                trial_attrs = TrialAttrs(
                    trial_id=trial["trial_id"],
                    session_id=0,
                )
                trial_name = writer.add_trial(trial_attrs)

                n_samples = trial_data.shape[1]
                for i_slice, start in enumerate(
                    range(0, n_samples - self.window_samples + 1, self.stride_samples)
                ):
                    end = start + self.window_samples
                    slice_data = trial_data[:, start:end]

                    if SSVEP_INFO.task_type == DatasetTaskType.RESTING_STATE:
                        self.total_trials += 1
                        if not self._validate_trial(slice_data):
                            self.rejected_trials += 1
                            continue
                        self.valid_trials += 1

                    segment_attrs = SegmentAttrs(
                        segment_id=i_slice,
                        start_time=trial.get("onset_time", 0.0) + start / self.target_sfreq,
                        end_time=trial.get("onset_time", 0.0) + end / self.target_sfreq,
                        time_length=self.window_sec,
                        label=np.array([trial["label"]]),
                    )
                    writer.add_segment(trial_name, segment_attrs, slice_data)

        is_resting = SSVEP_INFO.task_type == DatasetTaskType.RESTING_STATE
        self._report_validation_stats(subject_id, is_resting)
        print(f"Saved {output_path}")
        return str(output_path)

    def build_all(self, subject_ids: list[int] | None = None) -> list[str]:
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
            if self.rejected_trials==self.total_trials:
                failed_subjects.append(subject_id)

        stats = {
            "total_subjects": len(subject_ids),
            "successful": len(output_paths)-len(failed_subjects),
            "failed": len(failed_subjects),
            "failed_subject_ids": failed_subjects,
            "total_trials_or_segments": all_total_trials,
            "valid_trials_or_segments": all_valid_trials,
            "rejected_trials_or_segments": all_rejected_trials,
        }
        self._save_dataset_info(stats)

        return output_paths


def build_your_dataset(
    raw_data_dir: str,
    output_dir: str = "./hdf5",
    subject_ids: list[int] | None = None,
    **kwargs,
) -> list[str]:
    """Convenience function to build YOUR_DATASET."""
    builder = YourDatasetBuilder(raw_data_dir, output_dir, **kwargs)
    return builder.build_all(subject_ids)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build YOUR_DATASET HDF5 dataset")
    parser.add_argument("raw_data_dir", help="Directory containing raw files")
    parser.add_argument("--output_dir", default="/mnt/dataset2/Processed_datasets/EEG_Bench", help="Output directory")
    parser.add_argument("--subjects", nargs="+", type=int, help="Subject IDs to process")
    args = parser.parse_args()

    build_your_dataset(args.raw_data_dir, args.output_dir, args.subjects)
