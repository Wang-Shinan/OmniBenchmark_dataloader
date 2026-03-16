"""
YOUR_DATASET_xClass Dataset Builder.

YOUR_DATASET: Brief description.
- N subjects
- M sessions per subject
- K trials per session
- C classes: class1, class2, ...
- link to dataset: http://example.com/your_dataset
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
# ## TODO: update dataset metadata, channels, and labels for your dataset
YOUR_DATASET_INFO = DatasetInfo(
    dataset_name="YOUR_DATASET_xClass",
    task_type=DatasetTaskType.OTHER,
    downstream_task_type=DownstreamTaskType.CLASSIFICATION,
    num_labels=0,
    category_list=[],
    sampling_rate=200.0,
    montage="10_20",
    channels=[],
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
        window_sec: float = 2.0,
        stride_sec: float = 2.0,
        filter_low: float = 0.1,
        filter_high: float = 75.0,
        filter_notch: float = 50.0,
        max_amplitude_uv: float = DEFAULT_MAX_AMPLITUDE_UV,
    ):
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir) / "YOUR_DATASET" ## TODO update dataset folder name
        self.target_sfreq = target_sfreq
        # ## TODO: set original sampling rate
        self.orig_sfreq = 200.0
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

    # ## TODO: update subject id range
    def get_subject_ids(self) -> list[int]:
        """Get list of subject IDs."""
        return list(range(1, 1))

    # ## TODO: implement file discovery
    def _find_files(self, subject_id: int):
        """Find all files for a subject."""
        raise NotImplementedError

    # ## TODO: implement raw file loading, need to check file format, first convert to MNE Raw object with volts unit
    def _read_raw(self, file_path: Path):
        """Read raw EEG file and convert to MNE Raw object."""
        raise NotImplementedError

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
            
        ## TODO: add channel reference or average reference if needed
        

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
                "name": YOUR_DATASET_INFO.dataset_name,
                "task_type": str(YOUR_DATASET_INFO.task_type.value),
                "downstream_task": str(YOUR_DATASET_INFO.downstream_task_type.value),
                "num_labels": YOUR_DATASET_INFO.num_labels,
                "category_list": YOUR_DATASET_INFO.category_list,
                "original_sampling_rate": YOUR_DATASET_INFO.sampling_rate,
                "channels": YOUR_DATASET_INFO.channels,
                "montage": YOUR_DATASET_INFO.montage,
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

        # ## TODO: adjust file discovery logic and session layout
        files = self._find_files(subject_id)
        if not files:
            raise FileNotFoundError(f"No files found for subject {subject_id}")

        all_trials = []
        ch_names = None

        # ## TODO: implement trial extraction from raw files
        for session_id, file_path in enumerate(files, 1):
            print(f"Reading {file_path}")
            raw = self._read_raw(file_path)
            raw = self._preprocess(raw)

            if ch_names is None:
                ch_names = raw.ch_names

            # Example placeholder: replace with your trial extraction, here let's change voltage unit back to uV, for further processing
            data_uv = raw.get_data() * 1e6
            all_trials.append({
                "data": data_uv,
                "label": 0,
                "trial_id": session_id - 1,
                "session_id": session_id,
                "onset_time": 0.0,
            })

        if not all_trials:
            raise ValueError(f"No valid trials extracted for subject {subject_id}")

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

        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / f"sub_{subject_id}.h5"

        with HDF5Writer(str(output_path), subject_attrs) as writer:
            for trial in all_trials:
                trial_data = trial["data"]

                # For non-resting-state tasks, validate at trial level
                if YOUR_DATASET_INFO.task_type != DatasetTaskType.RESTING_STATE:
                    self.total_trials += 1
                    if not self._validate_trial(trial_data):
                        self.rejected_trials += 1
                        continue
                    self.valid_trials += 1

                trial_attrs = TrialAttrs(
                    trial_id=trial["trial_id"],
                    session_id=trial["session_id"],
                )
                trial_name = writer.add_trial(trial_attrs)

                n_samples = trial_data.shape[1]
                for i_slice, start in enumerate(
                    range(0, n_samples - self.window_samples + 1, self.stride_samples)
                ):
                    end = start + self.window_samples
                    slice_data = trial_data[:, start:end]

                    if YOUR_DATASET_INFO.task_type == DatasetTaskType.RESTING_STATE:
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

        is_resting = YOUR_DATASET_INFO.task_type == DatasetTaskType.RESTING_STATE
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
