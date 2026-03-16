"""
HBN-EEG Dataset Builder (multi-task BIDS dataset).

Healthy Brain Network EEG (HBN-EEG):
- 10 task categories across passive/active paradigms
- 500 Hz sampling rate, 128-channel EGI HydroCel net (Cz reference)
- BIDS layout with per-task *_events.tsv
- Data URL: https://neuromechanist.github.io/data/hbn/
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import csv
import json
from typing import Any, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

try:
    import mne
    HAS_MNE = True
except ImportError:
    mne = None
    HAS_MNE = False

try:
    import mne_bids
    HAS_MNE_BIDS = True
except ImportError:
    mne_bids = None
    HAS_MNE_BIDS = False

try:
    from ..schema import SubjectAttrs, TrialAttrs, SegmentAttrs
    from ..hdf5_io import HDF5Writer
    from ..config import DatasetInfo, DatasetTaskType, DownstreamTaskType
except ImportError:
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
    from schema import SubjectAttrs, TrialAttrs, SegmentAttrs  # type: ignore
    from hdf5_io import HDF5Writer  # type: ignore
    from config import DatasetInfo, DatasetTaskType, DownstreamTaskType  # type: ignore


TASK_CATEGORY_LIST = [
    "thepresent",
    "symbolsearch",
    "surroundsupp",
    "seqlearning6target",
    "seqlearning8target",
    "restingstate",
    "funwithfractals",
    "diaryofawimpykid",
    "despicableme",
    "contrastchangedetection",
]

BIDS_TASK_TO_CATEGORY = {
    "ThePresent": "thepresent",
    "symbolSearch": "symbolsearch",
    "surroundSupp": "surroundsupp",
    "seqLearning6target": "seqlearning6target",
    "seqLearning8target": "seqlearning8target",
    "RestingState": "restingstate",
    "FunwithFractals": "funwithfractals",
    "DiaryOfAWimpyKid": "diaryofawimpykid",
    "DespicableMe": "despicableme",
    "contrastChangeDetection": "contrastchangedetection",
}

TASK_LABEL_MAP = {name: idx for idx, name in enumerate(TASK_CATEGORY_LIST)}

RESTING_TASK = "RestingState"
MOVIE_TASKS = {"ThePresent", "DespicableMe", "DiaryOfAWimpyKid", "FunwithFractals"}

RESTING_LABEL_MAP = {
    "instructed_toOpenEyes": "eo",
    "instructed_toCloseEyes": "ec",
}

IGNORE_EVENT_VALUES = {"break cnt", "n/a", "boundary", "9999"}

# Dataset configuration
HBN_EEG_INFO = DatasetInfo(
    dataset_name="HBN_EEG_task10",
    task_type=DatasetTaskType.MIX,
    downstream_task_type=DownstreamTaskType.CLASSIFICATION,
    num_labels=len(TASK_CATEGORY_LIST),
    category_list=TASK_CATEGORY_LIST,
    sampling_rate=500.0,
    montage="EGI_128",
    channels=[],
)

# Default amplitude threshold (uV)
DEFAULT_MAX_AMPLITUDE_UV = 800.0


class HBNEEGBuilder:
    """Builder for HBN-EEG dataset."""

    def __init__(
        self,
        raw_data_dir: str,
        output_dir: str = "./hdf5",
        target_sfreq: float = 200.0,
        window_sec: float = 1.0,
        stride_sec: float = 1.0,
        filter_low: float = 0.1,
        filter_high: float = 75.0,
        filter_notch: float = 60.0,
        max_amplitude_uv: float = DEFAULT_MAX_AMPLITUDE_UV,
    ):
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir) / "HBN_EEG"
        self.target_sfreq = target_sfreq
        self.orig_sfreq = 500.0
        self.window_sec = window_sec
        self.stride_sec = stride_sec
        self.filter_low = filter_low
        self.filter_high = filter_high
        self.filter_notch = filter_notch
        self.max_amplitude_uv = max_amplitude_uv

        self.window_samples = int(window_sec * target_sfreq)
        self.stride_samples = int(stride_sec * target_sfreq)

        self.total_trials = 0
        self.valid_trials = 0
        self.rejected_trials = 0

        self._subject_dirs = self._discover_subject_dirs()

    def _discover_subject_dirs(self) -> List[str]:
        subjects: List[str] = []
        if not self.raw_data_dir.exists():
            return subjects

        for item in sorted(self.raw_data_dir.iterdir()):
            if item.is_dir() and item.name.startswith("sub-"):
                subjects.append(item.name)
        return subjects

    def get_subject_ids(self) -> list[int]:
        """Get list of subject IDs."""
        return list(range(1, len(self._subject_dirs) + 1))

    def _subject_bids_id(self, subject_id: int) -> str:
        if subject_id < 1 or subject_id > len(self._subject_dirs):
            raise ValueError(
                f"Invalid subject_id {subject_id}; must be between 1 and {len(self._subject_dirs)}"
            )
        return self._subject_dirs[subject_id - 1]

    def _find_files(self, subject_id: int) -> list[Path]:
        """Find all EEG .set files for a subject."""
        bids_id = self._subject_bids_id(subject_id)
        eeg_dir = self.raw_data_dir / bids_id / "eeg"
        if not eeg_dir.exists():
            return []
        return sorted(eeg_dir.glob(f"{bids_id}_task-*_eeg.set"))

    def _parse_task_and_run(self, file_path: Path) -> Tuple[str, Optional[int]]:
        name = file_path.name
        task_part = name.split("_task-")[-1]
        if "_run-" in task_part:
            task_name, run_part = task_part.split("_run-", 1)
            run_str = run_part.split("_", 1)[0]
            return task_name, int(run_str)
        return task_part.split("_", 1)[0], None

    def _read_raw(self, file_path: Path):
        """Read raw EEG file and convert to MNE Raw object."""
        if not HAS_MNE:
            raise ImportError("MNE is required to build this dataset")

        bids_subject = file_path.name.split("_", 1)[0].replace("sub-", "")
        task_name, run = self._parse_task_and_run(file_path)

        if HAS_MNE_BIDS:
            import mne_bids as _mne_bids

            bids_path = _mne_bids.BIDSPath(
                root=self.raw_data_dir,
                subject=bids_subject,
                task=task_name,
                run=run,
                datatype="eeg",
                suffix="eeg",
                extension=".set",
            )
            raw = _mne_bids.read_raw_bids(bids_path, verbose=False)
        else:
            import mne as _mne

            raw = _mne.io.read_raw_eeglab(str(file_path), preload=True, verbose=False)

        return raw

    def _read_events(self, events_path: Path) -> list[dict[str, str]]:
        if not events_path.exists():
            return []
        with events_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            return [row for row in reader]

    def _preprocess(self, raw):
        """Apply preprocessing to raw data."""
        if self.filter_notch > 0:
            raw.notch_filter(freqs=self.filter_notch, verbose=False)
        raw.filter(l_freq=self.filter_low, h_freq=self.filter_high, verbose=False)
        raw.set_eeg_reference("average", verbose=False)  # Apply common average reference
        if raw.info["sfreq"] != self.target_sfreq:
            raw.resample(self.target_sfreq, verbose=False)
        return raw

    def _get_window(self, raw, start_time: float, duration: float) -> Optional[np.ndarray]:
        end_time = start_time + duration
        if end_time <= start_time:
            return None
        if end_time > raw.times[-1]:
            return None
        start_idx, end_idx = raw.time_as_index([start_time, end_time], use_rounding=True)
        if end_idx <= start_idx:
            return None
        data = raw.get_data(start=start_idx, stop=end_idx)
        return data * 1e6

    def _append_fixed_windows(
        self,
        trials: list[dict[str, Any]],
        raw,
        start_time: float,
        end_time: float,
        task_label: str,
        task_id: int,
        session_id: int,
        trial_id_start: int,
    ) -> int:
        current = start_time
        trial_id = trial_id_start
        while current + self.window_sec <= end_time:
            data = self._get_window(raw, current, self.window_sec)
            if data is not None:
                trials.append(
                    {
                        "data": data,
                        "label": task_id,
                        "task_label": task_label,
                        "trial_id": trial_id,
                        "session_id": session_id,
                        "onset_time": current,
                    }
                )
                trial_id += 1
            current += self.stride_sec
        return trial_id

    def _build_resting_trials(
        self,
        raw,
        events: list[dict[str, str]],
        task_id: int,
        session_id: int,
        trial_id_start: int,
    ) -> Tuple[list[dict[str, Any]], int]:
        trials: list[dict[str, Any]] = []
        timeline = []
        for row in events:
            value = (row.get("value") or "").strip()
            if value in RESTING_LABEL_MAP:
                try:
                    onset = float(row.get("onset", "0"))
                except ValueError:
                    continue
                timeline.append((onset, RESTING_LABEL_MAP[value]))
        timeline.sort(key=lambda item: item[0])

        trial_id = trial_id_start
        for idx, (onset, label) in enumerate(timeline):
            next_onset = timeline[idx + 1][0] if idx + 1 < len(timeline) else raw.times[-1]
            trial_id = self._append_fixed_windows(
                trials,
                raw,
                onset,
                next_onset,
                label,
                task_id,
                session_id,
                trial_id,
            )
        return trials, trial_id

    def _build_movie_trials(
        self,
        raw,
        events: list[dict[str, str]],
        task_id: int,
        task_label: str,
        session_id: int,
        trial_id_start: int,
    ) -> Tuple[list[dict[str, Any]], int]:
        trials: list[dict[str, Any]] = []
        trial_id = trial_id_start
        start_time: Optional[float] = None
        for row in events:
            value = (row.get("value") or "").strip()
            try:
                onset = float(row.get("onset", "0"))
            except ValueError:
                continue
            if value == "video_start":
                start_time = onset
            elif value == "video_stop" and start_time is not None:
                trial_id = self._append_fixed_windows(
                    trials,
                    raw,
                    start_time,
                    onset,
                    task_label,
                    task_id,
                    session_id,
                    trial_id,
                )
                start_time = None
        return trials, trial_id

    def _build_event_trials(
        self,
        raw,
        events: list[dict[str, str]],
        task_id: int,
        session_id: int,
        trial_id_start: int,
    ) -> Tuple[list[dict[str, Any]], int]:
        trials: list[dict[str, Any]] = []
        trial_id = trial_id_start
        for row in events:
            value = (row.get("value") or "").strip()
            if not value or value in IGNORE_EVENT_VALUES:
                continue
            try:
                onset = float(row.get("onset", "0"))
            except ValueError:
                continue
            data = self._get_window(raw, onset, self.window_sec)
            if data is None:
                continue
            trials.append(
                {
                    "data": data,
                    "label": task_id,
                    "task_label": value,
                    "trial_id": trial_id,
                    "session_id": session_id,
                    "onset_time": onset,
                }
            )
            trial_id += 1
        return trials, trial_id

    def _save_dataset_info(self, stats: dict) -> None:
        """Save dataset info and processing parameters to JSON."""
        info = {
            "dataset": {
                "name": HBN_EEG_INFO.dataset_name,
                "task_type": str(HBN_EEG_INFO.task_type.value),
                "downstream_task": str(HBN_EEG_INFO.downstream_task_type.value),
                "num_labels": HBN_EEG_INFO.num_labels,
                "category_list": HBN_EEG_INFO.category_list,
                "original_sampling_rate": HBN_EEG_INFO.sampling_rate,
                "channels": HBN_EEG_INFO.channels,
                "montage": HBN_EEG_INFO.montage,
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

    def _validate_trial(self, trial_data: np.ndarray) -> bool:
        return np.abs(trial_data).max() <= self.max_amplitude_uv

    def _report_validation_stats(self, subject_id: int):
        valid_pct = (self.valid_trials / self.total_trials * 100) if self.total_trials > 0 else 0
        print(f"Subject {subject_id} validation report:")
        print(f"  Amplitude threshold: {self.max_amplitude_uv} uV")
        print(f"  Total segments: {self.total_trials}")
        print(f"  Valid segments: {self.valid_trials} ({valid_pct:.1f}%)")
        print(f"  Rejected segments: {self.rejected_trials} ({100 - valid_pct:.1f}%)")

    def build_subject(self, subject_id: int) -> Tuple[str, int, int, int]:
        """Build HDF5 file for a single subject.

        Returns:
            Tuple of (output_path, total_trials, valid_trials, rejected_trials)
        """
        if not HAS_MNE:
            raise ImportError("MNE is required to build this dataset")

        self.total_trials = 0
        self.valid_trials = 0
        self.rejected_trials = 0

        files = self._find_files(subject_id)
        if not files:
            raise FileNotFoundError(f"No files found for subject {subject_id}")

        all_trials: list[dict[str, Any]] = []
        ch_names: Optional[list[str]] = None
        trial_id = 0

        for session_id, file_path in enumerate(files, 1):
            task_name, _ = self._parse_task_and_run(file_path)
            category_name = BIDS_TASK_TO_CATEGORY.get(task_name, task_name.lower())
            task_id = TASK_LABEL_MAP.get(category_name)
            if task_id is None:
                print(f"Skipping unknown task {task_name} for {file_path}")
                continue

            events_path = file_path.with_name(file_path.name.replace("_eeg.set", "_events.tsv"))
            events = self._read_events(events_path)

            raw = self._read_raw(file_path)
            raw = self._preprocess(raw)

            if ch_names is None:
                ch_names = raw.ch_names

            if task_name == RESTING_TASK:
                trials, trial_id = self._build_resting_trials(
                    raw, events, task_id, session_id, trial_id
                )
            elif task_name in MOVIE_TASKS:
                trials, trial_id = self._build_movie_trials(
                    raw,
                    events,
                    task_id,
                    category_name,
                    session_id,
                    trial_id,
                )
            else:
                trials, trial_id = self._build_event_trials(
                    raw, events, task_id, session_id, trial_id
                )

            for trial in trials:
                trial["task_name"] = task_name
            all_trials.extend(trials)

        if not all_trials:
            raise ValueError(f"No valid trials extracted for subject {subject_id}")

        bids_id = self._subject_bids_id(subject_id)
        subject_attrs = SubjectAttrs(
            subject_id=bids_id,
            dataset_name=HBN_EEG_INFO.dataset_name,
            task_type=HBN_EEG_INFO.task_type.value,
            downstream_task_type=HBN_EEG_INFO.downstream_task_type.value,
            rsFreq=self.target_sfreq,
            chn_name=ch_names or [],
            num_labels=HBN_EEG_INFO.num_labels,
            category_list=HBN_EEG_INFO.category_list,
            chn_pos=None,
            chn_ori=None,
            chn_type="EEG",
            montage=HBN_EEG_INFO.montage,
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / f"{bids_id.replace('-', '_')}.h5"

        with HDF5Writer(str(output_path), subject_attrs) as writer:
            for trial in all_trials:
                trial_data = trial["data"]
                self.total_trials += 1
                if not self._validate_trial(trial_data):
                    self.rejected_trials += 1
                    continue
                self.valid_trials += 1

                trial_attrs = TrialAttrs(
                    trial_id=trial["trial_id"],
                    session_id=trial["session_id"],
                    task_name=trial["task_name"],
                )
                trial_name = writer.add_trial(trial_attrs)

                segment_attrs = SegmentAttrs(
                    segment_id=0,
                    start_time=trial.get("onset_time", 0.0),
                    end_time=trial.get("onset_time", 0.0) + self.window_sec,
                    time_length=self.window_sec,
                    label=np.array([trial["label"]]),
                    task_label=trial["task_label"],
                )
                writer.add_segment(trial_name, segment_attrs, trial_data)

        self._report_validation_stats(subject_id)
        print(f"Saved {output_path}")
        return str(output_path), self.total_trials, self.valid_trials, self.rejected_trials

    def build_all(self, subject_ids: Optional[list[int]] = None, n_jobs: int = 1) -> list[str]:
        if subject_ids is None:
            subject_ids = self.get_subject_ids()

        output_paths = []
        failed_subjects = []
        all_total_trials = 0
        all_valid_trials = 0
        all_rejected_trials = 0

        if n_jobs == 1:
            # Sequential processing
            for subject_id in subject_ids:
                try:
                    output_path, total, valid, rejected = self.build_subject(subject_id)
                    output_paths.append(output_path)
                    all_total_trials += total
                    all_valid_trials += valid
                    all_rejected_trials += rejected
                except Exception as exc:
                    print(f"Error processing subject {subject_id}: {exc}")
                    failed_subjects.append(subject_id)
        else:
            # Parallel processing
            n_workers = n_jobs if n_jobs > 0 else None  # None = use all CPUs
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {
                    executor.submit(self.build_subject, sid): sid
                    for sid in subject_ids
                }
                for future in as_completed(futures):
                    subject_id = futures[future]
                    try:
                        output_path, total, valid, rejected = future.result()
                        output_paths.append(output_path)
                        all_total_trials += total
                        all_valid_trials += valid
                        all_rejected_trials += rejected
                        print(f"Completed subject {subject_id}")
                    except Exception as exc:
                        print(f"Error processing subject {subject_id}: {exc}")
                        failed_subjects.append(subject_id)

        stats = {
            "total_subjects": len(subject_ids),
            "successful": len(output_paths),
            "failed": len(failed_subjects),
            "failed_subject_ids": failed_subjects,
            "total_segments": all_total_trials,
            "valid_segments": all_valid_trials,
            "rejected_segments": all_rejected_trials,
        }
        self._save_dataset_info(stats)

        return output_paths


def build_hbn_eeg(
    raw_data_dir: str,
    output_dir: str = "./hdf5",
    subject_ids: Optional[list[int]] = None,
    n_jobs: int = 1,
    **kwargs,
) -> list[str]:
    """Convenience function to build HBN-EEG."""
    builder = HBNEEGBuilder(raw_data_dir, output_dir, **kwargs)
    return builder.build_all(subject_ids, n_jobs=n_jobs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build HBN-EEG HDF5 dataset")
    parser.add_argument("raw_data_dir", help="Directory containing raw files")
    parser.add_argument("--output_dir", default="../Processed_datasets/EEG_Bench", help="Output directory")
    parser.add_argument("--subjects", nargs="+", type=int, help="Subject IDs to process")
    parser.add_argument("--n_jobs", type=int, default=12, help="Number of parallel workers (default: 1, -1 for all CPUs)")
    args = parser.parse_args()

    build_hbn_eeg(args.raw_data_dir, args.output_dir, args.subjects, n_jobs=args.n_jobs)
