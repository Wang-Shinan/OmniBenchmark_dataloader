# -*- coding: utf-8 -*-
"""
Go-Nogo Dataset Builder (BIDS / EEGLAB .set)

✅ Team defaults (task requirements v1.1)
- target_sfreq=200.0
- window_sec=1.0, stride_sec=1.0  (event-locked; stride kept for interface consistency)
- filter_low=0.1, filter_high=75.0, filter_notch=50.0
- Output: one H5 per subject under <output_dir>/GoNogo/

Strategy (instant / event-locked):
- Trial  = each run file (*_eeg.set)
- Segment= each stimulus event window [onset, onset + window_sec]
- Label  = 1 if trial_type contains 'target', 0 if contains 'distractor'
- task_label keeps original fine-grained trial_type (easy/difficult/animal/nonanimal, etc.)

This version includes:
- robust subject discovery + robust run-file discovery (task-gonogo OR any *_eeg.set fallback)
- directory creation for dataset_info.json
- clear debug logs + traceback on failure
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import csv
import json
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import mne
    HAS_MNE = True
except ImportError:
    mne = None
    HAS_MNE = False

# Local imports (compatible with both package and script runs)
try:
    from ..schema import SubjectAttrs, TrialAttrs, SegmentAttrs
    from ..hdf5_io import HDF5Writer
    from ..config import DatasetInfo, DatasetTaskType, DownstreamTaskType
except Exception:
    # If you keep this file under benchmark_dataloader/datasets/, change these imports accordingly.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from schema import SubjectAttrs, TrialAttrs, SegmentAttrs
    from hdf5_io import HDF5Writer
    from config import DatasetInfo, DatasetTaskType, DownstreamTaskType

GO_NOGO_INFO = DatasetInfo(
    dataset_name="GoNogo_2Class",
    task_type=DatasetTaskType.COGNITIVE,
    downstream_task_type=DownstreamTaskType.CLASSIFICATION,
    num_labels=2,
    category_list=["distractor", "target"],
    sampling_rate=0.0,  # unknown here; will be read from raw
    montage="unknown",
    channels=[],
)

DEFAULT_MAX_AMPLITUDE_UV = 600.0

# Event values we do NOT want to treat as stimulus labels
IGNORE_EVENT_VALUES = {
    "", "n/a", "na", "boundary", "break", "break cnt", "9999",
    "correct", "incorrect"
}


def _safe_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None

### MOD-FIX: robust Go/NoGo label inference
def _infer_label_from_trial_type(trial_type: str) -> Optional[int]:
    """
    Map stimulus category to 2-class label.
    Returns:
        1 for target, 0 for distractor, None for non-stimulus/unrecognized.
    """
    tt = (trial_type or "").strip().lower()
    if not tt or tt in IGNORE_EVENT_VALUES:
        return None
    if "target" in tt:
        return 1
    if "distractor" in tt:
        return 0
    return None


def _pick_eeg_only(raw):
    """Keep EEG channels only (drop EOG/Stim/Misc) if channel types are available."""
    try:
        raw.pick_types(eeg=True, eog=False, ecg=False, emg=False, stim=False, misc=False)
    except Exception:
        pass
    return raw


class GoNogoBuilder:
    def __init__(
        self,
        raw_data_dir: str,
        output_dir: str = "./Processed_datasets/EEG_Bench",
        # Hard defaults (team-wide)
        target_sfreq: float = 200.0,
        window_sec: float = 1.0,
        stride_sec: float = 1.0,   # not used for event-locked, kept for interface consistency
        filter_low: float = 0.1,
        filter_high: float = 75.0,
        filter_notch: float = 50.0,
        # QC
        max_amplitude_uv: float = DEFAULT_MAX_AMPLITUDE_UV,
        # Debug
        verbose: bool = True,
    ):
        if not HAS_MNE:
            raise ImportError("mne is required to build Go-Nogo dataset")

        self.raw_data_dir = Path(raw_data_dir)
        self.output_root = Path(output_dir)           # user-provided root
        self.output_dir = self.output_root / "GoNogo" # dataset subdir (consistent with other builders)
        self.target_sfreq = float(target_sfreq)
        self.window_sec = float(window_sec)
        self.stride_sec = float(stride_sec)
        self.filter_low = float(filter_low)
        self.filter_high = float(filter_high)
        self.filter_notch = float(filter_notch)
        self.max_amplitude_uv = float(max_amplitude_uv)
        self.verbose = bool(verbose)

        # stats
        self.total_segments = 0
        self.valid_segments = 0
        self.rejected_segments = 0
        self.label_counts = {0: 0, 1: 0}

        self._subjects = self._discover_subjects()

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def _discover_subjects(self) -> List[str]:
        """
        Discover BIDS subjects at <raw_data_dir>/sub-*
        """
        if not self.raw_data_dir.exists():
            return []
        subs = sorted([p.name for p in self.raw_data_dir.iterdir() if p.is_dir() and p.name.startswith("sub-")])
        return subs

    def get_subject_ids(self) -> List[str]:
        return list(self._subjects)

    def _subject_bids_id(self, subject_id: Union[str, int]) -> str:
        """
        Accept:
          - 'sub-001'
          - '001'
          - 1  (mapped by sorted order if possible, else padded)
        """
        if isinstance(subject_id, str):
            s = subject_id.strip()
            if s.startswith("sub-"):
                return s
            if re.fullmatch(r"\d+", s):
                cand = f"sub-{int(s):03d}"
                if cand in self._subjects:
                    return cand
                cand2 = f"sub-{s}"
                if cand2 in self._subjects:
                    return cand2
                return cand
            return s

        # int case
        if 1 <= int(subject_id) <= len(self._subjects):
            return self._subjects[int(subject_id) - 1]
        return f"sub-{int(subject_id):03d}"

    def _find_run_files(self, bids_sub: str) -> List[Path]:
        """
        Find run EEG .set files under sub-xxx/ses-xx/eeg/.
        First try strict Go-Nogo task match; if nothing, fallback to all *_eeg.set.
        """
        sub_dir = self.raw_data_dir / bids_sub
        if not sub_dir.exists():
            return []

        strict = sorted(sub_dir.glob("ses-*/eeg/*_task-gonogo*_eeg.set"))
        if strict:
            return strict

        # Fallback (keeps you unblocked if task name differs slightly)
        fallback = sorted(sub_dir.glob("ses-*/eeg/*_eeg.set"))
        return fallback
### MOD-FIX: robust events.tsv reader (UTF-8 BOM + column normalization)
    def _read_events_tsv(self, events_path: Path) -> List[Dict[str, str]]:
        """
        Read BIDS events.tsv robustly.
        Handles UTF-8 BOM (common in some BIDS exports) and strips header/value whitespace.
        """
        if not events_path.exists():
            return []
        # utf-8-sig will strip BOM if present
        with events_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f, delimiter="	")
            rows: List[Dict[str, str]] = []
            for row in reader:
                # normalize keys (strip whitespace / BOM artifacts) and values
                norm = {}
                for k, v in row.items():
                    kk = (k or "").strip()
                    vv = v.strip() if isinstance(v, str) else v
                    norm[kk] = vv
                rows.append(norm)
            return rows

    def _preprocess(self, raw):
        raw = _pick_eeg_only(raw)

        if self.filter_notch and self.filter_notch > 0:
            raw.notch_filter(freqs=self.filter_notch, verbose=False)
        raw.filter(l_freq=self.filter_low, h_freq=self.filter_high, verbose=False)

        if raw.info["sfreq"] != self.target_sfreq:
            raw.resample(self.target_sfreq, verbose=False)

        return raw
### MOD-FIX: correct unit detection (Volt → µV) to avoid false rejection
    def _get_window_uv(self, raw, onset_sec: float) -> Optional[np.ndarray]:
        """Return (C, T) in uV for [onset, onset + window_sec]."""
        start = onset_sec
        end = onset_sec + self.window_sec
        if end <= start:
            return None
        if end > raw.times[-1]:
            return None

        start_idx, end_idx = raw.time_as_index([start, end], use_rounding=True)
        if end_idx <= start_idx:
            return None

        data = raw.get_data(start=start_idx, stop=end_idx)  # (C, T) in whatever unit EEGLAB stored
        # Auto-detect unit:
        # - If values are ~1e-5 to 1e-4, it's likely Volts -> convert to uV
        # - If values are already ~10-100, it's likely uV -> keep
        max_abs = float(np.nanmax(np.abs(data))) if data.size else 0.0
        if max_abs < 1e-3:  # treat as Volts
            data_uv = data * 1e6
        else:  # treat as already uV
            data_uv = data
        return data_uv

    def _validate_segment(self, seg_uv: np.ndarray) -> bool:
        if seg_uv is None:
            return False
        if not np.isfinite(seg_uv).all():
            return False
        if np.abs(seg_uv).max() > self.max_amplitude_uv:
            return False
        return True

    def _parse_session_id_from_path(self, file_path: Path) -> int:
        # ses-01 -> 1, ses-02 -> 2, else 0
        m = re.search(r"(ses-\d+)", str(file_path))
        if not m:
            return 0
        ses = m.group(1)
        mm = re.search(r"ses-(\d+)", ses)
        return int(mm.group(1)) if mm else 0
### MOD-FIX: event-driven segment creation (prevents valid=0 bug)
    def build_subject(self, subject_id: Union[str, int]) -> str:
        """
        Build a single subject HDF5 file.
        Returns: output_path (str)
        """
        bids_sub = self._subject_bids_id(subject_id)
        run_files = self._find_run_files(bids_sub)

        if not run_files:
            raise FileNotFoundError(
                f"No EEG run files found for {bids_sub}. "
                f"Expected under: {self.raw_data_dir / bids_sub / 'ses-*/eeg/'}"
            )

        # Per-subject stats
        sub_total = 0
        sub_valid = 0
        sub_reject = 0
        sub_label_counts = {0: 0, 1: 0}

        ch_names: Optional[List[str]] = None

        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / f"{bids_sub.replace('-', '_')}.h5"

        # trial_id increments across runs
        trial_id = 0

        # Create writer AFTER ensuring output_dir exists
        with HDF5Writer(
            str(output_path),
            subject_attrs=SubjectAttrs(
                subject_id=bids_sub,
                dataset_name=GO_NOGO_INFO.dataset_name,
                task_type=GO_NOGO_INFO.task_type.value,
                downstream_task_type=GO_NOGO_INFO.downstream_task_type.value,
                rsFreq=self.target_sfreq,
                chn_name=[],  # patched later once known
                num_labels=GO_NOGO_INFO.num_labels,
                category_list=GO_NOGO_INFO.category_list,
                chn_pos=None,
                chn_ori=None,
                chn_type="EEG",
                montage=GO_NOGO_INFO.montage,
            ),
        ) as writer:

            for run_fp in run_files:
                events_fp = run_fp.with_name(run_fp.name.replace("_eeg.set", "_events.tsv"))
                events = self._read_events_tsv(events_fp)

                self._log(f"[GoNogo] {bids_sub}: run={run_fp.name} events={events_fp.name} n_events={len(events)}")

                raw = mne.io.read_raw_eeglab(str(run_fp), preload=True, verbose=False)
                raw = self._preprocess(raw)

                if ch_names is None:
                    ch_names = list(raw.ch_names)

                session_id = self._parse_session_id_from_path(run_fp)
                trial_attrs = TrialAttrs(trial_id=trial_id, session_id=session_id, task_name="gonogo")
                trial_name = writer.add_trial(trial_attrs)

                seg_id = 0
                for row in events:
                    onset = _safe_float(row.get("onset", None))
                    if onset is None:
                        continue

                    # robust column fallback
                    # In GoNogo BIDS events.tsv, column `trial_type` is usually the event category (e.g., "stimulus"/"response"),
                    # while column `value` stores the actual stimulus label (e.g., "animal_target").
                    trial_type = (row.get("trial_type") or "").strip().lower()
                    value = (row.get("value") or row.get("stim_type") or "").strip()

                    # Keep only stimulus events; skip response/correctness markers
                    if trial_type and trial_type not in ("stimulus", "stim", "cue", "go", "nogo"):
                        # For some datasets, trial_type itself may carry target/distractor
                        label = _infer_label_from_trial_type(trial_type)
                        task_label = trial_type
                    else:
                        label = _infer_label_from_trial_type(value)
                        task_label = value

                    if label is None:
                        continue

                    seg_uv = self._get_window_uv(raw, onset)
                    sub_total += 1

                    if seg_uv is None or not self._validate_segment(seg_uv):
                        sub_reject += 1
                        seg_id += 1
                        continue

                    sub_valid += 1
                    sub_label_counts[label] += 1

                    segment_attrs = SegmentAttrs(
                        segment_id=seg_id,
                        start_time=float(onset),
                        end_time=float(onset) + self.window_sec,
                        time_length=self.window_sec,
                        label=np.array([label], dtype=np.int64),
                        task_label=task_label,
                    )
                    writer.add_segment(trial_name, segment_attrs, seg_uv)
                    seg_id += 1

                trial_id += 1

        # Patch root attrs chn_name after writing (keeps writer API unchanged)
        try:
            import h5py
            with h5py.File(str(output_path), "a") as f:
                if ch_names is not None and len(ch_names) > 0:
                    f.attrs["chn_name"] = ch_names
                f.attrs["rsFreq"] = self.target_sfreq
        except Exception as e:
            self._log(f"[GoNogo] WARN: failed to patch root attrs for {output_path}: {e}")

        # Update global stats
        self.total_segments += sub_total
        self.valid_segments += sub_valid
        self.rejected_segments += sub_reject
        for k in (0, 1):
            self.label_counts[k] += sub_label_counts[k]

        self._log(
            f"[GoNogo] {bids_sub}: total={sub_total}, valid={sub_valid}, rejected={sub_reject}, "
            f"label_counts={sub_label_counts} -> {output_path}"
        )
        return str(output_path)

    def _save_dataset_info(self, stats: Dict[str, Any]) -> None:
        """Save dataset info and processing parameters to JSON."""
        # ✅ Ensure directory exists (prevents FileNotFoundError)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        info = {
            "dataset": {
                "name": GO_NOGO_INFO.dataset_name,
                "task_type": GO_NOGO_INFO.task_type.value,
                "downstream_task": GO_NOGO_INFO.downstream_task_type.value,
                "num_labels": GO_NOGO_INFO.num_labels,
                "category_list": GO_NOGO_INFO.category_list,
                "format": "BIDS / EEGLAB .set",
            },
            "strategy": {
                "trial_definition": "each run file (*_eeg.set)",
                "segment_definition": f"event-locked window [onset, onset+{self.window_sec}s] per stimulus event",
                "label_rule": "trial_type contains 'target'->1; contains 'distractor'->0; ignore others",
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
            "segment_policy": "event_locked",
            "segment_parameters": {
            "alignment": "stimulus_onset",
                "window_sec": self.window_sec,
                "stride_sec": None,   # event-locked 通常一事件一段
                "notes": "Go–Nogo: keep only trial_type==stimulus; label from value contains target/distractor"
            },
            "statistics": stats,
            "generated_at": datetime.now().isoformat(),
        }

        json_path = self.output_dir / "dataset_info.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        self._log(f"[GoNogo] Saved dataset info -> {json_path}")

    def build_all(self, subject_ids: Optional[List[Union[str, int]]] = None) -> List[str]:
        if subject_ids is None:
            subject_ids = self.get_subject_ids()

        self._log(f"[GoNogo] Discovered subjects: {len(self._subjects)} (show first 5) -> {self._subjects[:5]}")
        self._log(f"[GoNogo] Building subjects: {len(subject_ids)}")

        output_paths: List[str] = []
        failed: List[str] = []

        # reset global stats
        self.total_segments = 0
        self.valid_segments = 0
        self.rejected_segments = 0
        self.label_counts = {0: 0, 1: 0}

        for sid in subject_ids:
            bids_sub = self._subject_bids_id(sid)
            try:
                out = self.build_subject(bids_sub)
                output_paths.append(out)
            except Exception as e:
                import traceback
                print(f"[GoNogo] Error processing {bids_sub}: {e}")
                traceback.print_exc()
                failed.append(bids_sub)

        stats = {
            "total_subjects": len(subject_ids),
            "successful": len(output_paths),
            "failed": len(failed),
            "failed_subject_ids": failed,
            "total_segments": int(self.total_segments),
            "valid_segments": int(self.valid_segments),
            "rejected_segments": int(self.rejected_segments),
            "label_counts": {
                "distractor(0)": int(self.label_counts[0]),
                "target(1)": int(self.label_counts[1]),
            },
        }
        self._save_dataset_info(stats)

        if len(output_paths) == 0:
            self._log("[GoNogo] WARNING: No subjects succeeded. See traceback above for the real cause.")

        return output_paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build Go-Nogo HDF5 dataset (event-locked, 1s windows).")
    parser.add_argument("raw_data_dir", help="Path to BIDS root (e.g., ds002680-download)")
    parser.add_argument("--output_dir", default="./Processed_datasets/EEG_Bench", help="Output root dir")
    parser.add_argument("--subjects", nargs="+", help="Subject IDs to process (e.g., sub-001 sub-002 or 1 2)", default=None)
    parser.add_argument("--quiet", action="store_true", help="Disable verbose logs")

    args = parser.parse_args()

    subj_list: Optional[List[Union[str, int]]] = None
    if args.subjects:
        subj_list = args.subjects  # keep as str; builder can parse ints embedded in strings too

    builder = GoNogoBuilder(
        raw_data_dir=args.raw_data_dir,
        output_dir=args.output_dir,
        verbose=(not args.quiet),
        # Hard defaults already aligned with team requirements
    )
    builder.build_all(subj_list)
