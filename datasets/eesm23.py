"""
EESM23 Dataset Builder (OpenNeuro ds005178) — Ear-EEG Sleep Staging
==================================================================

Raw dataset (BIDS-like, per EESM23 conversion scripts):
- root/
  - sub-XXX/
    - ses-YYY/
      - eeg/
        - sub-XXX_ses-YYY_task-sleep_acq-earEEG_eeg.set
        - sub-XXX_ses-YYY_task-sleep_acq-earEEG_eeg.json
        - sub-XXX_ses-YYY_task-sleep_acq-earEEG_channels.tsv
        - sub-XXX_ses-YYY_task-sleep_acq-scoring_events.tsv
        - sub-XXX_ses-YYY_task-sleep_acq-earEEGRecordingFailures_events.tsv (optional)
        - ... other PSG/trigger/dataQual files (ignored)

Task / Labels:
- Task type: sleep staging (supervised classification).
- Labels: 6-class (keep "UNSCORED"): ["W","N1","N2","N3","REM","UNSCORED"].

Segmentation policy (DATASET-DRIVEN):
- policy = "epoch_based_from_scoring_events"
- Each segment corresponds to ONE sleep-scoring epoch (default 30s).
- Epoch boundaries come from "*_task-sleep_acq-scoring_events.tsv" (per-session).
- CLI args window_sec/stride_sec are accepted (project requirement) but UNUSED here
  because the dataset is natively epoch-labeled by scoring events.

Preprocessing (group-standard defaults, configurable via CLI):
- Bandpass: 0.1–75 Hz
- Notch: 50 Hz
- Resample: target_sfreq=200 Hz (if needed)
- Channel selection: keep only ear-EEG channels; to ensure fixed dimensionality across trials,
  we compute a global channel set across the dataset (default: intersection across all recordings;
  fallback: frequency>=min_channel_presence).

Quality Control (QC):
- Reject segments with NaN/Inf.
- Reject segments with abs amplitude > max_amplitude_uv (µV) after conversion to µV.
- Reject segments overlapping with "*_acq-earEEGRecordingFailures_events.tsv" intervals (if present).
- If the whole dataset yields 0 valid segments, exit non-zero.

Outputs:
- One HDF5 per subject: OUTPUT_DIR/<dataset_name>/sub-XXX.h5
- One dataset-level dataset_info.json: OUTPUT_DIR/<dataset_name>/dataset_info.json

CLI:
  python -m benchmark_dataloader.datasets.eesm23 RAW_DATA_DIR --output_dir OUT_DIR \
      --target_sfreq 200 --filter_low 0.1 --filter_high 75 --filter_notch 50 \
      --epoch_sec 30 --max_amplitude_uv 600

Notes:
- Uses MNE to read EEGLAB .set.
- MNE stores signals in Volts; we write µV to HDF5 as recommended by the project spec.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from dataclasses import asdict
from datetime import datetime
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import pandas as pd
except ImportError as e:
    raise ImportError("EESM23 builder requires pandas to read events.tsv/channels.tsv") from e

try:
    import mne
    HAS_MNE = True
except ImportError:
    HAS_MNE = False

# ---- Project-compatible imports (preferred), with standalone fallback ----
try:
    from ..schema import SubjectAttrs, TrialAttrs, SegmentAttrs
    from ..hdf5_io import HDF5Writer
    from ..config import DatasetInfo, DatasetTaskType, DownstreamTaskType
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from schema import SubjectAttrs, TrialAttrs, SegmentAttrs  # type: ignore
    from hdf5_io import HDF5Writer  # type: ignore
    from config import DatasetInfo, DatasetTaskType, DownstreamTaskType  # type: ignore

# =============================================================================
# Mandatory constants
# =============================================================================

BUILDER_VERSION = "1.2.1"

EESM23_INFO = DatasetInfo(
    dataset_name="EESM23",
    task_type=DatasetTaskType.SLEEP,
    downstream_task_type=DownstreamTaskType.CLASSIFICATION,
    num_labels=6,
    category_list=["W", "N1", "N2", "N3", "REM", "UNSCORED"],
    sampling_rate=200.0,   # target sampling rate after resampling
    montage="earEEG",      # dataset-specific montage
    channels=[],           # filled in dataset_info.json after channel unification
)

DEFAULT_MAX_AMPLITUDE_UV = 600.0
DEFAULT_EPOCH_SEC = 30.0
DEFAULT_MIN_CHANNEL_PRESENCE = 0.90


# =============================================================================
# Helpers
# =============================================================================

def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _safe_int(s: str) -> Optional[int]:
    try:
        return int(s)
    except Exception:
        return None


def _read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", na_values=["n/a", "NA", "NaN", ""], keep_default_na=True)


def _find_one(paths: List[Path], desc: str) -> Path:
    if len(paths) == 0:
        raise FileNotFoundError(f"Missing required file: {desc}")
    if len(paths) > 1:
        # deterministic choice: shortest name, then lexicographic
        paths = sorted(paths, key=lambda p: (len(p.name), p.name))
    return paths[0]


def _stage_to_label(stage: str) -> int:
    """Map stage string to label index in EESM23_INFO.category_list."""
    s = str(stage).strip().upper()
    # common variants
    if s in {"W", "WAKE"}:
        return 0
    if s in {"N1", "S1"}:
        return 1
    if s in {"N2", "S2"}:
        return 2
    if s in {"N3", "S3", "S4"}:
        return 3
    if s in {"REM", "R"}:
        return 4
    # treat everything else as UNSCORED
    return 5


def _pick_stage_column(df: pd.DataFrame) -> str:
    """Pick a plausible stage label column from a scoring events.tsv."""
    candidates = [
        "sleep_stage", "stage", "trial_type", "value", "condition", "description", "event"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: first non-onset/duration column
    for c in df.columns:
        if c not in {"onset", "duration", "sample", "trial"}:
            return c
    raise ValueError(f"Cannot find stage column in scoring tsv. Columns={list(df.columns)}")


def _load_failure_intervals(failure_tsv: Optional[Path]) -> List[Tuple[float, float]]:
    """Return list of (start,end) in seconds."""
    if failure_tsv is None or not failure_tsv.exists():
        return []
    df = _read_tsv(failure_tsv)
    if "onset" not in df.columns:
        return []
    onset = df["onset"].astype(float).to_numpy()
    if "duration" in df.columns:
        dur = df["duration"].fillna(0).astype(float).to_numpy()
    else:
        dur = np.zeros_like(onset)
    intervals = [(float(o), float(o + d)) for o, d in zip(onset, dur)]
    # remove degenerate
    return [(a, b) for a, b in intervals if b > a]


def _overlaps_any(start: float, end: float, intervals: List[Tuple[float, float]]) -> bool:
    for a, b in intervals:
        if (start < b) and (end > a):
            return True
    return False


def _to_microvolts(data_v: np.ndarray) -> np.ndarray:
    """MNE returns Volts -> convert to microvolts for storage/QC."""
    return data_v * 1e6


def _ensure_mne() -> None:
    if not HAS_MNE:
        raise ImportError("mne is required to run the EESM23 builder. Please `pip install mne`.")


def _read_ear_channels(channels_tsv: Path) -> List[str]:
    df = _read_tsv(channels_tsv)
    # BIDS channels.tsv typically has 'name' or 'channel' column
    name_col = "name" if "name" in df.columns else ("channel" if "channel" in df.columns else None)
    if name_col is None:
        # fallback: first column
        name_col = df.columns[0]
    names = [str(x).strip() for x in df[name_col].tolist()]
    # keep non-empty, unique order-preserving
    out: List[str] = []
    for n in names:
        if n and n not in out:
            out.append(n)
    return out


def _collect_channels_global(
    raw_root: Path,
    min_presence: float = DEFAULT_MIN_CHANNEL_PRESENCE,
) -> List[str]:
    """
    Compute a global ear-EEG channel set across all recordings.

    Strategy:
    1) Read all "*_acq-earEEG_channels.tsv" and collect channel names.
    2) Prefer intersection (most stable).
    3) If intersection is empty, fall back to channels with presence >= min_presence.
    """
    ch_files = sorted(raw_root.glob("sub-*/ses-*/eeg/*_acq-earEEG_channels.tsv"))
    if not ch_files:
        raise FileNotFoundError(f"No earEEG channels.tsv files found under {raw_root}")

    all_lists = []
    for p in ch_files:
        try:
            all_lists.append(_read_ear_channels(p))
        except Exception:
            continue

    if not all_lists:
        raise RuntimeError("Failed to read any earEEG channels.tsv files.")

    # intersection
    inter = set(all_lists[0])
    for lst in all_lists[1:]:
        inter &= set(lst)
    if inter:
        # keep a stable ordering: order from the first file
        ordered = [ch for ch in all_lists[0] if ch in inter]
        return ordered

    # fallback: frequency-based
    cnt = Counter()
    for lst in all_lists:
        cnt.update(set(lst))
    total = len(all_lists)
    chosen = [ch for ch, c in cnt.items() if (c / total) >= float(min_presence)]
    chosen.sort()
    return chosen


def _load_scoring_epochs(scoring_tsv: Path, epoch_sec: float) -> Dict[int, int]:
    """
    Parse scoring events into mapping: epoch_index -> label_id.

    Uses onset seconds and bins into epoch_index = round(onset / epoch_sec).
    (We intentionally avoid float drift by rounding to nearest integer.)
    """
    df = _read_tsv(scoring_tsv)
    if "onset" not in df.columns:
        raise ValueError(f"scoring events file has no 'onset': {scoring_tsv}")
    onset = df["onset"].astype(float).to_numpy()
    stage_col = _pick_stage_column(df)
    stage = df[stage_col].astype(str).to_numpy()

    mapping: Dict[int, int] = {}
    for o, s in zip(onset, stage):
        idx = int(round(float(o) / float(epoch_sec)))
        mapping[idx] = _stage_to_label(s)
    return mapping


# =============================================================================
# Builder
# =============================================================================

class EESM23Builder:
    def __init__(
        self,
        raw_data_dir: str,
        output_dir: str,
        target_sfreq: float = 200.0,
        window_sec: float = 1.0,   # accepted but unused (epoch-based)
        stride_sec: float = 1.0,   # accepted but unused (epoch-based)
        filter_low: float = 0.1,
        filter_high: float = 75.0,
        filter_notch: float = 50.0,
        epoch_sec: float = DEFAULT_EPOCH_SEC,
        max_amplitude_uv: float = DEFAULT_MAX_AMPLITUDE_UV,
        min_channel_presence: float = DEFAULT_MIN_CHANNEL_PRESENCE,
        subjects: Optional[List[str]] = None,
    ):
        _ensure_mne()
        self.raw_root = Path(raw_data_dir)
        self.out_root = Path(output_dir) / EESM23_INFO.dataset_name
        self.out_root.mkdir(parents=True, exist_ok=True)

        self.target_sfreq = float(target_sfreq)
        self.window_sec = float(window_sec)
        self.stride_sec = float(stride_sec)
        self.filter_low = float(filter_low)
        self.filter_high = float(filter_high)
        self.filter_notch = float(filter_notch)
        self.epoch_sec = float(epoch_sec)
        self.max_amplitude_uv = float(max_amplitude_uv)
        self.min_channel_presence = float(min_channel_presence)

        # Discover subjects
        all_subjects = sorted([p.name for p in self.raw_root.glob("sub-*") if p.is_dir()])
        if subjects:
            wanted = set(subjects)
            self.subject_ids = [s for s in all_subjects if s in wanted]
        else:
            self.subject_ids = all_subjects

        # Global channels for fixed dimensionality
        self.channels_final = _collect_channels_global(self.raw_root, self.min_channel_presence)
        if not self.channels_final:
            raise RuntimeError("CHANNELS_FINAL is empty. Cannot build fixed-dim earEEG segments.")

        # bookkeeping for dataset_info.json
        self._run_stats = {
            "subjects_total": len(all_subjects),
            "subjects_selected": len(self.subject_ids),
            "total_trials": 0,
            "total_segments": 0,
            "valid_segments": 0,
            "rejected_overlap_failure": 0,
            "rejected_nonfinite": 0,
            "nonfinite_values_replaced": 0,
            "rejected_amplitude": 0,
            "errors": [],
                "warnings": [],
            "per_subject": {},
        }

    # ------------------------ scanning helpers ------------------------

    def _iter_trials_for_subject(self, subject_id: str) -> List[Path]:
        """
        Return list of session eeg directories for this subject that contain earEEG .set.
        """
        subj_dir = self.raw_root / subject_id
        eeg_dirs = sorted(subj_dir.glob("ses-*/eeg"))
        out = []
        for eeg_dir in eeg_dirs:
            if list(eeg_dir.glob(f"{subject_id}_*_task-sleep_acq-earEEG_eeg.set")):
                out.append(eeg_dir)
        return out

    # ------------------------ build methods ------------------------

    def build(self) -> None:
        print("=" * 72)
        print(f"Dataset: {EESM23_INFO.dataset_name}")
        print(f"Raw dir: {self.raw_root}")
        print(f"Output : {self.out_root}")
        print(f"Preproc: bandpass {self.filter_low}-{self.filter_high} Hz, notch {self.filter_notch} Hz, target_sfreq {self.target_sfreq} Hz")
        print(f"Seg    : epoch-based from scoring events (epoch_sec={self.epoch_sec}s); window_sec/stride_sec accepted but unused")
        print(f"Labels : {EESM23_INFO.category_list} (6-class; keep UNSCORED)")
        print("=" * 72)
        print(f"[EESM23] Found subjects: {self._run_stats['subjects_total']} | Will process: {self._run_stats['subjects_selected']}")
        print(f"[EESM23] CHANNELS_FINAL (n={len(self.channels_final)}): {self.channels_final}")

        any_valid = False
        for i, sid in enumerate(self.subject_ids, start=1):
            print(f"[{i}/{len(self.subject_ids)}] Processing subject: {sid}")
            try:
                ok = self.build_subject(sid)
                any_valid = any_valid or ok
            except Exception as e:
                msg = f"Subject {sid} failed: {repr(e)}"
                print(f"  ❌ {msg}")
                self._run_stats["errors"].append(msg)

        # dataset-level info
        self._write_dataset_info_json()

        if not any_valid:
            raise RuntimeError("EESM23 builder produced 0 valid segments overall (must exit non-zero).")

        print("[EESM23] Done.")
        print(f"[EESM23] Summary: total_segments={self._run_stats['total_segments']} valid={self._run_stats['valid_segments']} "
              f"rej_overlap={self._run_stats['rejected_overlap_failure']} rej_nonfinite={self._run_stats['rejected_nonfinite']} "
              f"rej_amp={self._run_stats['rejected_amplitude']} errors={len(self._run_stats['errors'])}")

    def build_subject(self, subject_id: str) -> bool:
        trials = self._iter_trials_for_subject(subject_id)
        self._run_stats["per_subject"][subject_id] = {
            "trials": len(trials),
            "total_segments": 0,
            "valid_segments": 0,
            "rejected_overlap_failure": 0,
            "rejected_nonfinite": 0,
            "nonfinite_values_replaced": 0,
            "rejected_amplitude": 0,
            "errors": [],
        }

        if not trials:
            print(f"  ⚠️  No earEEG trials found for {subject_id}")
            return False

        out_path = self.out_root / f"{subject_id}.h5"

        subject_has_valid = False
        # We'll create writer once per subject, and add trials sequentially
        with HDF5Writer(
            filepath=str(out_path),
            subject_attrs=SubjectAttrs(
                subject_id=subject_id,
                dataset_name=EESM23_INFO.dataset_name,
                task_type=EESM23_INFO.task_type.value,
                downstream_task_type=EESM23_INFO.downstream_task_type.value,
                rsFreq=self.target_sfreq,
                chn_name=self.channels_final,
                num_labels=EESM23_INFO.num_labels,
                category_list=EESM23_INFO.category_list,
                chn_pos=None,
                chn_ori=None,
                chn_type="EEG",
                montage=EESM23_INFO.montage,
            ),
        ) as writer:
            trial_id = 0
            for eeg_dir in trials:
                trial_id += 1
                try:
                    valid = self._build_trial(writer, subject_id, eeg_dir, trial_id)
                    subject_has_valid = subject_has_valid or valid
                except Exception as e:
                    msg = f"Trial build failed in {eeg_dir}: {repr(e)}"
                    print(f"    ❌ {msg}")
                    self._run_stats["per_subject"][subject_id]["errors"].append(msg)

        return subject_has_valid

    def _build_trial(self, writer: HDF5Writer, subject_id: str, eeg_dir: Path, trial_id: int) -> bool:
        # session_id from ses-YYY
        ses_name = eeg_dir.parent.name  # ses-YYY
        ses_num = _safe_int(ses_name.split("-")[-1]) or trial_id

        # files
        set_file = _find_one(
            list(eeg_dir.glob(f"{subject_id}_{ses_name}_task-sleep_acq-earEEG_eeg.set")),
            desc=f"earEEG .set for {subject_id}/{ses_name}",
        )
        # scoring events: try several common BIDS-conversion naming patterns
        scoring_candidates = []
        scoring_candidates += list(eeg_dir.glob(f"{subject_id}_{ses_name}_task-sleep_acq-scoring_events.tsv"))
        scoring_candidates += list(eeg_dir.glob(f"{subject_id}_{ses_name}_task-sleep*_acq-scoring*_events.tsv"))
        scoring_candidates += list(eeg_dir.glob(f"{subject_id}_{ses_name}_task-sleep*_events.tsv"))
        scoring_candidates += list(eeg_dir.glob(f"{subject_id}_{ses_name}_*scoring*_events.tsv"))
        scoring_candidates = sorted(set(scoring_candidates))
        if len(scoring_candidates) == 0:
            # Some sessions may not contain scoring; skip gracefully (do not fail whole subject)
            msg = f"Missing scoring events.tsv (or equivalent) for {subject_id}/{ses_name}; skipping session."
            print(f"    ⚠️  {msg}")
            self._run_stats["per_subject"][subject_id]["warnings"].append(msg)
            return False
        scoring_tsv = scoring_candidates[0]

        failure_tsvs = list(eeg_dir.glob(f"{subject_id}_{ses_name}_task-sleep_acq-earEEGRecordingFailures_events.tsv"))
        failure_tsv = failure_tsvs[0] if failure_tsvs else None
        failure_intervals = _load_failure_intervals(failure_tsv)

        # load raw
        raw = mne.io.read_raw_eeglab(str(set_file), preload=True, verbose="ERROR")

        # pick channels
        picked = [ch for ch in self.channels_final if ch in raw.ch_names]
        raw.pick(picked)
        # Ensure channel order matches CHANNELS_FINAL even on older MNE versions
        try:
            raw.reorder_channels(picked)
        except Exception:
            pass
        if len(raw.ch_names) != len(self.channels_final):
            missing = [ch for ch in self.channels_final if ch not in raw.ch_names]
            raise RuntimeError(f"Missing channels in {set_file.name}: {missing}")

        # filtering / resampling
        # notch
        if self.filter_notch and self.filter_notch > 0:
            raw.notch_filter(freqs=[self.filter_notch], verbose="ERROR")
        # bandpass
        raw.filter(l_freq=self.filter_low, h_freq=self.filter_high, verbose="ERROR")
        # resample
        if abs(raw.info["sfreq"] - self.target_sfreq) > 1e-6:
            raw.resample(self.target_sfreq, verbose="ERROR")

        # parse scoring
        epoch_map = _load_scoring_epochs(scoring_tsv, self.epoch_sec)

        # create epoch-based segments
        n_total = 0
        n_valid = 0
        rej_overlap = 0
        rej_nonfinite = 0
        rej_amp = 0

        # trial group
        trial_name = writer.add_trial(
            TrialAttrs(
                trial_id=trial_id,
                session_id=ses_num,
                task_name="task-sleep",
            )
        )

        # iterate epoch indices based on raw length
        total_dur = raw.times[-1]  # seconds
        n_epochs = int(np.floor(total_dur / self.epoch_sec))
        if n_epochs <= 0:
            return False

        # fetch data as (n_channels, n_times) in Volts
        data_v = raw.get_data()  # V
        # Some EEGLAB exports contain NaNs (e.g., boundaries). For benchmark-building we replace nonfinite values with 0 and track stats.
        nonfinite_count = int(np.size(data_v) - np.isfinite(data_v).sum())
        if nonfinite_count > 0:
            self._run_stats['nonfinite_values_replaced'] = self._run_stats.get('nonfinite_values_replaced', 0) + nonfinite_count
            data_v = np.nan_to_num(data_v, nan=0.0, posinf=0.0, neginf=0.0)
        sfreq = float(raw.info["sfreq"])
        samples_per_epoch = int(round(self.epoch_sec * sfreq))

        segment_id = 0
        for ei in range(n_epochs):
            start = ei * self.epoch_sec
            end = start + self.epoch_sec
            start_samp = int(round(start * sfreq))
            end_samp = start_samp + samples_per_epoch
            if end_samp > data_v.shape[1]:
                break

            n_total += 1

            # failure overlap
            if failure_intervals and _overlaps_any(start, end, failure_intervals):
                rej_overlap += 1
                continue

            seg_v = data_v[:, start_samp:end_samp]
            seg_uv = _to_microvolts(seg_v)

            # QC nonfinite
            if not np.isfinite(seg_uv).all():
                rej_nonfinite += 1
                continue

            # QC amplitude
            if np.max(np.abs(seg_uv)) > self.max_amplitude_uv:
                rej_amp += 1
                continue

            label_id = epoch_map.get(ei, 5)  # default UNSCORED
            segment_id += 1
            writer.add_segment(
                trial_name=trial_name,
                segment_attrs=SegmentAttrs(
                    segment_id=segment_id,
                    start_time=float(start),
                    end_time=float(end),
                    time_length=float(self.epoch_sec),
                    label=np.asarray([label_id], dtype=np.int64),
                    task_label="sleep_stage",
                ),
                eeg_data=seg_uv.astype(np.float32),
            )
            n_valid += 1

        # bookkeeping
        self._run_stats["total_trials"] += 1
        self._run_stats["total_segments"] += n_total
        self._run_stats["valid_segments"] += n_valid
        self._run_stats["rejected_overlap_failure"] += rej_overlap
        self._run_stats["rejected_nonfinite"] += rej_nonfinite
        self._run_stats["rejected_amplitude"] += rej_amp

        ps = self._run_stats["per_subject"][subject_id]
        ps["total_segments"] += n_total
        ps["valid_segments"] += n_valid
        ps["rejected_overlap_failure"] += rej_overlap
        ps["rejected_nonfinite"] += rej_nonfinite
        ps["rejected_amplitude"] += rej_amp

        print(f"  [EESM23] {subject_id} {eeg_dir.parent.name}: valid={n_valid} / epochs={n_total} | "
              f"rej_overlap={rej_overlap}, rej_amp={rej_amp}, rej_nonfinite={rej_nonfinite}")

        return n_valid > 0

    # ------------------------ dataset_info.json ------------------------

    def _write_dataset_info_json(self) -> None:
        info_path = self.out_root / "dataset_info.json"

        dataset_info = {
            "dataset_name": EESM23_INFO.dataset_name,
            "builder_version": BUILDER_VERSION,
            "generated_at": _now_iso(),
            "source": {
                "openneuro_id": "ds005178",
                "notes": "EESM23 BIDS-converted dataset; builder uses earEEG .set + scoring events.tsv.",
            },
            "preprocessing": {
                "target_sfreq": self.target_sfreq,
                "filter_low": self.filter_low,
                "filter_high": self.filter_high,
                "filter_notch": self.filter_notch,
                # accepted-but-unused for this policy
                "window_sec": self.window_sec,
                "stride_sec": self.stride_sec,
                "accept_but_unused": ["window_sec", "stride_sec"],
                "units_written": "uV",
            },
            "data_sanity": {
                "nonfinite_handling": "nan_to_num",
                "replacement_value_uv": 0.0,
                "values_replaced_total": self._run_stats.get("nonfinite_values_replaced", 0),
                "rationale": (
                    "EEGLAB .set files may contain NaN/Inf values (e.g., recording boundaries, dropped samples, or export artifacts). "
                    "If left untreated, sparse non-finite samples can cause entire 30s sleep-scoring epochs to be rejected, "
                    "leading to unusable subjects (valid=0) and biased retention. "
                    "For benchmark/foundation-model pretraining we replace non-finite values with 0 µV prior to filtering/segmentation, "
                    "while still tracking the number of replacements in this JSON. "
                    "After replacement, segments are still checked for any remaining non-finite values as a safety guard."
                ),
            },
            "segmentation": {
                "policy": "epoch_based_from_scoring_events",
                "parameters": {
                    "epoch_sec": self.epoch_sec,
                    "window_sec": None,
                    "stride_sec": None,
                    "alignment": "epoch_start",
                    "event_source": "*_task-sleep_acq-scoring_events.tsv",
                    "notes": "Segments are dataset-native sleep-scoring epochs; window/stride are unused.",
                },
                "rationale": (
                    "EESM23 provides per-epoch sleep staging in scoring events.tsv; using native epochs "
                    "preserves label semantics and avoids arbitrary sliding windows."
                ),
            },
            "labeling": {
                "label_policy": "supervised",
                "label_source": "*_task-sleep_acq-scoring_events.tsv",
                "classes": EESM23_INFO.category_list,
                "mapping_notes": "Unknown/other stages mapped to UNSCORED; N3 includes S3/S4 if present.",
            },
            "channels": {
                "selection": "earEEG_only",
                "unification": {
                    "method": "intersection_then_frequency_fallback",
                    "min_presence": self.min_channel_presence,
                },
                "channels_final": self.channels_final,
                "n_channels": len(self.channels_final),
            },
            "qc": {
                "rules": [
                    "reject_nonfinite",
                    "reject_amplitude",
                    "reject_overlap_recording_failures",
                ],
                "notes": "Non-finite samples are replaced with 0 µV at the raw level (see data_sanity). Remaining non-finite values are still rejected as a safety guard.",
                "max_amplitude_uv": self.max_amplitude_uv,
                "nonfinite_values_replaced_total": self._run_stats.get("nonfinite_values_replaced", 0),
                "counts": {
                    "total_segments": self._run_stats["total_segments"],
                    "valid_segments": self._run_stats["valid_segments"],
                    "rejected_overlap_failure": self._run_stats["rejected_overlap_failure"],
                    "rejected_nonfinite": self._run_stats["rejected_nonfinite"],
                    "rejected_amplitude": self._run_stats["rejected_amplitude"],
                },
            },
            "subjects": {
                "subjects_total": self._run_stats["subjects_total"],
                "subjects_selected": self._run_stats["subjects_selected"],
                "per_subject": self._run_stats["per_subject"],
            },
            "errors": self._run_stats["errors"],
        }

        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(dataset_info, f, ensure_ascii=False, indent=2)
        print(f"[EESM23] Wrote dataset_info.json -> {info_path}")


# =============================================================================
# CLI entrypoint (MUST)
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Build EESM23 (EarEEG sleep staging) into HDF5 format.")
    parser.add_argument("raw_data_dir", type=str, help="Path to raw EESM23 BIDS directory (ds005178-download).")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory root for EEG_Bench datasets.")

    parser.add_argument("--subjects", nargs="*", default=None, help="Optional list of subject IDs like sub-001 sub-002 ...")

    # Unified preprocessing args (must accept even if unused)
    parser.add_argument("--target_sfreq", type=float, default=200.0)
    parser.add_argument("--window_sec", type=float, default=1.0)
    parser.add_argument("--stride_sec", type=float, default=1.0)
    parser.add_argument("--filter_low", type=float, default=0.1)
    parser.add_argument("--filter_high", type=float, default=75.0)
    parser.add_argument("--filter_notch", type=float, default=50.0)

    # Dataset-specific segmentation args
    parser.add_argument("--epoch_sec", type=float, default=DEFAULT_EPOCH_SEC, help="Sleep scoring epoch length in seconds (default 30).")

    # QC args
    parser.add_argument("--max_amplitude_uv", type=float, default=DEFAULT_MAX_AMPLITUDE_UV)

    # Channel unification
    parser.add_argument("--min_channel_presence", type=float, default=DEFAULT_MIN_CHANNEL_PRESENCE, help="Fallback channel inclusion ratio if intersection is empty.")

    args = parser.parse_args()

    builder = EESM23Builder(
        raw_data_dir=args.raw_data_dir,
        output_dir=args.output_dir,
        target_sfreq=args.target_sfreq,
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        filter_low=args.filter_low,
        filter_high=args.filter_high,
        filter_notch=args.filter_notch,
        epoch_sec=args.epoch_sec,
        max_amplitude_uv=args.max_amplitude_uv,
        min_channel_presence=args.min_channel_presence,
        subjects=args.subjects,
    )
    builder.build()


if __name__ == "__main__":
    main()