#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MusicEEG Builder (v1.4 compliant + ADHD-style dataset_info.json + pleasant median-split)
======================================================================================

Dataset
-------
- Name: MusicEEG (OpenNeuro ds002721)
- Raw directory (BIDS-like):
    ds002721-download/
      dataset_description.json
      participants.tsv
      sub-01/
        eeg/
          sub-01_task-run2_eeg.edf
          sub-01_task-run2_events.tsv
          sub-01_task-run2_events.json
          ...
      ...

Benchmark framing
-----------------
- Category: III. Cognition & Emotion (music-evoked affect)
- Temporal type: transient dynamics (瞬态动态型; stimulus-block dynamics)
- Downstream task: binary classification

Labeling (NEW requirement, overrides any prior)
-----------------------------------------------
Goal: Pleasant high/low labels via median split on *clip-wise mean pleasant rating*.

Event semantics (from events.json, MUST follow dataset definition):
- Clip identity: trial_type 301–360 (mp3 index = code - 300)
- Music-play interval: trial_type 788 with duration (the interval to segment EEG within)
- Pleasant question: trial_type 800
- Pleasant answer: trial_type 901–909 (score = code-900), fallback response 833–841 (score = code-832)

Important: MusicEEG uses block/run-level organization; 301–360, 788 and questionnaire events are not guaranteed
to be tightly adjacent. This builder uses a robust, deterministic pairing rule (below).

Robust pairing rule (run-level, deterministic)
----------------------------------------------
For each run:
1) Identify clip blocks by the sequence of clip codes (301–360) ordered by onset.
   Each clip block spans [clip_onset, next_clip_onset) (or end of run).
2) Collect pleasant scores by finding each occurrence of Question 800 and its first subsequent
   Answer 901–909 (preferred) or Response 833–841 (fallback).
3) Assign pleasant score to a clip with priority:
   A) If a pleasant score onset falls within a clip block window, assign to that clip.
   B) Remaining unassigned pleasant scores are assigned by order to remaining clips in the run
      (stable left-to-right), to handle “questionnaire batch at the end” cases.
4) A clip contributes to global statistics only if a pleasant score is assigned.

Global label computation:
- For each clip code, aggregate all assigned pleasant scores across all subjects & runs (2–5).
- Compute mean pleasant per clip.
- Compute median of clip means.
- Label per clip: mean >= median -> 1 (pleasant_high), else 0 (pleasant_low).

Segmentation (v1.4)
-------------------
Policy: "within_event_sliding_window"
- Segment EEG only inside music-play intervals (trial_type 788 with duration) that overlap each clip block.
- Sliding window parameters: window_sec / stride_sec (CLI configurable; defaults 1s/1s).
- Each segment inherits the clip’s binary label (stimulus-level label).

Preprocessing (group-standard defaults; CLI configurable)
--------------------------------------------------------
- Bandpass: 0.1–75 Hz
- Notch: 50 Hz
- Resample: 200 Hz
- Keep EEG only, drop VEOG/HEOG if present.
- Use intersection of EEG channel names across subjects (common channels) for consistency.

QC (v1.4)
---------
- Reject NaN/Inf segments
- Reject segments whose max(|amplitude|) exceeds max_amplitude_uv (µV)
- Counts are recorded in dataset_info.json.
- If total valid segments overall == 0, builder exits non-zero.

Outputs
-------
- Per-subject HDF5: OUTPUT_DIR/MusicEEG/sub-XX.h5 (non-empty signals)
- Dataset metadata: OUTPUT_DIR/MusicEEG/dataset_info.json
  - MUST follow ADHD-style top-level schema: dataset / processing / statistics / generated_at
  - Additional important details are stored under notes.

CLI
---
python musiceeg_builder_v4.py RAW_DIR --output_dir OUT_DIR [--subjects sub-01 sub-02 ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import mne

# --- project-compatible imports (preferred package import, with standalone fallback) ---
try:
    from .schema import SubjectAttrs, TrialAttrs, SegmentAttrs
    from .hdf5_io import HDF5Writer
    from .config import DatasetInfo, DatasetTaskType, DownstreamTaskType
except Exception:
    _HERE = Path(__file__).resolve().parent
    if str(_HERE) not in sys.path:
        sys.path.insert(0, str(_HERE))
    from schema import SubjectAttrs, TrialAttrs, SegmentAttrs  # type: ignore
    from hdf5_io import HDF5Writer  # type: ignore
    from config import DatasetInfo, DatasetTaskType, DownstreamTaskType  # type: ignore

BUILDER_VERSION = "musiceeg_builder_v1.4_v4_blockaware_medianpleasant"

MUSICEEG_INFO = DatasetInfo(
    dataset_name="MusicEEG",
    task_type=DatasetTaskType.EMOTION,
    downstream_task_type=DownstreamTaskType.CLASSIFICATION,
    num_labels=2,
    category_list=["pleasant_low", "pleasant_high"],
    sampling_rate=200.0,
    montage="10_20",
    channels=[],
)

SKIP_RUNS = {1, 6}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build MusicEEG into EEG-FM HDF5 (pleasant median split, v1.4).")
    p.add_argument("raw_data_dir", type=str, help="Path to ds002721-download (raw root).")
    p.add_argument("--output_dir", type=str, required=True, help="Base output directory (creates MusicEEG/).")
    p.add_argument("--subjects", nargs="*", default=None, help="Optional subject IDs: sub-01 sub-02 ...")

    # group-standard defaults (must accept; may be unused)
    p.add_argument("--target_sfreq", type=float, default=200.0)
    p.add_argument("--window_sec", type=float, default=1.0)
    p.add_argument("--stride_sec", type=float, default=1.0)
    p.add_argument("--filter_low", type=float, default=0.1)
    p.add_argument("--filter_high", type=float, default=75.0)
    p.add_argument("--filter_notch", type=float, default=50.0)

    # QC
    p.add_argument("--max_amplitude_uv", type=float, default=600.0)
    return p.parse_args()


def _read_events(tsv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(tsv_path, sep="\t")
    need = {"onset", "duration", "trial_type"}
    if not need.issubset(df.columns):
        raise ValueError(f"events.tsv missing required columns {need - set(df.columns)}: {tsv_path}")
    df["onset"] = pd.to_numeric(df["onset"], errors="coerce")
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce")
    df["trial_type"] = pd.to_numeric(df["trial_type"], errors="coerce")
    df = df.dropna(subset=["onset", "duration", "trial_type"]).copy()
    df["trial_type"] = df["trial_type"].astype(int)
    return df.sort_values("onset").reset_index(drop=True)


def _answer_code_to_score(code: int) -> Optional[int]:
    if 901 <= code <= 909:
        return code - 900
    if 833 <= code <= 841:
        return code - 832
    return None


def _extract_pleasant_scores(df: pd.DataFrame) -> List[Tuple[float, int]]:
    out: List[Tuple[float, int]] = []
    q = df[df["trial_type"] == 800]
    if q.empty:
        return out
    for _, row in q.iterrows():
        q_on = float(row["onset"])
        next_dim = df[(df["trial_type"].between(801, 807)) & (df["onset"] > q_on)]
        q_off = float(next_dim.iloc[0]["onset"]) if not next_dim.empty else float(df["onset"].iloc[-1]) + 1e6

        cand = df[(df["onset"] > q_on) & (df["onset"] < q_off) & (df["trial_type"].between(833, 909))]
        if cand.empty:
            continue
        score = _answer_code_to_score(int(cand.iloc[0]["trial_type"]))
        if score is None:
            continue
        out.append((float(cand.iloc[0]["onset"]), int(score)))
    return out


def _extract_clip_blocks(df: pd.DataFrame) -> List[Tuple[int, float, float]]:
    clips = df[df["trial_type"].between(301, 360)][["onset", "trial_type"]]
    if clips.empty:
        return []
    clips = clips.sort_values("onset").reset_index(drop=True)
    blocks: List[Tuple[int, float, float]] = []
    run_end = float(df["onset"].iloc[-1] + df["duration"].iloc[-1])
    for i in range(len(clips)):
        start = float(clips.loc[i, "onset"])
        code = int(clips.loc[i, "trial_type"])
        end = float(clips.loc[i + 1, "onset"]) if i + 1 < len(clips) else run_end
        if end > start:
            blocks.append((code, start, end))
    return blocks


def _music_intervals(df: pd.DataFrame) -> List[Tuple[float, float]]:
    music = df[df["trial_type"] == 788][["onset", "duration"]]
    out: List[Tuple[float, float]] = []
    for _, r in music.iterrows():
        s = float(r["onset"])
        d = float(r["duration"])
        if d > 0:
            out.append((s, s + d))
    return out


def _assign_pleasant_to_clips(df: pd.DataFrame) -> Dict[int, List[int]]:
    blocks = _extract_clip_blocks(df)
    scores = _extract_pleasant_scores(df)  # (onset, score)
    if not blocks or not scores:
        return {}

    assigned = {code: [] for (code, _, _) in blocks}
    used = [False] * len(scores)

    # A) onset-in-block
    for (code, s0, s1) in blocks:
        for i, (on, sc) in enumerate(scores):
            if used[i]:
                continue
            if s0 <= on < s1:
                assigned[code].append(sc)
                used[i] = True
                break

    # B) order fallback
    remaining_scores = [scores[i][1] for i in range(len(scores)) if not used[i]]
    remaining_codes = [code for (code, _, _) in blocks if len(assigned[code]) == 0]
    k = min(len(remaining_scores), len(remaining_codes))
    for j in range(k):
        assigned[remaining_codes[j]].append(int(remaining_scores[j]))

    return {code: vals for code, vals in assigned.items() if vals}


def _compute_clip_labels(clip_to_scores: Dict[int, List[int]]) -> Tuple[Dict[int, float], float, Dict[int, int]]:
    means: Dict[int, float] = {}
    for c, vals in clip_to_scores.items():
        if vals:
            means[int(c)] = float(np.mean(vals))
    if not means:
        raise RuntimeError("No clip pleasant ratings found; cannot compute median split labels.")
    med = float(np.median(list(means.values())))
    labels = {c: int(means[c] >= med) for c in means}
    return means, med, labels


def _load_raw(edf_path: Path, target_sfreq: float, fl: float, fh: float, notch: float) -> mne.io.BaseRaw:
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
    raw.pick_types(eeg=True)
    drop = [ch for ch in raw.ch_names if ch.upper() in ("VEOG", "HEOG")]
    if drop:
        raw.drop_channels(drop)
    raw.filter(fl, fh, verbose=False)
    raw.notch_filter(notch, verbose=False)
    raw.resample(target_sfreq, verbose=False)
    return raw


def _scan_common_channels(subject_dirs: List[Path]) -> List[str]:
    common: Optional[set] = None
    for subdir in subject_dirs:
        eeg_dir = subdir / "eeg"
        cand = None
        for run in [2, 3, 4, 5]:
            p = eeg_dir / f"{subdir.name}_task-run{run}_eeg.edf"
            if p.exists():
                cand = p
                break
        if cand is None:
            continue
        raw = mne.io.read_raw_edf(str(cand), preload=False, verbose=False)
        raw.pick_types(eeg=True)
        chs = [c for c in raw.ch_names if c.upper() not in ("VEOG", "HEOG")]
        if not chs:
            continue
        s = set(chs)
        common = s if common is None else (common & s)
    if not common:
        return []
    return list(sorted(common))


class MusicEEGBuilder:
    def __init__(self, raw_root: Path, out_dir: Path, args: argparse.Namespace):
        self.raw_root = raw_root
        self.out_dir = out_dir
        self.args = args

        self.subject_ids: List[str] = []
        self.subject_dirs: List[Path] = []
        self.common_channels: List[str] = []

        self.clip_to_scores: Dict[int, List[int]] = {}
        self.clip_mean: Dict[int, float] = {}
        self.clip_median: float = float("nan")
        self.clip_label: Dict[int, int] = {}

        self.total_segments = 0
        self.valid_segments = 0
        self.rej_nan_inf = 0
        self.rej_amp = 0

        self.built_subjects: List[str] = []
        self.failed_subjects: Dict[str, str] = {}
        self.skips: List[Dict] = []

    def _discover_subjects(self) -> None:
        if self.args.subjects:
            subs = [s.strip() for s in self.args.subjects if s.strip()]
        else:
            subs = sorted([p.name for p in self.raw_root.iterdir() if p.is_dir() and p.name.startswith("sub-")])
        self.subject_ids = subs
        self.subject_dirs = [self.raw_root / s for s in subs]

    def _print_header(self) -> None:
        print("=" * 72)
        print("Dataset: MusicEEG")
        print(f"Raw dir: {self.raw_root}")
        print(f"Output dir: {self.out_dir}")
        print(f"Subjects: {len(self.subject_ids)}")
        print(
            f"Preproc: bandpass {self.args.filter_low}-{self.args.filter_high} Hz, "
            f"notch {self.args.filter_notch} Hz, target_sfreq={self.args.target_sfreq} Hz"
        )
        print(
            f"Segmentation: within_event_sliding_window window={self.args.window_sec}s "
            f"stride={self.args.stride_sec}s (within 788 music-play intervals inside clip blocks)"
        )
        print(f"QC: max_amplitude_uv={self.args.max_amplitude_uv}")
        print(f"Skip runs: {sorted(list(SKIP_RUNS))}")
        print("=" * 72)

    def _prepare_labels(self) -> None:
        clip_to: Dict[int, List[int]] = {}
        for sub in self.subject_ids:
            eeg_dir = self.raw_root / sub / "eeg"
            for run in [2, 3, 4, 5]:
                tsv = eeg_dir / f"{sub}_task-run{run}_events.tsv"
                if not tsv.exists():
                    self.skips.append({"subject": sub, "run": run, "reason": "missing_events_tsv"})
                    continue
                df = _read_events(tsv)
                run_map = _assign_pleasant_to_clips(df)
                if not run_map:
                    if _extract_clip_blocks(df) == []:
                        self.skips.append({"subject": sub, "run": run, "reason": "no_clip_codes_301_360"})
                    elif _extract_pleasant_scores(df) == []:
                        self.skips.append({"subject": sub, "run": run, "reason": "no_pleasant_q800_answers"})
                    else:
                        self.skips.append({"subject": sub, "run": run, "reason": "unable_to_assign_pleasant"})
                    continue
                for clip, vals in run_map.items():
                    clip_to.setdefault(int(clip), []).extend([int(v) for v in vals])
        self.clip_to_scores = clip_to
        self.clip_mean, self.clip_median, self.clip_label = _compute_clip_labels(clip_to)

    def _subject_int(self, sub: str) -> int:
        try:
            return int(sub.split("-")[1])
        except Exception:
            return -1

    def build_subject(self, sub: str) -> None:
        eeg_dir = self.raw_root / sub / "eeg"
        run_files: List[Tuple[int, Path, Path]] = []
        for run in [2, 3, 4, 5]:
            edf = eeg_dir / f"{sub}_task-run{run}_eeg.edf"
            tsv = eeg_dir / f"{sub}_task-run{run}_events.tsv"
            if edf.exists() and tsv.exists():
                run_files.append((run, edf, tsv))
        if not run_files:
            self.failed_subjects[sub] = "missing_run2_5_files"
            return

        subject_attrs = SubjectAttrs(
            subject_id=self._subject_int(sub),
            dataset_name=MUSICEEG_INFO.dataset_name,
            task_type=str(MUSICEEG_INFO.task_type),
            downstream_task_type=str(MUSICEEG_INFO.downstream_task_type),
            rsFreq=float(self.args.target_sfreq),
            chn_name=self.common_channels,
            num_labels=2,
            category_list=["pleasant_low", "pleasant_high"],
            montage="10_20",
        )

        out_path = self.out_dir / f"{sub}.h5"
        seg_id = 0
        trial_id = 0

        try:
            with HDF5Writer(str(out_path), subject_attrs=subject_attrs) as writer:
                for (run, edf_path, tsv_path) in run_files:
                    df = _read_events(tsv_path)
                    blocks = _extract_clip_blocks(df)
                    music = _music_intervals(df)
                    if not blocks or not music:
                        self.skips.append({"subject": sub, "run": run, "reason": "no_blocks_or_no_music788"})
                        continue

                    raw = _load_raw(edf_path, self.args.target_sfreq, self.args.filter_low, self.args.filter_high, self.args.filter_notch)
                    raw.pick_channels(self.common_channels, ordered=True)
                    sf = float(raw.info["sfreq"])

                    music_sorted = sorted(music, key=lambda x: x[0])

                    for (clip_code, b0, b1) in blocks:
                        if clip_code not in self.clip_label:
                            continue
                        y = int(self.clip_label[clip_code])
                        task_label = f"clip_{clip_code}"

                        intervals = []
                        for (m0, m1) in music_sorted:
                            if m1 <= b0:
                                continue
                            if m0 >= b1:
                                break
                            s0 = max(m0, b0)
                            s1 = min(m1, b1)
                            if s1 > s0:
                                intervals.append((s0, s1))
                        if not intervals:
                            continue

                        trial_attrs = TrialAttrs(trial_id=trial_id, session_id=int(run), task_name=task_label)
                        trial_name = writer.add_trial(trial_attrs)
                        trial_id += 1

                        for (t0, t1) in intervals:
                            t = t0
                            while t + self.args.window_sec <= t1 + 1e-9:
                                s0 = int(round(t * sf))
                                s1 = int(round((t + self.args.window_sec) * sf))
                                self.total_segments += 1
                                if s1 <= s0:
                                    t += self.args.stride_sec
                                    continue
                                seg_v = raw.get_data(start=s0, stop=s1)  # volts
                                if not np.isfinite(seg_v).all():
                                    self.rej_nan_inf += 1
                                    t += self.args.stride_sec
                                    continue
                                seg_uv = seg_v * 1e6
                                if np.max(np.abs(seg_uv)) > float(self.args.max_amplitude_uv):
                                    self.rej_amp += 1
                                    t += self.args.stride_sec
                                    continue

                                self.valid_segments += 1
                                seg_attrs = SegmentAttrs(
                                    segment_id=seg_id,
                                    start_time=float(t),
                                    end_time=float(t + self.args.window_sec),
                                    time_length=float(self.args.window_sec),
                                    label=np.array([y], dtype=np.int64),
                                    task_label=task_label,
                                )
                                writer.add_segment(trial_name, seg_attrs, seg_uv.astype(np.float32))
                                seg_id += 1
                                t += self.args.stride_sec

            if seg_id == 0:
                self.failed_subjects[sub] = "no_valid_segments_after_qc_or_no_labeled_trials"
                try:
                    out_path.unlink(missing_ok=True)  # type: ignore
                except Exception:
                    pass
                return

            self.built_subjects.append(sub)

        except Exception as e:
            self.failed_subjects[sub] = f"exception:{type(e).__name__}:{e}"
            try:
                out_path.unlink(missing_ok=True)  # type: ignore
            except Exception:
                pass

    def _write_dataset_info(self) -> None:
        payload = {
            "dataset": {
                "dataset_name": MUSICEEG_INFO.dataset_name,
                "dataset_identifier": "openneuro:ds002721",
                "experiment_name": "MusicEEG",
                "task_type": "emotion",
                "task_characteristic": "transient_dynamic",
                "subjects": len(self.subject_ids),
                "trials_per_subject": None,
            },
            "processing": {
                "raw_data_dir": str(self.raw_root),
                "output_dir": str(self.out_dir),
                "preprocessing": {
                    "target_sfreq": float(self.args.target_sfreq),
                    "filter_low": float(self.args.filter_low),
                    "filter_high": float(self.args.filter_high),
                    "filter_notch": float(self.args.filter_notch),
                    "accept_but_unused": [],
                },
                "channels": {
                    "policy": "intersection_across_subjects",
                    "common_eeg_channels": self.common_channels,
                    "n_channels": len(self.common_channels),
                },
                "segmentation": {
                    "policy": "within_event_sliding_window",
                    "parameters": {
                        "window_sec": float(self.args.window_sec),
                        "stride_sec": float(self.args.stride_sec),
                        "alignment": None,
                        "event_source": "events.tsv (clip 301–360; music 788)",
                        "notes": "Windows sampled only during music-play (788) intervals that overlap clip blocks defined by successive clip codes (301–360).",
                    },
                    "rationale": "Continuous music listening; segment only when music is playing and assign clip-level affect label."
                },
                "labeling": {
                    "policy": "supervised",
                    "labels": ["pleasant_low", "pleasant_high"],
                    "num_labels": 2,
                    "definition": (
                        "Pleasant dimension from Question 800. Answer preferred 901–909 (score=code-900), "
                        "fallback 833–841 (score=code-832). Clip identity from 301–360. "
                        "Compute per-clip mean rating across subjects/runs; median of means used for binary split."
                    ),
                    "threshold_median": float(self.clip_median),
                },
                "qc": {
                    "max_amplitude_uv": float(self.args.max_amplitude_uv),
                    "reject_nan_inf": True,
                    "reject_over_amplitude": True,
                },
            },
            "statistics": {
                "subjects_total": len(self.subject_ids),
                "subjects_built": len(self.built_subjects),
                "subjects_failed": len(self.failed_subjects),
                "failed_subjects": self.failed_subjects,
                "segments_total": int(self.total_segments),
                "segments_valid": int(self.valid_segments),
                "segments_rejected_nan_inf": int(self.rej_nan_inf),
                "segments_rejected_over_amplitude": int(self.rej_amp),
                "num_clips_with_labels": int(len(self.clip_label)),
            },
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "notes": {
                "builder_version": BUILDER_VERSION,
                "clip_mean_pleasant": self.clip_mean,
                "clip_binary_label": self.clip_label,
                "pleasant_pairing_rule": "A) assign by question onset within clip block; B) order fallback",
                "skipped_records_sample": self.skips[:200],
                "skip_runs": sorted(list(SKIP_RUNS)),
            },
        }
        out_path = self.out_dir / "dataset_info.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def build_all(self) -> None:
        self._discover_subjects()
        self.common_channels = _scan_common_channels(self.subject_dirs)
        if not self.common_channels:
            raise RuntimeError("Could not determine common EEG channels across subjects.")
        self._print_header()
        self._prepare_labels()

        for i, sub in enumerate(self.subject_ids, 1):
            print(f"[{i}/{len(self.subject_ids)}] Processing {sub}...")
            self.build_subject(sub)

        self._write_dataset_info()

        if self.valid_segments == 0:
            raise RuntimeError("No valid segments produced overall (after QC / labeling).")

        print("-" * 72)
        print(f"Built subjects: {len(self.built_subjects)} / {len(self.subject_ids)}")
        print(f"Failed subjects: {len(self.failed_subjects)}")
        print(f"Segments: total={self.total_segments} valid={self.valid_segments} rej_nan_inf={self.rej_nan_inf} rej_amp={self.rej_amp}")
        print(f"Wrote dataset_info.json: {self.out_dir / 'dataset_info.json'}")
        print("-" * 72)


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_data_dir)
    out_dir = Path(args.output_dir) / MUSICEEG_INFO.dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    builder = MusicEEGBuilder(raw_root=raw_root, out_dir=out_dir, args=args)
    builder.build_all()


if __name__ == "__main__":
    main()
