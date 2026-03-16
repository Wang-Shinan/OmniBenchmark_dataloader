"""MusicEEG builder (OpenNeuro ds002721) — v1.3 compliant.

Implements:
- Mandatory CLI (argparse)
- Preprocessing parameters (0.1–75 Hz, 50 Hz notch, resample 200 Hz) configurable via CLI
- Segmentation policy: sliding_window (window=1s, stride=1s)
- Segments are generated **only within music-play intervals** (event code 788; duration from events.tsv)
- Resting-state runs (run1/run6) are skipped with explicit rationale written to dataset_info.json

Trial/label semantics (from sub-*_task-run*_events.json):
- 788: "Music played" (start of music play)
- Music identity codes are *concurrent with music onset* and are NOT limited to 301–360 in practice.
  We therefore select the stimulus code as any integer in [300, 800) (excluding known non-stim codes)
  at the same onset time as 788.
- 800–807: questions Q1..Q8
- 901–909: answers 1..9 (Likert; authoritative for ratings)
- 833–841: user selected response 1..9 (may appear; treated as fallback)

We compute an 8-D trial label (Likert 1..9) by pairing each question with the first Answer
(901..909) that occurs at-or-after the question onset and before the next question (or before the
next trial block). If an Answer code is missing, we fall back to Response (833..841).
Segments inherit the trial-level 8-D label.

Outputs:
- Per-subject HDF5: sub_<id>.h5
- Dataset-level dataset_info.json in the output dataset directory

"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except Exception:
    HAS_PANDAS = False

try:
    import mne
    HAS_MNE = True
except Exception:
    HAS_MNE = False

# ---- import project modules (support both package and standalone usage) ----
try:
    from ..schema import SubjectAttrs, TrialAttrs, SegmentAttrs
    from ..hdf5_io import HDF5Writer
    from ..config import DatasetInfo, DatasetTaskType, DownstreamTaskType
except Exception:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from schema import SubjectAttrs, TrialAttrs, SegmentAttrs
    from hdf5_io import HDF5Writer
    from config import DatasetInfo, DatasetTaskType, DownstreamTaskType


MUSICEEG_INFO = DatasetInfo(
    dataset_name="MusicEEG",
    task_type=DatasetTaskType.EMOTION,
    downstream_task_type=DownstreamTaskType.REGRESSION,
    num_labels=8,  # 8 question dimensions
    category_list=[
        "pleasant",
        "energetic",
        "tense",
        "angry",
        "afraid",
        "happy",
        "sad",
        "tender",
    ],
    sampling_rate=1000.0,  # raw (will resample)
    montage="standard_1020",
)

SEGMENT_POLICY = "sliding_window"  # v1.3: must be one of sliding_window | event_locked

# --- Codes from events.json (we still code defensively) ---
CODE_MUSIC_PLAYED = 788
CODE_FIXATION = 786

QUESTION_CODES = list(range(800, 808))  # 800..807
ANSWER_CODES = list(range(901, 910))    # 901..909 -> rating 1..9 (authoritative)
RESPONSE_CODES = list(range(833, 842))  # 833..841 -> response 1..9 (fallback)

ARTIFACT_CODES = {257, 259, 263}

# Dataset-specific rationale required in dataset_info.json
REST_SKIP_REASON = (
    "Resting-state runs (run1/run6) are skipped because MusicEEG is benchmarked here as an affective "
    "music listening dataset (transient, trial-based). Rest runs are out-of-task and do not provide "
    "trial-level emotion labels; including them would mix unlabeled segments with labeled trials and "
    "complicate unified evaluation."
)


@dataclass
class RunFiles:
    run_id: int
    eeg_path: Path
    events_tsv: Path
    events_json: Optional[Path]


def _find_subject_dirs(raw_data_dir: Path) -> List[Path]:
    return sorted([p for p in raw_data_dir.glob("sub-*") if p.is_dir()])


def _find_run_files(subject_dir: Path, run_id: int) -> Optional[RunFiles]:
    eeg_dir = subject_dir / "eeg"
    if not eeg_dir.exists():
        return None

    # Typical BIDS file stem: sub-01_task-run2
    # We accept any file matching *run{run_id}_eeg.edf
    eeg = next(eeg_dir.glob(f"*run{run_id}*_eeg.edf"), None)
    if eeg is None:
        return None

    events_tsv = next(eeg_dir.glob(f"*run{run_id}*_events.tsv"), None)
    if events_tsv is None:
        return None

    events_json = next(eeg_dir.glob(f"*run{run_id}*_events.json"), None)

    return RunFiles(run_id=run_id, eeg_path=eeg, events_tsv=events_tsv, events_json=events_json)


def _safe_int(x) -> Optional[int]:
    try:
        if x is None:
            return None
        if isinstance(x, (int, np.integer)):
            return int(x)
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return None
        return int(float(x))
    except Exception:
        return None


def _load_events_meta(events_json: Optional[Path]) -> Dict[str, dict]:
    if events_json is None or (not events_json.exists()):
        return {}
    try:
        with open(events_json, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _read_events_tsv(path: Path) -> "pd.DataFrame":
    if not HAS_PANDAS:
        raise ImportError("pandas is required to read events.tsv")
    df = pd.read_csv(path, sep="\t")
    for col in ["onset", "duration", "trial_type"]:
        if col not in df.columns:
            raise ValueError(f"events.tsv missing required column '{col}': {path}")
    # normalize
    df = df.copy()
    df["onset"] = df["onset"].astype(float)
    df["duration"] = df["duration"].astype(float)
    df["trial_type"] = df["trial_type"].apply(_safe_int)
    df = df.dropna(subset=["trial_type"]).reset_index(drop=True)
    df["trial_type"] = df["trial_type"].astype(int)
    df = df.sort_values("onset").reset_index(drop=True)
    return df


def _pick_stimulus_code_at_onset(df: "pd.DataFrame", onset: float) -> Optional[int]:
    """Pick stimulus identity code that co-occurs with music onset.

    The events.json states a nominal range 301–360, but in practice we observed codes like 362.
    We therefore select any code in [300, 800) at the same onset, excluding known non-stim codes.
    """
    same = df[np.isclose(df["onset"].values, onset, atol=1e-6)]
    if same.empty:
        return None

    candidates: List[int] = []
    for c in same["trial_type"].tolist():
        if c in ARTIFACT_CODES:
            continue
        if c in {CODE_MUSIC_PLAYED, CODE_FIXATION}:
            continue
        if 300 <= c < 800 and (c not in QUESTION_CODES) and (c not in RESPONSE_CODES):
            candidates.append(int(c))

    if not candidates:
        return None

    # if multiple, choose the smallest (stable / deterministic)
    return int(sorted(candidates)[0])


def _extract_ratings_for_block(df: "pd.DataFrame", t_start: float, t_end: float) -> Optional[np.ndarray]:
    """Extract 8-D ratings (Likert 1..9) between [t_start, t_end).

    Authoritative codes (per events.json):
      - Questions: 800..807
      - Answers:   901..909  (maps to rating 1..9)

    In the provided MusicEEG files, question and answer often share the *same onset*.
    Therefore we allow answer_time >= question_time.

    Some files also include Response codes (833..841). We treat these as a fallback only
    when an Answer code is not found for a question.
    """
    block = df[(df["onset"] >= t_start) & (df["onset"] < t_end)].copy()
    if block.empty:
        return None

    # keep only question/answer/response events
    keep_codes = QUESTION_CODES + ANSWER_CODES + RESPONSE_CODES
    block = block[block["trial_type"].isin(keep_codes)].sort_values("onset").reset_index(drop=True)
    if block.empty:
        return None

    # index rows by time for scanning
    times = block["onset"].values
    codes = block["trial_type"].values

    ratings = np.full((8,), np.nan, dtype=np.float32)

    # For each question, find the first Answer (preferred) in [q_time, q_end],
    # else the first Response in [q_time, q_end].
    for qi, qcode in enumerate(QUESTION_CODES):
        # all occurrences (sometimes question may be repeated); choose first in block
        q_idx_all = np.where(codes == qcode)[0]
        if q_idx_all.size == 0:
            continue
        q_idx = int(q_idx_all[0])
        q_time = float(times[q_idx])

        # define search window end = next question time after q_time, else t_end
        next_q_times = times[(codes >= QUESTION_CODES[0]) & (codes <= QUESTION_CODES[-1]) & (times > q_time)]
        q_end = float(next_q_times.min()) if next_q_times.size > 0 else t_end

        # 1) preferred: Answer 901..909 in [q_time, q_end]
        ans_mask = (times >= q_time) & (times <= q_end) & (codes >= ANSWER_CODES[0]) & (codes <= ANSWER_CODES[-1])
        ans_idx_all = np.where(ans_mask)[0]
        if ans_idx_all.size > 0:
            ans_code = int(codes[int(ans_idx_all[0])])
            rating = ans_code - (ANSWER_CODES[0] - 1)  # 901->1 ... 909->9
            ratings[qi] = float(rating)
            continue

        # 2) fallback: Response 833..841 in [q_time, q_end]
        resp_mask = (times >= q_time) & (times <= q_end) & (codes >= RESPONSE_CODES[0]) & (codes <= RESPONSE_CODES[-1])
        resp_idx_all = np.where(resp_mask)[0]
        if resp_idx_all.size == 0:
            continue
        resp_code = int(codes[int(resp_idx_all[0])])
        rating = resp_code - (RESPONSE_CODES[0] - 1)  # 833->1 ... 841->9
        ratings[qi] = float(rating)

    if np.any(np.isnan(ratings)):
        return None

    return ratings.astype(np.float32)


def _iter_music_trials(df: "pd.DataFrame") -> List[Tuple[float, float, int, np.ndarray]]:
    """Return list of (music_onset, music_end, stim_code, ratings_8d).

    We define each *trial block* as [music_onset, next_music_onset) for extracting ratings,
    but segments are created only within [music_onset, music_onset+duration] from the 788 row.
    """
    music_rows = df[df["trial_type"] == CODE_MUSIC_PLAYED].copy()
    if music_rows.empty:
        return []

    music_onsets = music_rows["onset"].tolist()
    durations = music_rows["duration"].tolist()

    trials: List[Tuple[float, float, int, np.ndarray]] = []
    for i, (t0, dur) in enumerate(zip(music_onsets, durations)):
        music_end = float(t0 + float(dur))
        block_end = float(music_onsets[i + 1]) if i + 1 < len(music_onsets) else float(df["onset"].max() + 1.0)

        stim_code = _pick_stimulus_code_at_onset(df, float(t0))
        if stim_code is None:
            # still allow processing without stim id by setting to -1
            stim_code = -1

        ratings = _extract_ratings_for_block(df, float(t0), block_end)
        if ratings is None:
            # If ratings cannot be recovered deterministically, we skip this trial.
            # (Avoids NaN labels violating QC / downstream expectations.)
            continue

        trials.append((float(t0), music_end, int(stim_code), ratings))

    return trials


def _make_windows(t0: float, t1: float, window_sec: float, stride_sec: float) -> List[Tuple[float, float]]:
    win = float(window_sec)
    st = float(stride_sec)
    if win <= 0 or st <= 0:
        raise ValueError("window_sec and stride_sec must be > 0")

    out: List[Tuple[float, float]] = []
    t = float(t0)
    while t + win <= t1 + 1e-9:
        out.append((t, t + win))
        t += st
    return out


class MusicEEGBuilder:
    def __init__(
        self,
        raw_data_dir: str,
        output_dir: str,
        subjects: Optional[List[str]] = None,
        target_sfreq: float = 200.0,
        filter_low: float = 0.1,
        filter_high: float = 75.0,
        filter_notch: float = 50.0,
        window_sec: float = 1.0,
        stride_sec: float = 1.0,
        max_amplitude_uv: float = 600.0,
        skip_rest_runs: bool = True,
    ):
        if not HAS_MNE:
            raise ImportError("mne is required")
        if not HAS_PANDAS:
            raise ImportError("pandas is required")

        self.raw_data_dir = Path(raw_data_dir)
        self.output_root = Path(output_dir)
        self.output_dir = self.output_root / MUSICEEG_INFO.dataset_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.subject_filter = set(subjects) if subjects else None

        self.target_sfreq = float(target_sfreq)
        self.filter_low = float(filter_low)
        self.filter_high = float(filter_high)
        self.filter_notch = float(filter_notch)
        self.window_sec = float(window_sec)
        self.stride_sec = float(stride_sec)
        self.max_amplitude_uv = float(max_amplitude_uv)

        self.skip_rest_runs = bool(skip_rest_runs)

        self.subject_dirs = _find_subject_dirs(self.raw_data_dir)

    def _selected_subject_dirs(self) -> List[Path]:
        if self.subject_filter is None:
            return self.subject_dirs
        out: List[Path] = []
        for p in self.subject_dirs:
            sid = p.name.replace("sub-", "")
            if sid in self.subject_filter or p.name in self.subject_filter:
                out.append(p)
        return sorted(out)

    def build_all(self) -> None:
        subs = self._selected_subject_dirs()

        print("=" * 72)
        print(f"Dataset: {MUSICEEG_INFO.dataset_name}")
        print(f"Raw dir: {self.raw_data_dir}")
        print(f"Output dir: {self.output_dir}")
        print(f"Subjects: {len(subs)}")
        print(
            f"Preproc: bandpass {self.filter_low}-{self.filter_high} Hz, "
            f"notch {self.filter_notch} Hz, target_sfreq={self.target_sfreq} Hz"
        )
        print(
            f"Segmentation: {SEGMENT_POLICY} window={self.window_sec}s stride={self.stride_sec}s "
            f"(within music-play intervals only)"
        )
        print(f"QC: max_amplitude_uv={self.max_amplitude_uv}")
        print(f"Skip runs: {[1, 6] if self.skip_rest_runs else []}")
        print("=" * 72)

        built = 0
        failed = 0

        for i, subdir in enumerate(subs, start=1):
            sid = subdir.name.replace("sub-", "")
            print(f"[{i}/{len(subs)}] Processing sub-{sid}...")
            try:
                self.build_subject(subdir)
                built += 1
            except Exception as e:
                failed += 1
                print(f"  ❌ sub-{sid} failed: {e}")

        self._write_dataset_info_json(total_subjects=len(subs), built_subjects=built, failed_subjects=failed)

        if built == 0:
            raise RuntimeError("No subjects were built successfully (valid=0). See logs above.")

    def build_subject(self, subdir: Path) -> None:
        sid = subdir.name.replace("sub-", "")
        out_h5 = self.output_dir / f"sub_{sid}.h5"
        if out_h5.exists():
            return

        run_ids = [2, 3, 4, 5]
        if not self.skip_rest_runs:
            run_ids = [1, 6] + run_ids

        # Determine common EEG channels by scanning headers (preload=False)
        common_ch: Optional[List[str]] = None
        for r in run_ids:
            rf = _find_run_files(subdir, r)
            if rf is None:
                continue
            raw = mne.io.read_raw_edf(str(rf.eeg_path), preload=False, verbose=False)
            picks = mne.pick_types(raw.info, eeg=True, eog=False, ecg=False, emg=False, stim=False, misc=False)
            chs = [raw.ch_names[p] for p in picks]
            if common_ch is None:
                common_ch = chs
            else:
                common_ch = [c for c in common_ch if c in chs]

        if not common_ch:
            raise RuntimeError(f"No common EEG channels found for {sid}")

        subject_attrs = SubjectAttrs(
            subject_id=sid,
            dataset_name=MUSICEEG_INFO.dataset_name,
            task_type=MUSICEEG_INFO.task_type.value,
            downstream_task_type=MUSICEEG_INFO.downstream_task_type.value,
            rsFreq=self.target_sfreq,
            chn_name=common_ch,
            num_labels=MUSICEEG_INFO.num_labels,
            category_list=MUSICEEG_INFO.category_list,
            montage=MUSICEEG_INFO.montage,
        )

        total_segments = 0
        valid_segments = 0
        rejected_segments = 0

        trial_id = 0
        segment_id = 0

        with HDF5Writer(str(out_h5), subject_attrs) as writer:
            for r in run_ids:
                if self.skip_rest_runs and r in (1, 6):
                    continue

                rf = _find_run_files(subdir, r)
                if rf is None:
                    continue

                df = _read_events_tsv(rf.events_tsv)
                _ = _load_events_meta(rf.events_json)  # not required for logic, but kept for auditability

                # Extract trial list from events
                trials = _iter_music_trials(df)
                if not trials:
                    continue

                # Load EEG
                raw = mne.io.read_raw_edf(str(rf.eeg_path), preload=True, verbose=False)
                raw.pick_channels(common_ch, ordered=True)

                # Preprocessing
                raw.load_data()
                raw.filter(l_freq=self.filter_low, h_freq=self.filter_high, verbose=False)
                raw.notch_filter(freqs=[self.filter_notch], verbose=False)
                raw.resample(self.target_sfreq, npad="auto", verbose=False)

                data_v = raw.get_data()  # volts
                data_uv = data_v * 1e6

                sfreq = float(raw.info["sfreq"])

                for (t0, t1, stim_code, ratings) in trials:
                    # Create trial group
                    t_attrs = TrialAttrs(trial_id=trial_id, session_id=r, task_name=f"run{r}")
                    tname = writer.add_trial(t_attrs)

                    # Segment only within [t0, t1]
                    windows = _make_windows(t0, t1, self.window_sec, self.stride_sec)

                    for (ws, we) in windows:
                        total_segments += 1

                        s0 = int(round(ws * sfreq))
                        s1 = int(round(we * sfreq))
                        if s0 < 0 or s1 > data_uv.shape[1] or s1 <= s0:
                            rejected_segments += 1
                            continue

                        seg = data_uv[:, s0:s1]

                        # QC: NaN/Inf
                        if not np.isfinite(seg).all():
                            rejected_segments += 1
                            continue

                        # QC: amplitude threshold in uV
                        if float(np.max(np.abs(seg))) > float(self.max_amplitude_uv):
                            rejected_segments += 1
                            continue

                        seg_attrs = SegmentAttrs(
                            segment_id=segment_id,
                            start_time=float(ws),
                            end_time=float(we),
                            time_length=float(self.window_sec),
                            label=ratings.astype(np.float32),
                            task_label=f"run{r}|stim{stim_code}|trial{trial_id}",
                        )
                        writer.add_segment(tname, seg_attrs, seg)

                        valid_segments += 1
                        segment_id += 1

                    trial_id += 1

        # Ensure non-trivial output
        if valid_segments == 0:
            raise RuntimeError(f"Subject {sid} produced valid=0 segments")

    def _write_dataset_info_json(self, total_subjects: int, built_subjects: int, failed_subjects: int) -> None:
        info = {
            "dataset_name": MUSICEEG_INFO.dataset_name,
            "task_type": MUSICEEG_INFO.task_type.value,
            "downstream_task_type": MUSICEEG_INFO.downstream_task_type.value,
            "labels": {
                "type": "regression",
                "num_dims": 8,
                "dims": MUSICEEG_INFO.category_list,
                "scale": "Likert 1..9 derived from Answer codes 901..909 (fallback: Response 833..841)",
                "pairing_rule": "For each question (800..807), take the first Answer (901..909) at-or-after the question onset and before the next question; if missing, fall back to Response (833..841) in the same window.",
            },
            "segment_policy": SEGMENT_POLICY,
            "segment_parameters": {
                "window_sec": self.window_sec,
                "stride_sec": self.stride_sec,
                "alignment": None,
                "notes": "Sliding windows are generated only within music-play intervals (event code 788, duration from events.tsv).",
            },
            "preprocessing": {
                "target_sfreq": self.target_sfreq,
                "filter_low": self.filter_low,
                "filter_high": self.filter_high,
                "filter_notch": self.filter_notch,
            },
            "qc": {
                "max_amplitude_uv": self.max_amplitude_uv,
                "reject_nan_inf": True,
                "reject_over_amplitude": True,
            },
            "resting_state": {
                "skipped": bool(self.skip_rest_runs),
                "skip_runs": [1, 6] if self.skip_rest_runs else [],
                "reason": REST_SKIP_REASON if self.skip_rest_runs else "",
            },
            "build_summary": {
                "total_subjects": int(total_subjects),
                "built_subjects": int(built_subjects),
                "failed_subjects": int(failed_subjects),
            },
            "builder_version": "musiceeg_builder_v1.3.3",
        }

        out = self.output_dir / "dataset_info.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build MusicEEG HDF5 dataset (v1.3 compliant)")
    p.add_argument("raw_data_dir", type=str, help="Path to MusicEEG BIDS root (ds002721-download)")
    p.add_argument("--output_dir", type=str, required=True, help="Root output directory (EEG_Bench)")
    p.add_argument("--subjects", nargs="*", default=None, help="Optional list of subjects (e.g., sub-01 sub-02 or 01 02)")

    # v1.3 required preprocessing args
    p.add_argument("--target_sfreq", type=float, default=200.0)
    p.add_argument("--filter_low", type=float, default=0.1)
    p.add_argument("--filter_high", type=float, default=75.0)
    p.add_argument("--filter_notch", type=float, default=50.0)
    p.add_argument("--window_sec", type=float, default=1.0)
    p.add_argument("--stride_sec", type=float, default=1.0)

    # QC / dataset options
    p.add_argument("--max_amplitude_uv", type=float, default=600.0)
    p.add_argument("--include_rest", action="store_true", help="Include rest runs (run1/run6) [not recommended]")

    return p.parse_args()


def _normalize_subject_list(subs: Optional[List[str]]) -> Optional[List[str]]:
    if not subs:
        return None
    out: List[str] = []
    for s in subs:
        if s.startswith("sub-"):
            out.append(s.replace("sub-", ""))
        else:
            out.append(s)
    return out


if __name__ == "__main__":
    args = _parse_args()

    builder = MusicEEGBuilder(
        raw_data_dir=args.raw_data_dir,
        output_dir=args.output_dir,
        subjects=_normalize_subject_list(args.subjects),
        target_sfreq=args.target_sfreq,
        filter_low=args.filter_low,
        filter_high=args.filter_high,
        filter_notch=args.filter_notch,
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        max_amplitude_uv=args.max_amplitude_uv,
        skip_rest_runs=(not args.include_rest),
    )

    builder.build_all()
