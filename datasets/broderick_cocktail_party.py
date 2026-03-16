
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Broderick_cocktail_party (Natural Speech) builder

Goal (EEG-FM Builder 任务要求 v1.6):
- Supervised classification ONLY (num_labels >= 2)
- EEG-only HDF5, consistent channels across all subjects
- Explicit electrode names (A1–D32), fixed order
- Every segment has a label (segment -> label)
- Write dataset_info.json strictly in ADHD-style schema:
  {"dataset":{}, "processing":{}, "statistics":{}, "generated_at":...}

Task we implement here:
- Binary classification: Attend-Left vs Attend-Right (between-subject)
- Preproc: bandpass 0.1–60 Hz, then resample to 200 Hz
- No re-referencing
- Remove mastoids (not written to H5)

Assumptions (documented in dataset_info.json note):
- The released .mat files include only eegData/fs/mastoids (no per-run instruction field),
  so labels are derived deterministically from subject grouping (between-subject design).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np

try:
    import scipy.io as sio
except Exception as e:
    raise ImportError("scipy is required to read .mat files (pip install scipy)") from e

try:
    import mne
except Exception as e:
    raise ImportError("mne is required for filtering/resampling (pip install mne)") from e

# ---- support both package and standalone usage ----
try:
    from .schema import SubjectAttrs, TrialAttrs, SegmentAttrs
    from .hdf5_io import HDF5Writer
except Exception:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from schema import SubjectAttrs, TrialAttrs, SegmentAttrs
    from hdf5_io import HDF5Writer


DATASET_NAME = "BroderickCocktailParty"
EXPERIMENT_NAME = "NaturalSpeech"
TASK_TYPE = "cognitive"
DOWNSTREAM_TASK = "classification"

# Channel names (fixed order; per PI confirmation)
CH_NAMES: List[str] = (
    [f"A{i}" for i in range(1, 33)]
    + [f"B{i}" for i in range(1, 33)]
    + [f"C{i}" for i in range(1, 33)]
    + [f"D{i}" for i in range(1, 33)]
)
assert len(CH_NAMES) == 128


def subject_label(subject_index_1based: int) -> int:
    """
    Deterministic between-subject label.
    NOTE: This mapping should match the dataset release / lab convention.
          If your lab has a different mapping, change it HERE only.
    """
    # Common release convention: subjects 1–17 attend left-ear stream, 18–33 attend right-ear stream.
    return 0 if subject_index_1based <= 17 else 1


def label_name(label: int) -> str:
    return "Attend-Left" if int(label) == 0 else "Attend-Right"


def _safe_float(x: Any, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _load_mat_eeg(mat_path: Path) -> Tuple[np.ndarray, float, Optional[np.ndarray]]:
    """
    Returns:
      eeg (128, T) float64
      fs (float)
      mastoids (2, T) or None
    """
    obj = sio.loadmat(str(mat_path))
    if "eegData" not in obj or "fs" not in obj:
        raise KeyError(f"Missing keys in {mat_path.name}: need eegData and fs")

    eeg = np.asarray(obj["eegData"])
    fs = float(np.asarray(obj["fs"]).squeeze())

    mast = None
    if "mastoids" in obj:
        mast = np.asarray(obj["mastoids"])

    # EEG could be (T, C) or (C, T) depending on export
    if eeg.ndim != 2:
        raise ValueError(f"eegData must be 2D, got shape {eeg.shape}")
    if eeg.shape[0] == 128:
        eeg_ct = eeg
    elif eeg.shape[1] == 128:
        eeg_ct = eeg.T
    else:
        raise ValueError(f"Expected 128 channels, got shape {eeg.shape}")

    # mastoids could be (T,2) or (2,T)
    mast_ct = None
    if mast is not None and mast.size > 0:
        if mast.ndim != 2:
            mast_ct = None
        elif mast.shape[0] == 2:
            mast_ct = mast
        elif mast.shape[1] == 2:
            mast_ct = mast.T
        else:
            mast_ct = None

    return eeg_ct.astype(np.float64, copy=False), fs, mast_ct


def _preprocess_eeg(
    eeg_ct: np.ndarray,
    fs: float,
    filter_low: float,
    filter_high: float,
    filter_notch: float,
    target_sfreq: float,
) -> np.ndarray:
    """
    eeg_ct: (128, T)
    output: (128, T') at target_sfreq
    """
    info = mne.create_info(ch_names=CH_NAMES, sfreq=fs, ch_types=["eeg"] * 128)
    raw = mne.io.RawArray(eeg_ct, info, verbose=False)

    # notch (optional)
    if filter_notch and filter_notch > 0:
        raw.notch_filter(freqs=[filter_notch], verbose=False)

    # bandpass
    raw.filter(l_freq=filter_low, h_freq=filter_high, verbose=False)

    # resample
    if abs(raw.info["sfreq"] - target_sfreq) > 1e-6:
        raw.resample(target_sfreq, verbose=False)

    return raw.get_data().astype(np.float32, copy=False)


def _sliding_window_segments(
    data_ct: np.ndarray,
    sfreq: float,
    window_sec: float,
    stride_sec: float,
) -> List[Tuple[np.ndarray, float, float]]:
    """
    data_ct: (C, T)
    returns list of (seg_ct, start_sec, end_sec)
    """
    win = int(round(window_sec * sfreq))
    hop = int(round(stride_sec * sfreq))
    T = data_ct.shape[1]
    out: List[Tuple[np.ndarray, float, float]] = []
    if win <= 0 or hop <= 0 or T < win:
        return out
    for start in range(0, T - win + 1, hop):
        end = start + win
        s = start / sfreq
        e = end / sfreq
        out.append((data_ct[:, start:end], s, e))
    return out


def build_dataset(
    raw_root: str,
    output_dir: str,
    target_sfreq: float = 200.0,
    window_sec: float = 1.0,
    stride_sec: float = 1.0,
    filter_low: float = 0.1,
    filter_high: float = 60.0,
    filter_notch: float = 0.0,
    max_amplitude_uv: float = 600.0,
    subjects: Optional[List[int]] = None,
) -> None:
    raw_root = str(raw_root)
    output_dir = str(output_dir)

    raw_root_p = Path(raw_root)
    eeg_root = raw_root_p / "EEG"
    if not eeg_root.exists():
        raise FileNotFoundError(f"EEG folder not found: {eeg_root}")

    out_root = Path(output_dir) / DATASET_NAME
    out_root.mkdir(parents=True, exist_ok=True)

    # Subjects present in folder
    subj_dirs = sorted([p for p in eeg_root.glob("Subject*") if p.is_dir()],
                       key=lambda p: int(p.name.replace("Subject", "")))
    subj_ids_all = [int(p.name.replace("Subject", "")) for p in subj_dirs]

    if subjects:
        subj_ids = [s for s in subj_ids_all if s in set(subjects)]
    else:
        subj_ids = subj_ids_all

    # ---- stats trackers ----
    stats = {
        "total_subjects": len(subj_ids),
        "successful": 0,
        "failed": 0,
        "failed_subject_ids": [],
        "total_segments": 0,
        "valid_segments": 0,
        "rejected_segments": 0,
    }

    # detailed logs (put into dataset.note)
    per_subject_detail: Dict[str, Any] = {}

    for sid in subj_ids:
        sub_name = f"Subject{sid}"
        sub_dir = eeg_root / sub_name
        sub_tag = f"sub-{sid:03d}"

        label = int(subject_label(sid))
        lbl_name = label_name(label)

        detail = {
            "subject_dir": str(sub_dir),
            "label": label,
            "label_name": lbl_name,
            "missing_runs": [],
            "failed_runs": [],   # list of {"run": int, "reason": str}
            "runs_with_no_valid_segments": [],
            "segments": {"total": 0, "valid": 0, "rejected": 0},
            "h5_written": False,
        }

        # prepare writer lazily; only create file if we have at least 1 valid segment
        writer: Optional[HDF5Writer] = None
        out_path = out_root / f"{sub_tag}.h5"

        # subject-level attrs (will be used when opening writer)
        subject_attrs = SubjectAttrs(
            subject_id=sub_tag,
            dataset_name=DATASET_NAME,
            task_type=TASK_TYPE,
            downstream_task_type=DOWNSTREAM_TASK,
            rsFreq=float(target_sfreq),
            chn_name=CH_NAMES,
            num_labels=2,
            category_list=["Attend-Left", "Attend-Right"],
            chn_pos=None,
            chn_ori=None,
            chn_type="EEG",
            montage="BioSemi128 (A1–D32)",
        )

        # iterate expected runs 1..30
        segment_global_counter = 0
        wrote_any_valid = False

        for run in range(1, 31):
            mat_path = sub_dir / f"{sub_name}_Run{run}.mat"
            if not mat_path.exists():
                detail["missing_runs"].append(run)
                continue

            try:
                eeg_ct, fs, _mastoids = _load_mat_eeg(mat_path)
            except Exception as e:
                detail["failed_runs"].append({"run": run, "reason": f"load_mat_failed: {type(e).__name__}: {e}"})
                continue

            # preprocess
            try:
                data_ct = _preprocess_eeg(
                    eeg_ct=eeg_ct,
                    fs=fs,
                    filter_low=float(filter_low),
                    filter_high=float(filter_high),
                    filter_notch=float(filter_notch),
                    target_sfreq=float(target_sfreq),
                )
            except Exception as e:
                detail["failed_runs"].append({"run": run, "reason": f"preprocess_failed: {type(e).__name__}: {e}"})
                continue

            # segments for this run
            segs = _sliding_window_segments(
                data_ct=data_ct,
                sfreq=float(target_sfreq),
                window_sec=float(window_sec),
                stride_sec=float(stride_sec),
            )

            if not segs:
                detail["failed_runs"].append({"run": run, "reason": "no_segments_generated"})
                continue

            # open writer when first needed
            if writer is None:
                writer = HDF5Writer(str(out_path), subject_attrs)

            # trial group
            trial_attrs = TrialAttrs(trial_id=run, session_id=0, task_name=EXPERIMENT_NAME)
            trial_name = writer.add_trial(trial_attrs)

            # write segments
            valid_this_run = 0
            total_this_run = 0
            rejected_this_run = 0

            for seg_local_id, (seg_ct, s_sec, e_sec) in enumerate(segs):
                total_this_run += 1
                absmax = float(np.nanmax(np.abs(seg_ct)))
                stats["total_segments"] += 1
                detail["segments"]["total"] += 1

                if (not np.isfinite(absmax)) or (absmax > float(max_amplitude_uv)):
                    stats["rejected_segments"] += 1
                    detail["segments"]["rejected"] += 1
                    rejected_this_run += 1
                    continue

                # valid
                stats["valid_segments"] += 1
                detail["segments"]["valid"] += 1
                valid_this_run += 1
                wrote_any_valid = True

                seg_attrs = SegmentAttrs(
                    segment_id=int(seg_local_id),
                    start_time=float(s_sec),
                    end_time=float(e_sec),
                    time_length=float(window_sec),
                    label=np.array([label], dtype=np.int64),
                    task_label=lbl_name,
                )
                writer.add_segment(trial_name, seg_attrs, seg_ct)

            if valid_this_run == 0:
                detail["runs_with_no_valid_segments"].append(run)

        # close writer / finalize subject
        if writer is not None:
            writer.close()

        if wrote_any_valid:
            detail["h5_written"] = True
            stats["successful"] += 1
        else:
            # remove empty file if created but no valid segments
            if out_path.exists():
                try:
                    out_path.unlink()
                except Exception:
                    pass
            stats["failed"] += 1
            stats["failed_subject_ids"].append(sub_tag)

        per_subject_detail[sub_tag] = detail

    # ---- write dataset_info.json ----
    payload = {
        "dataset": {
            "name": DATASET_NAME,
            "experiment_name": EXPERIMENT_NAME,
            "task_type": TASK_TYPE,
            "downstream_task": DOWNSTREAM_TASK,
            "num_labels": 2,
            "category_list": ["Attend-Left", "Attend-Right"],
            "original_sampling_rate": 128.0,  # released .mat files are 128 Hz
            "channels": CH_NAMES,
            "montage": "BioSemi128 (A1–D32)",
            "note": {
                "label_definition": {
                    "0": "Attend-Left",
                    "1": "Attend-Right",
                    "mapping_rule": "between-subject; label=0 for subjects 1–17, label=1 for subjects 18–33",
                },
                "preproc_order": "bandpass(0.1–60Hz) -> resample(200Hz); no re-reference; mastoids removed",
                "mastoids": "mastoids exist in .mat but are NOT written to HDF5 (EEG-only requirement).",
                "behavioral_scores": "Comprehension scores exist (Behavioural Data/Comprehension Scores.mat) but are not used as supervision labels here.",
                "subject_details": per_subject_detail,
            },
        },
        "processing": {
            "target_sampling_rate": float(target_sfreq),
            "window_sec": float(window_sec),
            "stride_sec": float(stride_sec),
            "filter_low": float(filter_low),
            "filter_high": float(filter_high),
            "filter_notch": float(filter_notch),
            "max_amplitude_uv": float(max_amplitude_uv),
        },
        "statistics": stats,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }

    info_path = out_root / "dataset_info.json"
    info_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{DATASET_NAME}] Saved dataset_info.json -> {info_path}")
    print(f"[{DATASET_NAME}] Done. H5 files: {stats['successful']} (failed subjects: {stats['failed']})")


def main():
    parser = argparse.ArgumentParser(
        description="Build Broderick_cocktail_party (Natural Speech) into EEG_Bench HDF5 + dataset_info.json",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("raw_root", type=str, help="Path to Broderick_cocktail_party root folder")
    parser.add_argument("--output_dir", type=str, required=True, help="Output root directory (will create BroderickCocktailParty/)")
    parser.add_argument("--target_sfreq", type=float, default=200.0)
    parser.add_argument("--window_sec", type=float, default=1.0)
    parser.add_argument("--stride_sec", type=float, default=1.0)
    parser.add_argument("--filter_low", type=float, default=0.1)
    parser.add_argument("--filter_high", type=float, default=60.0)
    parser.add_argument("--filter_notch", type=float, default=50.0)
    parser.add_argument("--max_amplitude_uv", type=float, default=600.0)
    parser.add_argument("--subjects", type=int, nargs="*", default=None, help="Optional list of subject indices, e.g. 1 2 3")

    args = parser.parse_args()

    build_dataset(
        raw_root=args.raw_root,
        output_dir=args.output_dir,
        target_sfreq=args.target_sfreq,
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        filter_low=args.filter_low,
        filter_high=args.filter_high,
        filter_notch=args.filter_notch,
        max_amplitude_uv=args.max_amplitude_uv,
        subjects=args.subjects,
    )


if __name__ == "__main__":
    main()
