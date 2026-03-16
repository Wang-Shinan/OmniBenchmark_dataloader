"""
BroderickReverse Builder (Natural Speech vs Time-Reversed Speech) — EEG-FM v1.6 compliant.

Task
----
Supervised binary classification:
  label 0: natural_speech
  label 1: time_reversed_speech

Raw layout (as provided by user):
  <RAW_ROOT>/Natural Speech/EEG/Subject{1..19}/Subject{X}_Run{1..20}.mat
  <RAW_ROOT>/Natural Speech - Reverse/EEG/Subject{1..10}/Subject{X}_Run{1..20}.mat

Each .mat contains:
  eegData : EEG (unfiltered, unreferenced), 128 channels
  mastoids: 2 mastoid channels (must be removed; NOT written to HDF5)
  fs      : sampling rate (expected 128 Hz)

This builder:
- keeps EEG only (128 channels), drops mastoids
- bandpass 0.1–60 Hz
- resamples to 200 Hz
- sliding-window segmentation (default 1s / 1s)
- optional amplitude rejection by |uV| threshold (default disabled by very large value)
- outputs 1 HDF5 per "global subject":
    * Natural subjects: sub_XXX.h5 (XXX = Natural subject id, 3-digit)
    * Reverse-only (mapping-unknown) subjects: sub_RXXX.h5 (XXX = Reverse subject id, 3-digit)
- generates dataset_info.json strictly aligned to EEG-FM Builder Task Requirements v1.6

Subject mapping (README-provided):
  Reverse subjects 5,6,7,10 correspond to Natural subjects 4,6,8,10 respectively.
  => reverse data from (R5,R6,R7,R10) are merged into (N4,N6,N8,N10) files.
  => reverse subjects without mapping (R1,R2,R3,R4,R8,R9) are kept as independent reverse-only subjects (2B).

Note: Baseline is NOT cropped (user confirmed).

CLI
---
python broderick_reverse.py <RAW_ROOT> --output_dir <OUT_ROOT> [args...]

"""
from __future__ import annotations

import json
import re
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt, resample_poly

try:
    from ..schema import SubjectAttrs, TrialAttrs, SegmentAttrs
    from ..hdf5_io import HDF5Writer
    from ..config import DatasetInfo, DatasetTaskType, DownstreamTaskType
except ImportError:
    # Allow running as a standalone script
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from schema import SubjectAttrs, TrialAttrs, SegmentAttrs
    from hdf5_io import HDF5Writer
    from config import DatasetInfo, DatasetTaskType, DownstreamTaskType


BASE_DATASET_NAME = "BroderickReverse"

LABEL_NAMES = ["natural_speech", "time_reversed_speech"]  # index = label id

# Channel naming (128-ch BioSemi-style cap grid A/B/C/D 1..32) — user confirmed to follow image.
CHANNELS_128 = (
    [f"A{i}" for i in range(1, 33)] +
    [f"B{i}" for i in range(1, 33)] +
    [f"C{i}" for i in range(1, 33)] +
    [f"D{i}" for i in range(1, 33)]
)

# Reverse -> Natural mapping explicitly provided by README snippet discussed with user/PI.
REVERSE_TO_NATURAL_MAP = {5: 4, 6: 6, 7: 8, 10: 10}
NATURAL_TO_REVERSE_MAP = {v: k for k, v in REVERSE_TO_NATURAL_MAP.items()}


BRODERICK_INFO = DatasetInfo(
    dataset_name=BASE_DATASET_NAME,
    task_type=DatasetTaskType.COGNITIVE,
    downstream_task_type=DownstreamTaskType.CLASSIFICATION,
    num_labels=2,
    category_list=LABEL_NAMES,
    sampling_rate=128.0,  # raw
    montage="BioSemi_128_custom",  # custom naming A1..D32
    channels=CHANNELS_128,
)


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _parse_subject_id_from_folder(p: Path) -> Optional[int]:
    m = re.search(r"Subject(\d+)$", p.name)
    return int(m.group(1)) if m else None


def _parse_run_id_from_filename(p: Path) -> Optional[int]:
    m = re.search(r"_Run(\d+)\.mat$", p.name)
    return int(m.group(1)) if m else None


def _safe_load_mat(mat_path: Path) -> Tuple[np.ndarray, float]:
    """Return (eeg[ch, t], fs). Drops mastoids (never returned)."""
    mat = loadmat(str(mat_path))
    if "eegData" not in mat or "fs" not in mat:
        raise KeyError(f"Missing keys in {mat_path.name}: expected eegData and fs")

    eeg = mat["eegData"]
    fs = float(np.asarray(mat["fs"]).squeeze())

    if eeg.ndim != 2:
        raise ValueError(f"Bad eegData shape in {mat_path.name}: {eeg.shape}")

    # eegData may be (T, C) or (C, T). We need (C, T) with C=128.
    if eeg.shape[0] == 128 and eeg.shape[1] != 128:
        eeg_ct = eeg.astype(np.float64)
    elif eeg.shape[1] == 128 and eeg.shape[0] != 128:
        eeg_ct = eeg.T.astype(np.float64)
    elif eeg.shape[0] == 128 and eeg.shape[1] == 128:
        # ambiguous but treat as (C,T)
        eeg_ct = eeg.astype(np.float64)
    else:
        raise ValueError(f"Cannot infer channel axis in {mat_path.name}: {eeg.shape}")

    if eeg_ct.shape[0] != 128:
        raise ValueError(f"Expected 128 EEG channels, got {eeg_ct.shape[0]} in {mat_path.name}")

    # Ensure finite
    if not np.isfinite(eeg_ct).all():
        raise ValueError(f"Non-finite values in {mat_path.name}")

    return eeg_ct, fs


def _design_bandpass(fs: float, low: float, high: float, order: int = 4):
    nyq = 0.5 * fs
    low_n = max(low / nyq, 1e-6)
    high_n = min(high / nyq, 0.999999)
    if not (0 < low_n < high_n < 1):
        raise ValueError(f"Invalid bandpass after normalization: low={low_n}, high={high_n}, fs={fs}")
    return butter(order, [low_n, high_n], btype="bandpass", output="sos")


def _apply_notch(x: np.ndarray, fs: float, notch: float, q: float = 30.0) -> np.ndarray:
    """x: (C,T)"""
    if notch is None or notch <= 0:
        return x
    w0 = notch / (fs / 2.0)
    if w0 <= 0 or w0 >= 1:
        return x
    b, a = iirnotch(w0=w0, Q=q)
    # filtfilt along time axis
    return filtfilt(b, a, x, axis=1)


def _preprocess_eeg(
    eeg_ct: np.ndarray,
    fs: float,
    target_sfreq: float,
    filter_low: float,
    filter_high: float,
    filter_notch: float,
) -> Tuple[np.ndarray, float]:
    """Return (eeg_ct_processed, new_fs). Keeps units unchanged."""
    x = eeg_ct

    # Bandpass first (stabilize), then notch (optional), then resample.
    sos = _design_bandpass(fs, filter_low, filter_high, order=4)
    x = sosfiltfilt(sos, x, axis=1)

    if filter_notch and filter_notch > 0:
        x = _apply_notch(x, fs, filter_notch, q=30.0)

    # Resample to target_sfreq using rational resample_poly when possible.
    if abs(fs - target_sfreq) > 1e-6:
        # Reduce ratio by gcd for common rates.
        # For fs=128, target=200 -> up=25, down=16
        from math import gcd
        up = int(round(target_sfreq))
        down = int(round(fs))
        g = gcd(up, down)
        up //= g
        down //= g
        x = resample_poly(x, up=up, down=down, axis=1)
        fs = float(target_sfreq)

    return x, fs


def _iter_run_files(subject_dir: Path) -> List[Path]:
    mats = sorted(subject_dir.glob("Subject*_Run*.mat"))
    # fallback (some listings may use Run1.mat style)
    if not mats:
        mats = sorted(subject_dir.glob("Run*.mat"))
    return mats


def _segment_sliding_window(
    eeg_ct: np.ndarray,
    sfreq: float,
    window_sec: float,
    stride_sec: float,
) -> List[Tuple[int, int, float, float]]:
    """Return list of (start_samp, end_samp, start_time, end_time)."""
    win = int(round(window_sec * sfreq))
    stride = int(round(stride_sec * sfreq))
    if win <= 0 or stride <= 0:
        raise ValueError(f"Bad window/stride: window_sec={window_sec}, stride_sec={stride_sec}, sfreq={sfreq}")

    n = eeg_ct.shape[1]
    if n < win:
        return []

    segments = []
    for s in range(0, n - win + 1, stride):
        e = s + win
        segments.append((s, e, s / sfreq, e / sfreq))
    return segments


class BroderickReverseBuilder:
    def __init__(
        self,
        raw_root: str,
        output_dir: str,
        target_sfreq: float = 200.0,
        window_sec: float = 1.0,
        stride_sec: float = 1.0,
        filter_low: float = 0.1,
        filter_high: float = 60.0,
        filter_notch: float = 50.0,
        max_amplitude_uv: float = 1e9,  # default "disabled"
    ):
        self.raw_root = Path(raw_root)
        self.out_root = Path(output_dir) / BASE_DATASET_NAME
        self.out_root.mkdir(parents=True, exist_ok=True)

        self.target_sfreq = float(target_sfreq)
        self.window_sec = float(window_sec)
        self.stride_sec = float(stride_sec)
        self.filter_low = float(filter_low)
        self.filter_high = float(filter_high)
        self.filter_notch = float(filter_notch)
        self.max_amplitude_uv = float(max_amplitude_uv)

        self.natural_eeg_dir = self.raw_root / "Natural Speech" / "EEG"
        self.reverse_eeg_dir = self.raw_root / "Natural Speech - Reverse" / "EEG"

        if not self.natural_eeg_dir.exists():
            raise FileNotFoundError(f"Missing directory: {self.natural_eeg_dir}")
        if not self.reverse_eeg_dir.exists():
            raise FileNotFoundError(f"Missing directory: {self.reverse_eeg_dir}")

        self.stats = {
            "total_subjects": 0,
            "successful": 0,
            "failed": 0,
            "failed_subject_ids": [],
            "total_segments": 0,
            "valid_segments": 0,
            "rejected_segments": 0,
        }
        self._fail_reasons: Dict[str, str] = {}
        self._built_files: List[str] = []

    # ---------- subject planning ----------
    def _list_natural_subjects(self) -> List[int]:
        subs = []
        for p in sorted(self.natural_eeg_dir.glob("Subject*")):
            sid = _parse_subject_id_from_folder(p)
            if sid is not None:
                subs.append(sid)
        return sorted(set(subs))

    def _list_reverse_subjects(self) -> List[int]:
        subs = []
        for p in sorted(self.reverse_eeg_dir.glob("Subject*")):
            sid = _parse_subject_id_from_folder(p)
            if sid is not None:
                subs.append(sid)
        return sorted(set(subs))

    def _plan_output_subjects(self) -> Tuple[List[int], List[int]]:
        """Return (natural_global_ids, reverse_only_reverse_ids)."""
        natural_ids = self._list_natural_subjects()
        reverse_ids = self._list_reverse_subjects()

        # reverse subjects that have explicit mapping are merged into natural subjects
        mapped_reverse_ids = set(REVERSE_TO_NATURAL_MAP.keys())
        reverse_only = [rid for rid in reverse_ids if rid not in mapped_reverse_ids]

        return natural_ids, reverse_only

    # ---------- building ----------
    def _build_h5_for_natural_subject(self, natural_id: int) -> str:
        out_file = self.out_root / f"sub_{natural_id:03d}.h5"
        if out_file.exists():
            return str(out_file)

        natural_dir = self.natural_eeg_dir / f"Subject{natural_id}"
        if not natural_dir.exists():
            raise FileNotFoundError(f"Natural subject folder missing: {natural_dir}")

        # Prepare subject attrs (channels are fixed & enforced)
        subject_attrs = SubjectAttrs(
            subject_id=int(natural_id),
            dataset_name=BRODERICK_INFO.dataset_name,
            task_type=BRODERICK_INFO.task_type.value,
            downstream_task_type=BRODERICK_INFO.downstream_task_type.value,
            rsFreq=self.target_sfreq,
            chn_name=CHANNELS_128,
            num_labels=BRODERICK_INFO.num_labels,
            category_list=BRODERICK_INFO.category_list,
            chn_pos=None,
            chn_ori=None,
            chn_type="EEG",
            montage=BRODERICK_INFO.montage,
        )

        trial_id = 0
        with HDF5Writer(str(out_file), subject_attrs) as writer:
            # ----- Natural trials (label 0) -----
            run_files = _iter_run_files(natural_dir)
            if not run_files:
                raise FileNotFoundError(f"No run .mat files found under {natural_dir}")

            for rf in run_files:
                run_id = _parse_run_id_from_filename(rf) or trial_id + 1
                eeg_ct, fs = _safe_load_mat(rf)
                eeg_ct, fs2 = _preprocess_eeg(
                    eeg_ct,
                    fs=fs,
                    target_sfreq=self.target_sfreq,
                    filter_low=self.filter_low,
                    filter_high=self.filter_high,
                    filter_notch=self.filter_notch,
                )
                if abs(fs2 - self.target_sfreq) > 1e-6:
                    raise RuntimeError(f"Resample failed: got sfreq={fs2}")

                trial_attrs = TrialAttrs(
                    trial_id=trial_id,
                    session_id=0,
                    task_name=f"natural_run{int(run_id):02d}",
                )
                trial_name = writer.add_trial(trial_attrs)

                seg_windows = _segment_sliding_window(eeg_ct, fs2, self.window_sec, self.stride_sec)
                if not seg_windows:
                    # counts as failure of this trial but not entire subject
                    continue

                seg_id = 0
                for (s, e, st, et) in seg_windows:
                    seg = eeg_ct[:, s:e]
                    self.stats["total_segments"] += 1

                    if not np.isfinite(seg).all():
                        self.stats["rejected_segments"] += 1
                        continue
                    if self.max_amplitude_uv and np.max(np.abs(seg)) > self.max_amplitude_uv:
                        self.stats["rejected_segments"] += 1
                        continue

                    seg_attrs = SegmentAttrs(
                        segment_id=seg_id,
                        start_time=float(st),
                        end_time=float(et),
                        time_length=float(self.window_sec),
                        label=np.array([0], dtype=np.int64),
                        task_label="natural_speech",
                    )
                    writer.add_segment(trial_name, seg_attrs, seg.astype(np.float32))
                    self.stats["valid_segments"] += 1
                    seg_id += 1

                trial_id += 1

            # ----- Reverse trials (label 1), only if mapping exists -----
            if natural_id in NATURAL_TO_REVERSE_MAP:
                reverse_id = NATURAL_TO_REVERSE_MAP[natural_id]
                reverse_dir = self.reverse_eeg_dir / f"Subject{reverse_id}"
                if reverse_dir.exists():
                    run_files_r = _iter_run_files(reverse_dir)
                    for rf in run_files_r:
                        run_id = _parse_run_id_from_filename(rf) or (trial_id + 1)
                        eeg_ct, fs = _safe_load_mat(rf)
                        eeg_ct, fs2 = _preprocess_eeg(
                            eeg_ct,
                            fs=fs,
                            target_sfreq=self.target_sfreq,
                            filter_low=self.filter_low,
                            filter_high=self.filter_high,
                            filter_notch=self.filter_notch,
                        )

                        trial_attrs = TrialAttrs(
                            trial_id=trial_id,
                            session_id=1,
                            task_name=f"reverse_run{int(run_id):02d}_srcSub{reverse_id:02d}",
                        )
                        trial_name = writer.add_trial(trial_attrs)

                        seg_windows = _segment_sliding_window(eeg_ct, fs2, self.window_sec, self.stride_sec)
                        if not seg_windows:
                            continue

                        seg_id = 0
                        for (s, e, st, et) in seg_windows:
                            seg = eeg_ct[:, s:e]
                            self.stats["total_segments"] += 1

                            if not np.isfinite(seg).all():
                                self.stats["rejected_segments"] += 1
                                continue
                            if self.max_amplitude_uv and np.max(np.abs(seg)) > self.max_amplitude_uv:
                                self.stats["rejected_segments"] += 1
                                continue

                            seg_attrs = SegmentAttrs(
                                segment_id=seg_id,
                                start_time=float(st),
                                end_time=float(et),
                                time_length=float(self.window_sec),
                                label=np.array([1], dtype=np.int64),
                                task_label="time_reversed_speech",
                            )
                            writer.add_segment(trial_name, seg_attrs, seg.astype(np.float32))
                            self.stats["valid_segments"] += 1
                            seg_id += 1

                        trial_id += 1

        return str(out_file)

    def _build_h5_for_reverse_only(self, reverse_id: int) -> str:
        out_file = self.out_root / f"sub_R{reverse_id:03d}.h5"
        if out_file.exists():
            return str(out_file)

        reverse_dir = self.reverse_eeg_dir / f"Subject{reverse_id}"
        if not reverse_dir.exists():
            raise FileNotFoundError(f"Reverse subject folder missing: {reverse_dir}")

        subject_attrs = SubjectAttrs(
            subject_id=f"R{reverse_id:03d}",
            dataset_name=BRODERICK_INFO.dataset_name,
            task_type=BRODERICK_INFO.task_type.value,
            downstream_task_type=BRODERICK_INFO.downstream_task_type.value,
            rsFreq=self.target_sfreq,
            chn_name=CHANNELS_128,
            num_labels=BRODERICK_INFO.num_labels,
            category_list=BRODERICK_INFO.category_list,
            chn_pos=None,
            chn_ori=None,
            chn_type="EEG",
            montage=BRODERICK_INFO.montage,
        )

        trial_id = 0
        with HDF5Writer(str(out_file), subject_attrs) as writer:
            run_files = _iter_run_files(reverse_dir)
            if not run_files:
                raise FileNotFoundError(f"No run .mat files found under {reverse_dir}")

            for rf in run_files:
                run_id = _parse_run_id_from_filename(rf) or trial_id + 1
                eeg_ct, fs = _safe_load_mat(rf)
                eeg_ct, fs2 = _preprocess_eeg(
                    eeg_ct,
                    fs=fs,
                    target_sfreq=self.target_sfreq,
                    filter_low=self.filter_low,
                    filter_high=self.filter_high,
                    filter_notch=self.filter_notch,
                )

                trial_attrs = TrialAttrs(
                    trial_id=trial_id,
                    session_id=1,
                    task_name=f"reverse_run{int(run_id):02d}_srcSub{reverse_id:02d}_reverseOnly",
                )
                trial_name = writer.add_trial(trial_attrs)

                seg_windows = _segment_sliding_window(eeg_ct, fs2, self.window_sec, self.stride_sec)
                if not seg_windows:
                    continue

                seg_id = 0
                for (s, e, st, et) in seg_windows:
                    seg = eeg_ct[:, s:e]
                    self.stats["total_segments"] += 1

                    if not np.isfinite(seg).all():
                        self.stats["rejected_segments"] += 1
                        continue
                    if self.max_amplitude_uv and np.max(np.abs(seg)) > self.max_amplitude_uv:
                        self.stats["rejected_segments"] += 1
                        continue

                    seg_attrs = SegmentAttrs(
                        segment_id=seg_id,
                        start_time=float(st),
                        end_time=float(et),
                        time_length=float(self.window_sec),
                        label=np.array([1], dtype=np.int64),
                        task_label="time_reversed_speech",
                    )
                    writer.add_segment(trial_name, seg_attrs, seg.astype(np.float32))
                    self.stats["valid_segments"] += 1
                    seg_id += 1

                trial_id += 1

        return str(out_file)

    def build_all(self) -> List[str]:
        natural_ids, reverse_only_ids = self._plan_output_subjects()
        planned_total = len(natural_ids) + len(reverse_only_ids)
        self.stats["total_subjects"] = planned_total

        built: List[str] = []

        # Build natural (with mapped reverse merged where applicable)
        for nid in natural_ids:
            sid = f"{nid:03d}"
            try:
                path = self._build_h5_for_natural_subject(nid)
                built.append(path)
                self.stats["successful"] += 1
            except Exception as e:
                self.stats["failed"] += 1
                self.stats["failed_subject_ids"].append(f"sub_{sid}")
                self._fail_reasons[f"sub_{sid}"] = f"{type(e).__name__}: {e}"

        # Build reverse-only (2B) as independent subjects
        for rid in reverse_only_ids:
            sid = f"R{rid:03d}"
            try:
                path = self._build_h5_for_reverse_only(rid)
                built.append(path)
                self.stats["successful"] += 1
            except Exception as e:
                self.stats["failed"] += 1
                self.stats["failed_subject_ids"].append(f"sub_{sid}")
                self._fail_reasons[f"sub_{sid}"] = f"{type(e).__name__}: {e}"

        self._built_files = built
        return built

    def finalize_dataset_info(self) -> str:
        info_path = self.out_root / "dataset_info.json"

        payload = {
            "dataset": {
                "name": BRODERICK_INFO.dataset_name,
                "task_type": BRODERICK_INFO.task_type.value,
                "downstream_task": "classification",
                "num_labels": int(BRODERICK_INFO.num_labels),
                "category_list": list(BRODERICK_INFO.category_list),
                "original_sampling_rate": float(BRODERICK_INFO.sampling_rate),
                "channels": list(BRODERICK_INFO.channels),
                "montage": BRODERICK_INFO.montage,
                "note": (
                    "Binary classification: natural speech vs time-reversed speech. "
                    "EEG-only 128 channels named A1..D32 (user-confirmed). Mastoids are dropped (not written). "
                    "Bandpass 0.1–60 Hz then resample to 200 Hz. Baseline not cropped. "
                    f"Explicit reverse→natural mapping merged: {REVERSE_TO_NATURAL_MAP}. "
                    "Reverse subjects without mapping are retained as independent reverse-only subjects (sub_RXXX.h5, label=1 only). "
                    "Subject-level split is required to avoid leakage."
                ),
            },
            "processing": {
                "target_sampling_rate": float(self.target_sfreq),
                "window_sec": float(self.window_sec),
                "stride_sec": float(self.stride_sec),
                "filter_low": float(self.filter_low),
                "filter_high": float(self.filter_high),
                "filter_notch": float(self.filter_notch),
                "max_amplitude_uv": float(self.max_amplitude_uv),
            },
            "statistics": {
                "total_subjects": int(self.stats["total_subjects"]),
                "successful": int(self.stats["successful"]),
                "failed": int(self.stats["failed"]),
                "failed_subject_ids": list(self.stats["failed_subject_ids"]),
                "total_segments": int(self.stats["total_segments"]),
                "valid_segments": int(self.stats["valid_segments"]),
                "rejected_segments": int(self.stats["rejected_segments"]),
            },
            "generated_at": _now_iso(),
        }

        # Add failure reasons into note (as additional transparency) without breaking schema.
        if self._fail_reasons:
            extra = json.dumps(self._fail_reasons, ensure_ascii=False)
            payload["dataset"]["note"] += f" | build_fail_reasons={extra}"

        info_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(info_path)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build BroderickReverse (Natural vs Time-Reversed) into EEG-FM HDF5 format.")
    parser.add_argument("raw_root", type=str, help="Raw dataset root (contains 'Natural Speech' and 'Natural Speech - Reverse').")
    parser.add_argument("--output_dir", type=str, required=True, help="Output root dir (dataset will be under <output_dir>/BroderickReverse/).")

    parser.add_argument("--target_sfreq", type=float, default=200.0)
    parser.add_argument("--window_sec", type=float, default=1.0)
    parser.add_argument("--stride_sec", type=float, default=1.0)

    parser.add_argument("--filter_low", type=float, default=0.1)
    parser.add_argument("--filter_high", type=float, default=60.0)
    parser.add_argument("--filter_notch", type=float, default=0.0)

    parser.add_argument("--max_amplitude_uv", type=float, default=1e9, help="Reject segments with abs(uV) > threshold. Default disables rejection.")

    args = parser.parse_args()

    builder = BroderickReverseBuilder(
        raw_root=args.raw_root,
        output_dir=args.output_dir,
        target_sfreq=args.target_sfreq,
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        filter_low=args.filter_low,
        filter_high=args.filter_high,
        filter_notch=args.filter_notch,
        max_amplitude_uv=args.max_amplitude_uv,
    )

    built = builder.build_all()
    info_path = builder.finalize_dataset_info()

    print("=" * 72)
    print(f"Dataset: {BASE_DATASET_NAME}")
    print(f"Raw dir: {args.raw_root}")
    print(f"Output : {Path(args.output_dir) / BASE_DATASET_NAME}")
    print(f"Built H5 files: {len(built)}")
    print(f"dataset_info.json: {info_path}")
    print("Statistics:", json.dumps(builder.stats, ensure_ascii=False))
    print("=" * 72)


if __name__ == "__main__":
    main()
