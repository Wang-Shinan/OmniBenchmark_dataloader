"""
FBCCA-DW (dataset5) EEG-FM Builder — v1.4 compliant
===================================================

Fix note (v1.4.1):
- Avoid MNE FIR warnings on short epochs by preprocessing the *entire recording* per .mat file
  (notch + bandpass + resample), then slicing stimulus segments in resampled time.
  This eliminates:
    RuntimeWarning: filter_length (...) is longer than the signal (...)

Dataset facts (verified)
------------------------
- Each .mat contains `data` with shape (65, N): 64 EEG channels + 1 trigger channel.
- Trigger codes include: 0, 1..40 (target id), 251 (cycle boundary / stimulus end), 252, 253.
- Let idx_end = indices where trigger == 251.
  For each interval [idx_end[i], idx_end[i+1]):
    - Exactly ONE target code in [1..40] occurs (deterministic label).
    - FIRST occurrence of that target marks stimulus onset.
    - target_onset -> next 251 ≈ 3 seconds (stimulus duration).
    - Full cycle ≈ 6 seconds (includes non-stimulus).

Segmentation strategy (kept as decided)
---------------------------------------
Policy: within_event_sliding_window
- Event = stimulus segment [target_onset, next_251).
- Windows: fixed 1s, stride 1s, within stimulus only.
- Labels: 40-way classification, label read from trigger (never by order).
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import mne
from scipy.io import loadmat

# ---- project imports (preferred) with standalone fallback ----
try:
    from ..schema import SubjectAttrs, TrialAttrs, SegmentAttrs
    from ..hdf5_io import HDF5Writer
    from ..config import DatasetInfo, DatasetTaskType, DownstreamTaskType, PreprocConfig
except Exception:
    THIS = Path(__file__).resolve()
    sys.path.insert(0, str(THIS.parent.parent))
    from schema import SubjectAttrs, TrialAttrs, SegmentAttrs
    from hdf5_io import HDF5Writer
    from config import DatasetInfo, DatasetTaskType, DownstreamTaskType, PreprocConfig


# -----------------------------
# Mandatory constants (v1.4)
# -----------------------------
BUILDER_VERSION = "fbcca_dw_builder_v1.4.1"

FBCCA_DW_INFO = DatasetInfo(
    dataset_name="FBCCA-DW",
    task_type=DatasetTaskType.COGNITIVE,
    downstream_task_type=DownstreamTaskType.CLASSIFICATION,
    num_labels=40,
    category_list=[str(i) for i in range(1, 41)],
    sampling_rate=250.0,   # raw sampling rate (Hz)
    montage="10_20",
    channels=[f"EEG{i}" for i in range(1, 65)],  # placeholder names
)

DEFAULT_MAX_AMPLITUDE_UV = 600.0


def _infer_block_id(filename: str) -> int:
    m = re.search(r"block(\d+)", filename)
    return int(m.group(1)) if m else 0


def _load_mat(mat_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Return eeg_uv (64, N) float64 and trig (N,) int32."""
    mat = loadmat(str(mat_path))
    if "data" not in mat:
        raise KeyError(f"'data' not found in: {mat_path}")
    data = mat["data"]
    if data.ndim != 2 or data.shape[0] != 65:
        raise ValueError(f"Unexpected data shape {data.shape} in {mat_path} (expected 65 x N)")
    eeg_uv = data[:-1, :].astype(np.float64, copy=False)
    trig = data[-1, :].astype(np.int32, copy=False).reshape(-1)
    return eeg_uv, trig


def _preprocess_full_recording_uv(
    eeg_uv: np.ndarray,
    sfreq_in: float,
    preproc: PreprocConfig,
) -> np.ndarray:
    """
    Preprocess the whole recording (µV -> V -> µV) to avoid FIR length warnings on short epochs.
    Output: (C, T_resampled) float32 in µV at preproc.target_sfreq.
    """
    eeg_v = eeg_uv * 1e-6
    info = mne.create_info(
        ch_names=[f"EEG{i}" for i in range(1, eeg_v.shape[0] + 1)],
        sfreq=sfreq_in,
        ch_types=["eeg"] * eeg_v.shape[0],
    )
    raw = mne.io.RawArray(eeg_v, info, verbose=False)

    # notch -> bandpass (group convention)
    if preproc.filter_notch and preproc.filter_notch > 0:
        raw.notch_filter(freqs=[preproc.filter_notch], verbose=False)
    raw.filter(l_freq=preproc.filter_low, h_freq=preproc.filter_high, verbose=False)

    if sfreq_in != preproc.target_sfreq:
        raw.resample(preproc.target_sfreq, verbose=False)

    return (raw.get_data() * 1e6).astype(np.float32)


def _segment_within_stimulus(
    eeg_uv: np.ndarray,
    preproc: PreprocConfig,
) -> List[Tuple[int, int, np.ndarray]]:
    """
    Fixed windows within stimulus segment.
    Returns list of (start_sample, end_sample, seg_uv).
    """
    fs = float(preproc.target_sfreq)
    win = int(round(preproc.window_sec * fs))
    stride = int(round(preproc.stride_sec * fs))
    if win <= 0 or stride <= 0:
        raise ValueError("window_sec and stride_sec must be positive")

    T = eeg_uv.shape[1]
    out = []
    for s in range(0, T - win + 1, stride):
        e = s + win
        seg = eeg_uv[:, s:e]
        if seg.shape[1] == win:
            out.append((s, e, seg))
    return out


def _qc_segment(seg_uv: np.ndarray, max_amp_uv: float) -> Tuple[bool, str]:
    """QC per v1.4."""
    if not np.isfinite(seg_uv).all():
        return False, "nan_or_inf"
    if np.max(np.abs(seg_uv)) > max_amp_uv:
        return False, "amplitude_exceed"
    return True, ""


class FBCCADWBuilder:
    def __init__(
        self,
        raw_data_dir: Path,
        output_dir: Path,
        preproc: PreprocConfig,
        max_amplitude_uv: float = DEFAULT_MAX_AMPLITUDE_UV,
    ):
        self.raw_data_dir = Path(raw_data_dir)
        self.dataset_out_dir = Path(output_dir) / FBCCA_DW_INFO.dataset_name
        self.dataset_out_dir.mkdir(parents=True, exist_ok=True)

        self.preproc = preproc
        self.max_amplitude_uv = float(max_amplitude_uv)

        # verified raw sfreq
        self.raw_sfreq = 250.0

    def list_subjects(self) -> List[str]:
        if not self.raw_data_dir.exists():
            raise FileNotFoundError(f"Raw data dir not found: {self.raw_data_dir}")
        return sorted([p.name for p in self.raw_data_dir.iterdir() if p.is_dir()])

    def _mat_files(self, subject_id: str) -> List[Path]:
        subject_dir = self.raw_data_dir / subject_id
        return sorted(subject_dir.glob("*.mat"))

    def build_subject(self, subject_id: str) -> Tuple[str, Dict]:
        mats = self._mat_files(subject_id)
        if len(mats) == 0:
            raise FileNotFoundError(f"No .mat files for subject {subject_id} under {self.raw_data_dir}")

        subject_attrs = SubjectAttrs(
            subject_id=subject_id,
            dataset_name=FBCCA_DW_INFO.dataset_name,
            task_type=str(FBCCA_DW_INFO.task_type.value),
            downstream_task_type=str(FBCCA_DW_INFO.downstream_task_type.value),
            rsFreq=float(self.preproc.target_sfreq),
            chn_name=[f"EEG{i}" for i in range(1, 65)],
            num_labels=int(FBCCA_DW_INFO.num_labels),
            category_list=FBCCA_DW_INFO.category_list,
            montage=FBCCA_DW_INFO.montage,
        )

        out_h5 = self.dataset_out_dir / f"{subject_id}.h5"

        stats = {
            "subject_id": subject_id,
            "h5_path": str(out_h5),
            "trials_total": 0,
            "segments_total": 0,
            "segments_written": 0,
            "segments_rejected": 0,
            "reject_reasons": {"nan_or_inf": 0, "amplitude_exceed": 0},
            "files_processed": 0,
            "files_skipped": 0,
        }

        with HDF5Writer(str(out_h5), subject_attrs) as writer:
            trial_id = 0

            for mat_path in mats:
                try:
                    eeg_uv_raw, trig = _load_mat(mat_path)
                except Exception:
                    stats["files_skipped"] += 1
                    continue

                stats["files_processed"] += 1

                # Preprocess full recording ONCE (avoids short-epoch FIR warning)
                eeg_uv_proc = _preprocess_full_recording_uv(
                    eeg_uv_raw, self.raw_sfreq, self.preproc
                )

                idx_end = np.where(trig == 251)[0]
                if idx_end.size < 2:
                    continue

                block_id = _infer_block_id(mat_path.name)

                # Map raw sample index -> resampled sample index via time
                # idx_rs = round(idx_raw / raw_sfreq * target_sfreq)
                tf = float(self.preproc.target_sfreq) / float(self.raw_sfreq)

                for i in range(idx_end.size - 1):
                    a = int(idx_end[i])
                    b = int(idx_end[i + 1])

                    seg_trig = trig[a:b]
                    targets = np.unique(seg_trig[(seg_trig >= 1) & (seg_trig <= 40)])
                    if targets.size != 1:
                        continue

                    label = int(targets[0])

                    rel = int(np.where(seg_trig == label)[0][0])
                    stim_start_raw = a + rel
                    stim_end_raw = b

                    # Convert to resampled indices
                    stim_start_rs = int(round(stim_start_raw * tf))
                    stim_end_rs = int(round(stim_end_raw * tf))

                    # Clamp to bounds
                    stim_start_rs = max(0, min(stim_start_rs, eeg_uv_proc.shape[1]))
                    stim_end_rs = max(0, min(stim_end_rs, eeg_uv_proc.shape[1]))

                    if stim_end_rs - stim_start_rs <= 0:
                        continue

                    stim_eeg_uv = eeg_uv_proc[:, stim_start_rs:stim_end_rs]

                    segments = _segment_within_stimulus(stim_eeg_uv, self.preproc)
                    if len(segments) == 0:
                        continue

                    trial_name = writer.add_trial(
                        TrialAttrs(trial_id=trial_id, session_id=block_id, task_name="SSVEP_40class")
                    )
                    stats["trials_total"] += 1

                    seg_id = 0
                    for (s_samp, e_samp, seg_uv) in segments:
                        stats["segments_total"] += 1
                        seg_uv = (seg_uv / 1000.0).astype(np.float32, copy=False)
                        ok, reason = _qc_segment(seg_uv, self.max_amplitude_uv)
                        if not ok:
                            stats["segments_rejected"] += 1
                            stats["reject_reasons"][reason] = stats["reject_reasons"].get(reason, 0) + 1
                            continue

                        start_time = float(s_samp) / float(self.preproc.target_sfreq)
                        end_time = float(e_samp) / float(self.preproc.target_sfreq)

                        seg_attrs = SegmentAttrs(
                            segment_id=seg_id,
                            start_time=start_time,
                            end_time=end_time,
                            time_length=float(self.preproc.window_sec),
                            label=np.asarray([label], dtype=np.int64),
                            task_label=str(label),
                        )
                        writer.add_segment(trial_name, seg_attrs, seg_uv)
                        stats["segments_written"] += 1
                        seg_id += 1

                    trial_id += 1

                # free memory between mats
                del eeg_uv_raw, trig, eeg_uv_proc
                gc.collect()

        if stats["segments_written"] == 0:
            raise RuntimeError(
                f"[FBCCA-DW] Subject {subject_id}: produced 0 valid segments. "
                f"Try increasing --max_amplitude_uv or inspect raw scaling for this subject."
            )

        return str(out_h5), stats

    def build_all(self, subject_ids: Optional[List[str]] = None) -> List[str]:
        if subject_ids is None:
            subject_ids = self.list_subjects()

        print("=" * 60)
        print(f"Dataset: {FBCCA_DW_INFO.dataset_name}")
        print(f"Raw data dir: {self.raw_data_dir}")
        print(f"Output dir: {self.dataset_out_dir}")
        print(f"Subjects: {len(subject_ids)}")
        print(
            f"Preproc: target_sfreq={self.preproc.target_sfreq}, window_sec={self.preproc.window_sec}, "
            f"stride_sec={self.preproc.stride_sec}, filter=[{self.preproc.filter_low}-{self.preproc.filter_high}]Hz, "
            f"notch={self.preproc.filter_notch}Hz"
        )
        print(f"QC: max_amplitude_uv={self.max_amplitude_uv} µV")
        print("=" * 60)

        outputs: List[str] = []
        failed: List[str] = []
        per_subject_stats: List[Dict] = []

        agg = {
            "trials_total": 0,
            "segments_total": 0,
            "segments_written": 0,
            "segments_rejected": 0,
            "reject_reasons": {"nan_or_inf": 0, "amplitude_exceed": 0},
        }

        for sid in subject_ids:
            try:
                print(f"[FBCCA-DW] Processing subject: {sid}")
                out_path, stats = self.build_subject(sid)
                outputs.append(out_path)
                per_subject_stats.append(stats)

                agg["trials_total"] += stats["trials_total"]
                agg["segments_total"] += stats["segments_total"]
                agg["segments_written"] += stats["segments_written"]
                agg["segments_rejected"] += stats["segments_rejected"]
                for k, v in stats["reject_reasons"].items():
                    agg["reject_reasons"][k] = agg["reject_reasons"].get(k, 0) + int(v)

                print(
                    f"  trials={stats['trials_total']} segments_written={stats['segments_written']} "
                    f"rejected={stats['segments_rejected']} reasons={stats['reject_reasons']}"
                )
            except Exception as e:
                print(f"  ❌ Failed subject {sid}: {e}")
                failed.append(sid)

        self._write_dataset_info(
            subject_ids=subject_ids,
            outputs=outputs,
            failed=failed,
            aggregate=agg,
            per_subject_stats=per_subject_stats,
        )

        if len(outputs) == 0:
            raise RuntimeError("No subjects processed successfully.")

        print("=" * 60)
        print(
            f"[FBCCA-DW] DONE. subjects_ok={len(outputs)}/{len(subject_ids)} "
            f"segments_written={agg['segments_written']} rejected={agg['segments_rejected']}"
        )
        print("=" * 60)

        return outputs

    def _write_dataset_info(
        self,
        subject_ids: List[str],
        outputs: List[str],
        failed: List[str],
        aggregate: Dict,
        per_subject_stats: List[Dict],
    ) -> None:
        info = {
            "dataset_name": FBCCA_DW_INFO.dataset_name,
            "builder_version": BUILDER_VERSION,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "raw_data_dir": str(self.raw_data_dir),
            "output_dir": str(self.dataset_out_dir),
            "task_type": str(FBCCA_DW_INFO.task_type.value),
            "downstream_task_type": str(FBCCA_DW_INFO.downstream_task_type.value),
            "subjects": {
                "requested": subject_ids,
                "succeeded": [Path(p).stem for p in outputs],
                "failed": failed,
            },
            "preprocessing": {
                "target_sfreq": float(self.preproc.target_sfreq),
                "filter_low": float(self.preproc.filter_low),
                "filter_high": float(self.preproc.filter_high),
                "filter_notch": float(self.preproc.filter_notch),
                "window_sec": float(self.preproc.window_sec),
                "stride_sec": float(self.preproc.stride_sec),
                "notes": "Whole-recording preprocessing per .mat file (notch+bandpass+resample) then slice stimulus by trigger time; avoids short-epoch FIR filter_length warnings.",
            },
            "segmentation": {
                "policy": "within_event_sliding_window",
                "parameters": {
                    "window_sec": float(self.preproc.window_sec),
                    "stride_sec": float(self.preproc.stride_sec),
                    "alignment": "stimulus_onset",
                    "event_source": "trigger_channel (target code onset; boundary code 251)",
                    "notes": "Stimulus segment = [first target code in (251_i,251_{i+1}) interval, next 251). Windows are generated inside this segment only. Tail shorter than one window is dropped.",
                },
                "rationale": "SSVEP label is valid during stimulus only; restricting to stimulus avoids baseline mixing and matches benchmark intent.",
            },
            "labeling": {
                "label_policy": "supervised",
                "num_labels": int(FBCCA_DW_INFO.num_labels),
                "categories": FBCCA_DW_INFO.category_list,
                "mapping": "label = unique target code in [1..40] within each (251_i,251_{i+1}) interval",
                "notes": "Blocks may repeat targets and miss some targets (verified). Do NOT infer label from trial order.",
            },
            "signals": {
                "unit": "uV",
                "stored_dtype": "float32",
                "n_channels": 64,
                "channels": [f"EEG{i}" for i in range(1, 65)],
                "segment_shape": "[C, T]",
            },
            "qc": {
                "thresholds": {"max_amplitude_uv": float(self.max_amplitude_uv)},
                "counts": {
                    "trials_total": int(aggregate["trials_total"]),
                    "segments_total": int(aggregate["segments_total"]),
                    "segments_written": int(aggregate["segments_written"]),
                    "segments_rejected": int(aggregate["segments_rejected"]),
                    "rejected_by_reason": aggregate.get("reject_reasons", {}),
                },
            },
            "sources": {
                "original_paper_doi": "10.1142/S0129065718500284",
                "notes": "Stimulus duration (~3s) verified from trigger timing: target onset -> next 251.",
            },
            "run_summary": {
                "n_subjects_requested": len(subject_ids),
                "n_subjects_succeeded": len(outputs),
                "n_subjects_failed": len(failed),
            },
            "per_subject_stats": per_subject_stats,
        }

        json_path = self.dataset_out_dir / "dataset_info.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)

        print(f"[FBCCA-DW] Wrote dataset_info.json -> {json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="FBCCA-DW builder (EEG-FM v1.4 compliant)")
    parser.add_argument("raw_data_dir", type=str, help="Raw FBCCA-DW root directory (contains S1, S2, ...)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output base directory")
    parser.add_argument("--subjects", type=str, nargs="*", default=None, help="Optional subject IDs (e.g., S1 S2)")

    # Unified preprocessing params
    parser.add_argument("--target_sfreq", type=float, default=200.0)
    parser.add_argument("--window_sec", type=float, default=1.0)
    parser.add_argument("--stride_sec", type=float, default=1.0)
    parser.add_argument("--filter_low", type=float, default=0.1)
    parser.add_argument("--filter_high", type=float, default=75.0)
    parser.add_argument("--filter_notch", type=float, default=50.0)

    # QC
    parser.add_argument("--max_amplitude_uv", type=float, default=DEFAULT_MAX_AMPLITUDE_UV)

    args = parser.parse_args()

    preproc = PreprocConfig(
        filter_low=float(args.filter_low),
        filter_high=float(args.filter_high),
        filter_notch=float(args.filter_notch),
        target_sfreq=float(args.target_sfreq),
        window_sec=float(args.window_sec),
        stride_sec=float(args.stride_sec),
        output_dir=str(args.output_dir),
        num_workers=1,
    )

    builder = FBCCADWBuilder(
        raw_data_dir=Path(args.raw_data_dir),
        output_dir=Path(args.output_dir),
        preproc=preproc,
        max_amplitude_uv=float(args.max_amplitude_uv),
    )

    subject_ids = builder.list_subjects() if args.subjects is None else args.subjects
    builder.build_all(subject_ids)


if __name__ == "__main__":
    main()
