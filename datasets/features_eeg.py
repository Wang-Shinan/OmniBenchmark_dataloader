"""Features-EEG (ds004357) Builder for EEG-FM benchmark (v1.3).

This builder is **standalone runnable** (CLI) and produces:
  - One HDF5 per subject
  - One dataset-level dataset_info.json in the dataset output directory

Why sliding_window (paper-aligned):
  Features-EEG is an RSVP stream where neural representations of visual
  features persist and overlap across successive stimuli. Therefore, forcing
  a 1-second segment to correspond to a single stimulus label is not faithful
  to the data-generating process. We instead treat the data as continuous
  signal for representation learning and segment using 1s/1s sliding windows.

Input expectation (BIDS):
  <RAW_DATA_DIR>/sub-XX/eeg/sub-XX_task-rsvp_eeg.vhdr (BrainVision)
  (events.tsv may exist but is not used to generate supervised labels).

Compliance:
  - CLI entrypoint via argparse
  - Supports all required preprocessing parameters (target_sfreq, filters,
    window_sec, stride_sec)
  - Explicit segmentation policy recorded in dataset_info.json
  - QC: reject NaN/Inf; reject amplitude > threshold (uV)
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import mne
    HAS_MNE = True
except Exception:
    HAS_MNE = False

# Support both "package" and "standalone file" usage
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


BUILDER_VERSION = "features_eeg_builder_v1.3"


FEATURES_EEG_INFO = DatasetInfo(
    dataset_name="FeaturesEEG",
    task_type=DatasetTaskType.COGNITIVE,
    downstream_task_type=DownstreamTaskType.CLASSIFICATION,
    num_labels=0,
    category_list=[],
    sampling_rate=1000.0,  # per BIDS sidecar in this dataset; builder will resample
    montage="unknown",  # dataset uses a 64-ish cap; keep explicit and avoid guessing
    channels=[],
)


def _safe_read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_subject_ids(raw_data_dir: Path) -> List[str]:
    """Return subject IDs in the form '01', '02', ..."""
    # Prefer participants.tsv if present (BIDS canonical)
    p_tsv = raw_data_dir / "participants.tsv"
    if p_tsv.exists():
        lines = [x.strip() for x in p_tsv.read_text(encoding="utf-8", errors="ignore").splitlines() if x.strip()]
        if len(lines) >= 2:
            header = lines[0].split("\t")
            if header and header[0] == "participant_id":
                ids: List[str] = []
                for row in lines[1:]:
                    cols = row.split("\t")
                    if not cols:
                        continue
                    pid = cols[0]
                    if pid.startswith("sub-"):
                        ids.append(pid.replace("sub-", ""))
                if ids:
                    return sorted(ids)

    # Fallback: scan sub-* directories
    ids = []
    for p in raw_data_dir.glob("sub-*"):
        if p.is_dir():
            sid = p.name.replace("sub-", "")
            if sid:
                ids.append(sid)
    return sorted(set(ids))


def _find_vhdr(raw_data_dir: Path, subject_id: str) -> Path:
    # common BIDS location
    eeg_dir = raw_data_dir / f"sub-{subject_id}" / "eeg"
    if eeg_dir.exists():
        # prefer task-rsvp file
        cand = list(eeg_dir.glob(f"sub-{subject_id}_task-rsvp*_eeg.vhdr"))
        if cand:
            return sorted(cand)[0]
        # fallback: any vhdr
        cand = list(eeg_dir.glob("*.vhdr"))
        if cand:
            return sorted(cand)[0]

    # fallback: any match under subject directory
    cand = list((raw_data_dir / f"sub-{subject_id}").rglob("*.vhdr"))
    if not cand:
        raise FileNotFoundError(f"No .vhdr found for sub-{subject_id} under {raw_data_dir}")
    return sorted(cand)[0]


def _pick_eeg_only(raw: "mne.io.BaseRaw") -> "mne.io.BaseRaw":
    # Prefer MNE channel type info
    try:
        raw = raw.copy().pick_types(eeg=True, eog=False, ecg=False, emg=False, misc=False, stim=False)
        return raw
    except Exception:
        return raw


def _segment_sliding_window(
    raw: "mne.io.BaseRaw",
    window_sec: float,
    stride_sec: float,
) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
    """Return (X, time_spans) where X is (n_seg, n_ch, n_t) in Volts."""
    overlap = max(window_sec - stride_sec, 0.0)
    epochs = mne.make_fixed_length_epochs(
        raw,
        duration=float(window_sec),
        overlap=float(overlap),
        preload=True,
        reject_by_annotation=False,
        verbose=False,
    )
    X = epochs.get_data()  # (n_epochs, n_ch, n_t)
    # Construct (start,end) relative to recording start (sec)
    spans: List[Tuple[float, float]] = []
    for i in range(X.shape[0]):
        start_t = float(i * stride_sec)
        spans.append((start_t, start_t + float(window_sec)))
    return X, spans


class FeaturesEEGBuilder:
    """Builder for Features-EEG (ds004357)."""

    def __init__(
        self,
        raw_data_dir: str,
        output_dir: str,
        subjects: Optional[List[str]] = None,
        # required project-wide preprocessing knobs
        target_sfreq: float = 200.0,
        window_sec: float = 1.0,
        stride_sec: float = 1.0,
        filter_low: float = 0.1,
        filter_high: float = 75.0,
        filter_notch: float = 50.0,
        # QC
        max_amplitude_uv: float = 600.0,
    ):
        if not HAS_MNE:
            raise ImportError("MNE is required for FeaturesEEGBuilder")

        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir) / FEATURES_EEG_INFO.dataset_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.target_sfreq = float(target_sfreq)
        self.window_sec = float(window_sec)
        self.stride_sec = float(stride_sec)
        self.filter_low = float(filter_low)
        self.filter_high = float(filter_high)
        self.filter_notch = float(filter_notch)
        self.max_amplitude_uv = float(max_amplitude_uv)

        all_ids = _load_subject_ids(self.raw_data_dir)
        if subjects:
            # allow passing "sub-01" or "01"
            norm = [s.replace("sub-", "") for s in subjects]
            self.subject_ids = [s for s in all_ids if s in set(norm)]
        else:
            self.subject_ids = all_ids

        # dataset-level counters
        self._counts: Dict[str, Any] = {
            "total_subjects": len(all_ids),
            "processed_subjects": 0,
            "subjects_with_errors": 0,
            "segments_total": 0,
            "segments_valid": 0,
            "segments_rejected_naninf": 0,
            "segments_rejected_amp": 0,
        }

    def get_subject_ids(self) -> List[str]:
        return list(self.subject_ids)

    def _preprocess(self, raw: "mne.io.BaseRaw") -> "mne.io.BaseRaw":
        # notch first (recommended for stable bandpass)
        if self.filter_notch and self.filter_notch > 0:
            raw = raw.copy().notch_filter(freqs=[self.filter_notch], verbose=False)
        raw = raw.copy().filter(l_freq=self.filter_low, h_freq=self.filter_high, verbose=False)
        if raw.info["sfreq"] != self.target_sfreq:
            raw = raw.copy().resample(self.target_sfreq, verbose=False)
        return raw

    def build_subject(self, subject_id: str) -> str:
        vhdr_path = _find_vhdr(self.raw_data_dir, subject_id)
        out_file = self.output_dir / f"sub_{subject_id}.h5"
        if out_file.exists():
            return str(out_file)

        raw = mne.io.read_raw_brainvision(str(vhdr_path), preload=True, verbose=False)
        raw = _pick_eeg_only(raw)
        raw = self._preprocess(raw)

        # Segment in Volts, then convert to microvolts for storage
        X_v, spans = _segment_sliding_window(raw, self.window_sec, self.stride_sec)
        X_uv = X_v * 1e6  # V -> uV

        # Subject attributes
        subject_attrs = SubjectAttrs(
            subject_id=subject_id,
            dataset_name=FEATURES_EEG_INFO.dataset_name,
            task_type=FEATURES_EEG_INFO.task_type.value,
            downstream_task_type=FEATURES_EEG_INFO.downstream_task_type.value,
            rsFreq=float(self.target_sfreq),
            chn_name=list(raw.ch_names),
            num_labels=0,
            category_list=[],
            chn_type="EEG",
            montage=FEATURES_EEG_INFO.montage,
        )

        # Write HDF5
        with HDF5Writer(str(out_file), subject_attrs) as writer:
            trial_name = writer.add_trial(TrialAttrs(trial_id=0, session_id=0, task_name="rsvp_stream"))

            seg_id = 0
            for i in range(X_uv.shape[0]):
                self._counts["segments_total"] += 1

                seg = X_uv[i]
                if not np.isfinite(seg).all():
                    self._counts["segments_rejected_naninf"] += 1
                    continue

                if float(np.max(np.abs(seg))) > self.max_amplitude_uv:
                    self._counts["segments_rejected_amp"] += 1
                    continue

                start_t, end_t = spans[i]
                seg_attrs = SegmentAttrs(
                    segment_id=seg_id,
                    start_time=float(start_t),
                    end_time=float(end_t),
                    time_length=float(self.window_sec),
                    label=np.array([], dtype=np.int64),  # unlabeled by design
                    task_label="",
                )
                writer.add_segment(trial_name, seg_attrs, seg)
                seg_id += 1
                self._counts["segments_valid"] += 1

        if self._counts["segments_valid"] <= 0:
            raise RuntimeError(
                f"No valid segments produced for sub-{subject_id}. "
                f"Try adjusting QC threshold or check raw data integrity."
            )

        self._counts["processed_subjects"] += 1
        return str(out_file)

    def write_dataset_info(self) -> str:
        """Write dataset_info.json in output directory."""
        ds_desc = _safe_read_json(self.raw_data_dir / "dataset_description.json")
        eeg_json = _safe_read_json(self.raw_data_dir / "task-rsvp_eeg.json")

        info: Dict[str, Any] = {
            "builder_version": BUILDER_VERSION,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "dataset_name": FEATURES_EEG_INFO.dataset_name,
            "bids_dataset": True,
            "raw_data_dir": str(self.raw_data_dir),
            "paper_rationale": (
                "RSVP stream with overlapping feature representations across successive stimuli; "
                "therefore, we avoid forcing 1-second segments to map to single stimulus labels and "
                "use sliding-window segments for representation learning."
            ),
            "source": {
                "dataset_description": ds_desc or {},
                "task_sidecar": eeg_json or {},
            },
            "task": {
                "task_type": FEATURES_EEG_INFO.task_type.value,
                "downstream_task_type": FEATURES_EEG_INFO.downstream_task_type.value,
                "labels": {
                    "num_labels": 0,
                    "category_list": [],
                    "label_policy": "unlabeled",
                    "notes": (
                        "events.tsv contains rich per-stimulus annotations (e.g., stimnumber, "
                        "feature conditions, speed condition). This builder does not convert them "
                        "to supervised labels because 1s segments inherently contain overlapping "
                        "responses to multiple stimuli in RSVP streams."
                    ),
                },
            },
            "preprocessing": {
                "target_sfreq": self.target_sfreq,
                "filter_low": self.filter_low,
                "filter_high": self.filter_high,
                "filter_notch": self.filter_notch,
                "units": "uV",
            },
            "segment_policy": "sliding_window",
            "segment_parameters": {
                "window_sec": self.window_sec,
                "stride_sec": self.stride_sec,
                "alignment": None,
                "notes": "Full-coverage fixed-length windows on continuous RSVP recordings.",
            },
            "qc": {
                "max_amplitude_uv": self.max_amplitude_uv,
                "reject_nan_inf": True,
                "counts": {
                    "segments_total": int(self._counts["segments_total"]),
                    "segments_valid": int(self._counts["segments_valid"]),
                    "segments_rejected_naninf": int(self._counts["segments_rejected_naninf"]),
                    "segments_rejected_amp": int(self._counts["segments_rejected_amp"]),
                },
            },
            "subjects": {
                "total_subjects": int(self._counts["total_subjects"]),
                "requested_subjects": len(self.subject_ids),
            },
        }

        out_path = self.output_dir / "dataset_info.json"
        out_path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
        return str(out_path)


def _build_cli_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build Features-EEG (ds004357) into EEG-FM HDF5 format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("raw_data_dir", type=str, help="BIDS root directory of Features-EEG")
    p.add_argument("--output_dir", type=str, required=True, help="Output root directory")
    p.add_argument("--subjects", type=str, nargs="+", default=None, help="Specific subjects (e.g., 01 02 or sub-01)")

    # Required preprocessing knobs (v1.3)
    p.add_argument("--target_sfreq", type=float, default=200.0)
    p.add_argument("--filter_low", type=float, default=0.1)
    p.add_argument("--filter_high", type=float, default=75.0)
    p.add_argument("--filter_notch", type=float, default=50.0)
    p.add_argument("--window_sec", type=float, default=1.0)
    p.add_argument("--stride_sec", type=float, default=1.0)

    # QC
    p.add_argument("--max_amplitude_uv", type=float, default=600.0, help="Reject segments with max|amplitude| > threshold (uV)")

    return p


def main() -> None:
    args = _build_cli_parser().parse_args()

    builder = FeaturesEEGBuilder(
        raw_data_dir=args.raw_data_dir,
        output_dir=args.output_dir,
        subjects=args.subjects,
        target_sfreq=args.target_sfreq,
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        filter_low=args.filter_low,
        filter_high=args.filter_high,
        filter_notch=args.filter_notch,
        max_amplitude_uv=args.max_amplitude_uv,
    )

    subject_ids = builder.get_subject_ids()
    print("=" * 72)
    print(f"Dataset: {FEATURES_EEG_INFO.dataset_name} (Features-EEG / ds004357)")
    print(f"Raw dir: {args.raw_data_dir}")
    print(f"Output dir: {builder.output_dir}")
    print(f"Subjects: {len(subject_ids)}")
    print(
        "Preproc: bandpass "
        f"{builder.filter_low}-{builder.filter_high} Hz, "
        f"notch {builder.filter_notch} Hz, "
        f"target_sfreq={builder.target_sfreq} Hz"
    )
    print(f"Segmentation: sliding_window window={builder.window_sec}s stride={builder.stride_sec}s")
    print(f"QC: max_amplitude_uv={builder.max_amplitude_uv}")
    print("=" * 72)

    errors = 0
    for i, sid in enumerate(subject_ids, 1):
        print(f"[{i}/{len(subject_ids)}] Processing sub-{sid} ...")
        try:
            out = builder.build_subject(sid)
            print(f"  ✅ {out}")
        except Exception as e:
            errors += 1
            builder._counts["subjects_with_errors"] += 1
            print(f"  ❌ Error processing sub-{sid}: {e}")

    info_path = builder.write_dataset_info()
    print("\n" + "-" * 72)
    print(f"Wrote dataset_info.json: {info_path}")
    print(
        "Segments: total={t}, valid={v}, reject_naninf={n}, reject_amp={a}".format(
            t=builder._counts["segments_total"],
            v=builder._counts["segments_valid"],
            n=builder._counts["segments_rejected_naninf"],
            a=builder._counts["segments_rejected_amp"],
        )
    )
    print(f"Subjects with errors: {errors}")

    if builder._counts["segments_valid"] <= 0:
        raise SystemExit("No valid segments produced for the dataset. Aborting.")

    if errors > 0:
        raise SystemExit(f"Completed with {errors} subject-level errors.")


if __name__ == "__main__":
    main()
