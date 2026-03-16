"""Awakening (OpenNeuro ds005620) Dataset Builder.

Converts the Awakening BIDS (BrainVision) dataset into the EEG-FM benchmark HDF5
format: Subject -> Trial -> Segment.

Current strategy (team unified, per PI update):
- Include ALL EEG recordings under tasks:
  - task-awake_*  -> label 0 (awake)
  - task-sed*_*   -> label 1 (sedated)
  This includes stimulation recordings such as acq-tms. We do NOT do event-aligned
  epoching for stimulation; instead we apply a single unified sliding-window
  policy to all recordings.

- Segmenting: fixed sliding windows (configured via run_dataset.py)
  - window_sec=1.0, stride_sec=1.0 (non-overlapping)

- Preprocessing (team unified):
  - bandpass 0.1-75 Hz
  - notch 50 Hz
  - resample to 200 Hz

- QC (minimal):
  - reject segments with NaN/Inf
  - reject segments with max(|x|) > max_amplitude_uv (default 600 µV)

Notes
-----
1) The builder outputs one HDF5 file per subject: sub_<subject_id>.h5
2) Trial granularity: one recording file == one trial.
3) Segment time is relative to trial start (start_time/end_time in seconds).

Typical usage (direct):
    python awakening.py "Z:\\Datasets\\Awakening\\ds005620-download" \
        --output_dir "Z:\\Processed_datasets\\EEG_Bench\\Awakening"

Or (recommended) register AwakeningBuilder in your run_dataset.py DATASET_MAP.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import re

import numpy as np

try:
    import mne
    HAS_MNE = True
except Exception:
    mne = None
    HAS_MNE = False

try:
    # package relative imports
    from ..schema import SubjectAttrs, TrialAttrs, SegmentAttrs
    from ..hdf5_io import HDF5Writer
    from ..config import DatasetInfo, DatasetTaskType, DownstreamTaskType
    from ..utils import ElectrodeSet
except ImportError:
    # allow running as a standalone script
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from schema import SubjectAttrs, TrialAttrs, SegmentAttrs
    from hdf5_io import HDF5Writer
    from config import DatasetInfo, DatasetTaskType, DownstreamTaskType
    from utils import ElectrodeSet


AWAKENING_INFO = DatasetInfo(
    dataset_name="Awakening",
    task_type=DatasetTaskType.OTHER,  # consciousness / anesthesia; use OTHER unless you have a better enum
    downstream_task_type=DownstreamTaskType.CLASSIFICATION,
    num_labels=2,
    category_list=["awake", "sed2"],
    sampling_rate=200.0,  # target sampling rate for benchmark
    montage="10_10",
    channels=[],  # filled per subject after channel standardization
)


@dataclass(frozen=True)
class RecordingItem:
    """One BIDS recording (one BrainVision .vhdr)."""

    subject_id: str
    vhdr_path: Path
    task: str
    acq: str
    run: Optional[int]
    label: int  # 0 awake, 1 sed2
    session_id: int  # 0 awake, 1 sed2


def _safe_int(x: str) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


class AwakeningBuilder:
    """Builder for the Awakening dataset."""

    def __init__(
        self,
        raw_data_dir: str,
        output_dir: str,
        target_sfreq: float = 200.0,
        window_sec: float = 2.0,
        stride_sec: float = 2.0,
        filter_low: float = 0.1,
        filter_high: float = 75.0,
        filter_notch: float = 50.0,
        max_amplitude_uv: float = 600.0,
        include_tms: bool = False,
        skip_corrupt: bool = True,
        verbose: bool = True,
    ):
        if not HAS_MNE:
            raise ImportError("mne is required to build the Awakening dataset")

        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.target_sfreq = float(target_sfreq)
        self.window_sec = float(window_sec)
        self.stride_sec = float(stride_sec)
        self.filter_low = float(filter_low)
        self.filter_high = float(filter_high)
        self.filter_notch = float(filter_notch)
        self.max_amplitude_uv = float(max_amplitude_uv)
        self.include_tms = bool(include_tms)
        self.skip_corrupt = bool(skip_corrupt)
        self.verbose = bool(verbose)

        self.window_samples = int(round(self.window_sec * self.target_sfreq))
        self.stride_samples = int(round(self.stride_sec * self.target_sfreq))

        self.electrode_set = ElectrodeSet()

        # Build an index of available recordings per subject
        self._recordings: List[RecordingItem] = self._scan_bids_recordings()
        self._by_subject: Dict[str, List[RecordingItem]] = {}
        for rec in self._recordings:
            self._by_subject.setdefault(rec.subject_id, []).append(rec)

    # -------------------------
    # Scanning & parsing
    # -------------------------

    _FNAME_RE = re.compile(
        r"^sub-(?P<sub>[^_]+)_(?:ses-(?P<ses>[^_]+)_)?task-(?P<task>[^_]+)"
        r"(?:_acq-(?P<acq>[^_]+))?"
        r"(?:_run-(?P<run>\d+))?"
        r"_eeg\.vhdr$",
        flags=re.IGNORECASE,
    )

    def _scan_bids_recordings(self) -> List[RecordingItem]:
        """Scan for .vhdr files and select target recordings."""
        if not self.raw_data_dir.exists():
            raise FileNotFoundError(f"raw_data_dir not found: {self.raw_data_dir}")

        vhdr_files = list(self.raw_data_dir.rglob("*_eeg.vhdr"))
        if self.verbose:
            print(f"[Awakening] Scanning {self.raw_data_dir} ... found {len(vhdr_files)} .vhdr")

        # Updated inclusion rule (per PI): include ALL EEG recordings for
        # task-awake_* and task-sed*_* (including acq-tms), and apply a unified
        # fixed sliding-window segmentation to all.
        items: List[RecordingItem] = []
        skipped_badname = 0
        skipped_task = 0

        for p in vhdr_files:
            m = self._FNAME_RE.match(p.name)
            if not m:
                skipped_badname += 1
                continue

            sub = m.group("sub")
            task = (m.group("task") or "").lower()
            acq = (m.group("acq") or "").lower()
            run = _safe_int(m.group("run")) if m.group("run") else None

            if task == "awake":
                label = 0
                session_id = 0
            elif task.startswith("sed"):
                label = 1
                session_id = 1
            else:
                skipped_task += 1
                continue

            items.append(
                RecordingItem(
                    subject_id=sub,
                    vhdr_path=p,
                    task=task,
                    acq=acq,
                    run=run,
                    label=label,
                    session_id=session_id,
                )
            )

        # Deterministic order
        items.sort(key=lambda x: (x.subject_id, x.task, x.acq, x.run or -1, str(x.vhdr_path)))
        if self.verbose:
            print(
                f"[Awakening] Selected {len(items)} recordings after filtering. "
                f"(skipped_badname={skipped_badname}, skipped_task={skipped_task})"
            )
        return items

    def get_subject_ids(self) -> List[str]:
        """Return subject IDs (strings as in BIDS, e.g., '1010')."""
        return sorted(self._by_subject.keys())

    # -------------------------
    # Preprocess & QC
    # -------------------------

    def _read_raw_brainvision(self, vhdr_path: Path) -> "mne.io.BaseRaw":
        raw = mne.io.read_raw_brainvision(str(vhdr_path), preload=True, verbose=False)
        return raw

    def _standardize_channels(self, raw: "mne.io.BaseRaw") -> "mne.io.BaseRaw":
        """Keep EEG channels, standardize channel names, drop non-EEG by default."""
        # Keep EEG only (drop EOG by default, but you can change this if needed)
        try:
            raw.pick_types(eeg=True, eog=False, misc=False, stim=False, ecg=False, emg=False)
        except Exception:
            # fallback: pick all, but we'll keep as-is
            pass

        rename_map: Dict[str, str] = {}
        for ch in raw.ch_names:
            # BrainVision names often like 'Fp1' 'Fz' etc. Normalize to ElectrodeSet standard
            std = self.electrode_set.standardize_name(str(ch))
            rename_map[ch] = std
        try:
            raw.rename_channels(rename_map)
        except Exception:
            # If rename fails, keep original
            pass
        return raw

    def _preprocess(self, raw: "mne.io.BaseRaw") -> "mne.io.BaseRaw":
        # Notch
        if self.filter_notch and self.filter_notch > 0:
            try:
                raw.notch_filter(freqs=[self.filter_notch], verbose=False)
            except Exception:
                pass

        # Bandpass
        if self.filter_low is not None or self.filter_high is not None:
            try:
                raw.filter(l_freq=self.filter_low, h_freq=self.filter_high, verbose=False)
            except Exception:
                pass

        # Resample
        if abs(float(raw.info.get("sfreq", 0.0)) - self.target_sfreq) > 1e-6:
            raw.resample(self.target_sfreq, verbose=False)

        return raw

    def _to_uv(self, data: np.ndarray) -> np.ndarray:
        """Convert data to microvolts if it looks like Volts."""
        # MNE uses Volts for EEG. After raw.get_data() it is Volts.
        # Convert to µV.
        return data.astype(np.float32) * 1e6

    def _qc_segment(self, seg_uv: np.ndarray) -> Tuple[bool, str]:
        """Return (is_valid, reason_if_rejected)."""
        if not np.isfinite(seg_uv).all():
            return False, "nan_or_inf"
        if self.max_amplitude_uv is not None and self.max_amplitude_uv > 0:
            if float(np.abs(seg_uv).max()) > self.max_amplitude_uv:
                return False, "amplitude_exceed"
        return True, "ok"

    # -------------------------
    # Build
    # -------------------------

    def build_subject(self, subject_id: str) -> str:
        """Build a single subject HDF5 file."""
        recs = self._by_subject.get(subject_id)
        if not recs:
            raise ValueError(f"No recordings found for subject {subject_id}")

        # deterministic order inside subject
        recs = sorted(recs, key=lambda x: (x.task, x.acq, x.run or -1, str(x.vhdr_path)))

        out_path = self.output_dir / f"sub_{subject_id}.h5"
        if out_path.exists():
            out_path.unlink()

        # Stats
        stats = {
            "subject_id": subject_id,
            "trials_total": 0,
            "segments_total": 0,
            "segments_written": 0,
            "segments_rejected_nan_or_inf": 0,
            "segments_rejected_amplitude_exceed": 0,
            "label_counts": {"awake": 0, "sed2": 0},
        }

        # We determine channel list from first valid recording after standardization
        chn_name: Optional[List[str]] = None

        subject_attrs = SubjectAttrs(
            subject_id=subject_id,
            dataset_name=AWAKENING_INFO.dataset_name,
            task_type=AWAKENING_INFO.task_type.value,
            downstream_task_type=AWAKENING_INFO.downstream_task_type.value,
            rsFreq=self.target_sfreq,
            chn_name=[],  # fill after we know
            num_labels=AWAKENING_INFO.num_labels,
            category_list=AWAKENING_INFO.category_list,
            chn_pos=None,
            chn_ori=None,
            chn_type="EEG",
            montage=AWAKENING_INFO.montage,
        )

        with HDF5Writer(str(out_path), subject_attrs) as writer:
            global_segment_id = 0

            for trial_id, rec in enumerate(recs):
                stats["trials_total"] += 1

                # Read/preprocess
                try:
                    raw = self._read_raw_brainvision(rec.vhdr_path)
                except Exception as exc:
                    if self.verbose:
                        print(f"[Awakening] Skip {rec.vhdr_path.name}: read error: {exc}")
                    continue

                raw = self._standardize_channels(raw)
                raw = self._preprocess(raw)

                data_uv = self._to_uv(np.asarray(raw.get_data()))  # (C, T)

                if chn_name is None:
                    chn_name = list(raw.ch_names)
                    # update subject attrs in file root
                    writer.set_subject_channels(chn_name) if hasattr(writer, "set_subject_channels") else None
                    # Fallback: directly write root attrs if helper doesn't exist
                    try:
                        writer.h5.attrs["chn_name"] = np.array(chn_name, dtype="S")
                    except Exception:
                        pass

                # Ensure channel consistency across trials
                if chn_name is not None and list(raw.ch_names) != chn_name:
                    # Align by picking/reordering to first trial's channel order if possible
                    try:
                        raw.pick_channels(chn_name, ordered=True)
                        data_uv = self._to_uv(np.asarray(raw.get_data()))
                    except Exception:
                        if self.verbose:
                            print(f"[Awakening] Skip trial due to channel mismatch: {rec.vhdr_path.name}")
                        continue

                n_samples = data_uv.shape[1]

                # Create trial
                trial_attrs = TrialAttrs(
                    trial_id=trial_id,
                    session_id=rec.session_id,
                    task_name=f"{rec.task}_{rec.acq}" + (f"_run{rec.run}" if rec.run is not None else ""),
                )
                trial_name = writer.add_trial(trial_attrs)

                # Sliding windows
                for start in range(0, n_samples - self.window_samples + 1, self.stride_samples):
                    end = start + self.window_samples
                    seg = data_uv[:, start:end]

                    stats["segments_total"] += 1
                    ok, reason = self._qc_segment(seg)
                    if not ok:
                        if reason == "nan_or_inf":
                            stats["segments_rejected_nan_or_inf"] += 1
                        elif reason == "amplitude_exceed":
                            stats["segments_rejected_amplitude_exceed"] += 1
                        continue

                    # Label counts
                    if rec.label == 0:
                        stats["label_counts"]["awake"] += 1
                    else:
                        stats["label_counts"]["sed2"] += 1

                    seg_attrs = SegmentAttrs(
                        segment_id=global_segment_id,
                        start_time=float(start) / self.target_sfreq,
                        end_time=float(end) / self.target_sfreq,
                        time_length=self.window_sec,
                        label=np.array([rec.label], dtype=np.int64),
                    )
                    writer.add_segment(trial_name, seg_attrs, seg)
                    global_segment_id += 1
                    stats["segments_written"] += 1

        if self.verbose:
            print(
                f"[Awakening] Saved {out_path} | trials={stats['trials_total']} "
                f"segments_written={stats['segments_written']} (total={stats['segments_total']})"
            )

        return str(out_path)

    def build_all(self, subject_ids: Optional[List[str]] = None) -> List[str]:
        if subject_ids is None:
            subject_ids = self.get_subject_ids()

        output_paths: List[str] = []
        failed: List[str] = []

        all_stats = {
            "dataset": {
                "name": AWAKENING_INFO.dataset_name,
                "task_type": str(AWAKENING_INFO.task_type.value),
                "downstream_task": str(AWAKENING_INFO.downstream_task_type.value),
                "num_labels": AWAKENING_INFO.num_labels,
                "category_list": AWAKENING_INFO.category_list,
                "montage": AWAKENING_INFO.montage,
            },
            "processing": {
                "target_sampling_rate": self.target_sfreq,
                "window_sec": self.window_sec,
                "stride_sec": self.stride_sec,
                "filter_low": self.filter_low,
                "filter_high": self.filter_high,
                "filter_notch": self.filter_notch,
                "max_amplitude_uv": self.max_amplitude_uv,
                "include_tms": self.include_tms,
            },
            "statistics": {
                "total_subjects": len(subject_ids),
                "successful": 0,
                "failed": 0,
                "failed_subject_ids": [],
            },
            "generated_at": datetime.now().isoformat(),
        }

        for sid in subject_ids:
            try:
                out = self.build_subject(sid)
                output_paths.append(out)
            except Exception as exc:
                failed.append(sid)
                if self.verbose:
                    print(f"[Awakening] Error processing subject {sid}: {exc}")

        all_stats["statistics"]["successful"] = len(output_paths)
        all_stats["statistics"]["failed"] = len(failed)
        all_stats["statistics"]["failed_subject_ids"] = failed

        json_path = self.output_dir / "dataset_info.json"
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(all_stats, f, indent=2)
            if self.verbose:
                print(f"[Awakening] Wrote dataset_info.json -> {json_path}")
        except Exception as exc:
            if self.verbose:
                print(f"[Awakening] Failed to write dataset_info.json: {exc}")

        return output_paths


def build_awakening_dataset(
    raw_data_dir: str,
    output_dir: str,
    subject_ids: Optional[List[str]] = None,
    **kwargs,
) -> List[str]:
    builder = AwakeningBuilder(raw_data_dir=raw_data_dir, output_dir=output_dir, **kwargs)
    return builder.build_all(subject_ids)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build Awakening HDF5 dataset")
    parser.add_argument(
        "raw_data_dir",
        type=str,
        help="Raw BIDS directory (e.g., Z:\\Datasets\\Awakening\\ds005620-download)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="Z:\\Processed_datasets\\EEG_Bench\\Awakening",
        help="Output directory for HDF5 files",
    )
    parser.add_argument("--subjects", nargs="+", type=str, default=None, help="Subject IDs to process")
    parser.add_argument("--target_sfreq", type=float, default=200.0)
    parser.add_argument("--window_sec", type=float, default=2.0)
    parser.add_argument("--stride_sec", type=float, default=2.0)
    parser.add_argument("--filter_low", type=float, default=0.1)
    parser.add_argument("--filter_high", type=float, default=75.0)
    parser.add_argument("--filter_notch", type=float, default=50.0)
    parser.add_argument("--max_amplitude_uv", type=float, default=600.0)
    parser.add_argument("--include_tms", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()

    builder = AwakeningBuilder(
        raw_data_dir=args.raw_data_dir,
        output_dir=args.output_dir,
        target_sfreq=args.target_sfreq,
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        filter_low=args.filter_low,
        filter_high=args.filter_high,
        filter_notch=args.filter_notch,
        max_amplitude_uv=args.max_amplitude_uv,
        include_tms=args.include_tms,
        verbose=not args.quiet,
    )

    outs = builder.build_all(args.subjects)
    print(f"Done. Built {len(outs)} subjects.")
