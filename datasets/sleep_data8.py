"""SleepData8 Builder (EDF + hyp EDF) — EEG-FM Dataset Builder Specification v1.0.

Local folder layout (keep unchanged):
  - PSG:        <stem>.edf
  - Hypnogram:  <stem>_hyp.edf
  - RECORDS (optional): each line is PSG filename, e.g. sc4002e0.edf

This builder writes one HDF5 per subject and one trial per subject (one night),
segmented into fixed 30s epochs for sleep staging (5-class: W, N1, N2, N3, R).

Spec v1.0 compliance highlights:
  - SubjectAttrs.dataset_name is a *dataset identifier*: "<dataset_name>:<experiment_name>".
  - Output directory uses the recommended pattern: <output_dir>/<dataset_name>__<experiment_name>/
  - At least one QC rule is applied (NaN/Inf, amplitude threshold), with statistics exported.
  - dataset_info.json is generated in the output directory when finalize() is called.

Notes:
  - We intentionally do NOT infer meanings of stems like e0/j0.
  - We keep the benchmark preprocessing convention: bandpass up to 75 Hz.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import json
import numpy as np

try:
    import mne
    HAS_MNE = True
except ImportError:
    HAS_MNE = False

# Allow running as package or as flat script in project root
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


BASE_DATASET_NAME = "SleepData8"

SLEEPDATA8_INFO = DatasetInfo(
    dataset_name=BASE_DATASET_NAME,
    task_type=DatasetTaskType.SLEEP,
    downstream_task_type=DownstreamTaskType.CLASSIFICATION,
    num_labels=5,
    category_list=["W", "N1", "N2", "N3", "R"],
    sampling_rate=100.0,
    montage="standard_1020",
    channels=["FPZ", "PZ"],
)

# Hypnogram numeric code mapping (Sleep-EDF-style EDF hypnogram)
# 0: W, 1: N1, 2: N2, 3: N3, 4: Stage4->N3, 5: REM, 6: MT(ignore), 9: ?(ignore)
CODE2CLASS = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 4}
IGNORE_CODES = {6, 9}


@dataclass
class SubjectBuildStats:
    subject_id: str
    kept_segments: int
    dropped_segments: int
    drop_reasons: Dict[str, int]
    label_hist: Dict[int, int]


def _compute_experiment_name(
    *,
    target_sfreq: float,
    window_sec: float,
    stride_sec: float,
    filter_low: float,
    filter_high: float,
    filter_notch: float,
    channels: List[str],
    qc_abs_uv: float,
) -> str:
    """Create a compact, reproducible experiment_name string.

    Spec requires dataset identifier as: <dataset_name>:<experiment_name>.
    """
    ch = "-".join([c.lower() for c in channels])
    notch = "none" if (not filter_notch or filter_notch <= 0) else f"{filter_notch:g}"
    return (
        f"sf{target_sfreq:g}_win{window_sec:g}_st{stride_sec:g}_"
        f"bp{filter_low:g}-{filter_high:g}_notch{notch}_ch{ch}_qc{qc_abs_uv:g}uV"
    )


def _read_pairs_local(raw_data_dir: Path) -> Dict[str, Tuple[Path, Path]]:
    """Build stem -> (psg_edf, hyp_edf) mapping.

    Priority:
      1) If RECORDS exists: read PSG filenames, map hyp via <stem>_hyp.edf
      2) Else: scan *_hyp.edf and pair PSG via removing _hyp suffix.
    """
    stem2pair: Dict[str, Tuple[Path, Path]] = {}

    records_path = raw_data_dir / "RECORDS"
    if records_path.exists():
        lines = [x.strip() for x in records_path.read_text(errors="ignore").splitlines() if x.strip()]
        psg_list = [x for x in lines if x.lower().endswith(".edf") and (not x.lower().endswith("_hyp.edf"))]
        if not psg_list:
            raise FileNotFoundError(f"RECORDS exists but no PSG .edf lines found: {records_path}")

        for rel in psg_list:
            psg_path = raw_data_dir / rel
            stem = psg_path.stem
            hyp_path = raw_data_dir / f"{stem}_hyp.edf"
            if not psg_path.exists():
                raise FileNotFoundError(f"Missing PSG file: {psg_path}")
            if not hyp_path.exists():
                raise FileNotFoundError(f"Missing Hyp file: {hyp_path} (expected from {psg_path.name})")
            stem2pair[stem] = (psg_path, hyp_path)
        return stem2pair

    # fallback: scan for *_hyp.edf
    hyp_files = sorted(raw_data_dir.glob("*_hyp.edf"))
    if not hyp_files:
        raise FileNotFoundError(f"No RECORDS and no *_hyp.edf found in: {raw_data_dir}")

    for hyp_path in hyp_files:
        stem = hyp_path.stem.replace("_hyp", "")
        psg_path = raw_data_dir / f"{stem}.edf"
        if not psg_path.exists():
            raise FileNotFoundError(f"Found hyp but missing PSG: hyp={hyp_path.name}, expected PSG={psg_path.name}")
        stem2pair[stem] = (psg_path, hyp_path)

    return stem2pair


def _pick_and_rename_channels(raw: "mne.io.BaseRaw") -> "mne.io.BaseRaw":
    """Pick EEG channels (prefer Fpz-Cz / Pz-Oz) and rename to FPZ/PZ."""
    # Common variants
    candidates = [
        "EEG Fpz-Cz", "EEG Pz-Oz",
        "Fpz-Cz", "Pz-Oz",
        "EEG FPZ-CZ", "EEG PZ-OZ",
        "FPZ-CZ", "PZ-OZ",
    ]
    picks = [ch for ch in candidates if ch in raw.ch_names]

    if not picks:
        # fallback: first two EEG channels
        eeg_picks = mne.pick_types(raw.info, eeg=True, eog=False, emg=False, misc=False)
        if len(eeg_picks) >= 2:
            picks = [raw.ch_names[eeg_picks[0]], raw.ch_names[eeg_picks[1]]]
        elif len(eeg_picks) == 1:
            picks = [raw.ch_names[eeg_picks[0]]]

    if not picks:
        raise RuntimeError(f"No EEG channels found. Available ch_names (head): {raw.ch_names[:20]}")

    # Use new API instead of legacy pick_channels
    raw = raw.copy().pick(picks)

    rename_map = {}
    for a in ["EEG Fpz-Cz", "Fpz-Cz", "EEG FPZ-CZ", "FPZ-CZ"]:
        if a in raw.ch_names:
            rename_map[a] = "FPZ"
    for a in ["EEG Pz-Oz", "Pz-Oz", "EEG PZ-OZ", "PZ-OZ"]:
        if a in raw.ch_names:
            rename_map[a] = "PZ"
    if rename_map:
        mne.rename_channels(raw.info, rename_map)

    return raw


def _read_hyp_labels(hyp_path: Path) -> List[int]:
    """Read hypnogram EDF and map to 5-class labels; -1 means ignore."""
    hyp = mne.io.read_raw_edf(str(hyp_path), preload=True, verbose=False)
    hyp_data = hyp.get_data()
    if hyp_data.ndim != 2 or hyp_data.shape[0] < 1:
        raise RuntimeError(f"Bad hypnogram shape: {hyp_data.shape} in {hyp_path.name}")

    codes = hyp_data[0].astype(np.int64).tolist()

    labels: List[int] = []
    for c in codes:
        c_int = int(c)
        if c_int in IGNORE_CODES:
            labels.append(-1)
        else:
            labels.append(int(CODE2CLASS.get(c_int, -1)))

    return labels


class SleepData8Builder:
    """Builder implementing the unified interface used by run_dataset.py."""

    def __init__(
        self,
        raw_data_dir: str,
        output_dir: str = "/mnt/dataset2/hdf5_datasets",
        target_sfreq: float = 200.0,
        window_sec: float = 30.0,
        stride_sec: float = 30.0,
        filter_low: float = 0.1,
        filter_high: float = 75.0,
        filter_notch: float = 50.0,
        experiment_name: Optional[str] = None,
        qc_abs_uv: float = 600.0,
    ):
        if not HAS_MNE:
            raise ImportError("MNE is required for SleepData8Builder")

        self.raw_data_dir = Path(raw_data_dir)

        self.target_sfreq = float(target_sfreq)
        self.window_sec = float(window_sec)
        self.stride_sec = float(stride_sec)
        self.filter_low = float(filter_low)
        self.filter_high = float(filter_high)
        self.filter_notch = float(filter_notch)

        self.qc_abs_uv = float(qc_abs_uv)

        if experiment_name and str(experiment_name).strip():
            self.experiment_name = str(experiment_name).strip()
        else:
            self.experiment_name = _compute_experiment_name(
                target_sfreq=self.target_sfreq,
                window_sec=self.window_sec,
                stride_sec=self.stride_sec,
                filter_low=self.filter_low,
                filter_high=self.filter_high,
                filter_notch=self.filter_notch,
                channels=SLEEPDATA8_INFO.channels,
                qc_abs_uv=self.qc_abs_uv,
            )

        # Spec-required dataset identifier used in SubjectAttrs.dataset_name
        self.dataset_identifier = f"{BASE_DATASET_NAME}:{self.experiment_name}"

        # Spec-recommended output folder
        self.output_dir = Path(output_dir) / f"{BASE_DATASET_NAME}__{self.experiment_name}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Pair PSG/Hyp
        self._stem2pair = _read_pairs_local(self.raw_data_dir)

        # Per-subject stats exposed to run_dataset
        self._last_stats: Optional[SubjectBuildStats] = None

    def get_subject_ids(self) -> List[str]:
        return sorted(self._stem2pair.keys())

    def consume_last_stats(self) -> Optional[SubjectBuildStats]:
        """Return stats from the most recent build_subject call (and clear it)."""
        s = self._last_stats
        self._last_stats = None
        return s

    def build_subject(self, subject_id: str) -> str:
        if subject_id not in self._stem2pair:
            raise ValueError(f"Unknown subject_id/stem: {subject_id}")

        psg_path, hyp_path = self._stem2pair[subject_id]
        out_file = self.output_dir / f"sub_{subject_id}.h5"
        if out_file.exists():
            # still record empty stats (so caller can decide)
            self._last_stats = SubjectBuildStats(
                subject_id=subject_id,
                kept_segments=0,
                dropped_segments=0,
                drop_reasons={"skipped_existing": 1},
                label_hist={},
            )
            return str(out_file)

        # ---- Read PSG ----
        raw = mne.io.read_raw_edf(str(psg_path), preload=True, verbose=False)
        raw = _pick_and_rename_channels(raw)

        # ---- Resample & Filter (benchmark convention) ----
        if raw.info["sfreq"] != self.target_sfreq:
            raw.resample(self.target_sfreq, verbose=False)
        raw.filter(l_freq=self.filter_low, h_freq=self.filter_high, verbose=False)
        if self.filter_notch and self.filter_notch > 0:
            raw.notch_filter(freqs=self.filter_notch, verbose=False)

        # ---- Read labels ----
        labels = _read_hyp_labels(hyp_path)

        # ---- Epoch into fixed windows ----
        epochs = mne.make_fixed_length_epochs(
            raw,
            duration=self.window_sec,
            overlap=max(self.window_sec - self.stride_sec, 0.0),
            preload=True,
            verbose=False,
        )
        # V -> uV
        X = epochs.get_data() * 1e6  # (n_epochs, n_ch, n_t)

        n_epochs = min(X.shape[0], len(labels))
        if n_epochs <= 0:
            raise RuntimeError(f"No aligned epochs/labels. epochs={X.shape[0]}, labels={len(labels)}")

        # ---- QC + label filtering ----
        drop_reasons: Dict[str, int] = {"label_ignore": 0, "nan_or_inf": 0, "amp_too_large": 0}
        label_hist: Dict[int, int] = {}

        kept = 0
        dropped = 0

        # Subject attrs (spec: dataset_name must be dataset_identifier)
        subject_attrs = SubjectAttrs(
            subject_id=subject_id,
            dataset_name=self.dataset_identifier,
            task_type=SLEEPDATA8_INFO.task_type.value,
            downstream_task_type=SLEEPDATA8_INFO.downstream_task_type.value,
            rsFreq=self.target_sfreq,
            chn_name=raw.ch_names,
            num_labels=SLEEPDATA8_INFO.num_labels,
            category_list=SLEEPDATA8_INFO.category_list,
            chn_type="EEG",
            montage=SLEEPDATA8_INFO.montage,
        )

        with HDF5Writer(str(out_file), subject_attrs) as writer:
            trial_attrs = TrialAttrs(trial_id=0, session_id=0, task_name="sleep_staging")
            trial_name = writer.add_trial(trial_attrs)

            seg_id = 0
            for i in range(n_epochs):
                y = int(labels[i])
                if y < 0:
                    dropped += 1
                    drop_reasons["label_ignore"] += 1
                    continue

                seg = X[i]

                if not np.isfinite(seg).all():
                    dropped += 1
                    drop_reasons["nan_or_inf"] += 1
                    continue

                if self.qc_abs_uv > 0 and float(np.max(np.abs(seg))) > self.qc_abs_uv:
                    dropped += 1
                    drop_reasons["amp_too_large"] += 1
                    continue

                start_t = float(i * self.stride_sec)
                end_t = float(start_t + self.window_sec)

                seg_attrs = SegmentAttrs(
                    segment_id=seg_id,
                    start_time=start_t,
                    end_time=end_t,
                    time_length=self.window_sec,
                    label=np.array([y], dtype=np.int64),
                )
                writer.add_segment(trial_name, seg_attrs, seg)
                seg_id += 1

                kept += 1
                label_hist[y] = label_hist.get(y, 0) + 1

        self._last_stats = SubjectBuildStats(
            subject_id=subject_id,
            kept_segments=kept,
            dropped_segments=dropped,
            drop_reasons=drop_reasons,
            label_hist=label_hist,
        )

        return str(out_file)

    def finalize(self, aggregated: dict) -> str:
        """Write dataset_info.json (recommended by spec).

        run_dataset.py should call this once at the end if available.
        """
        info_path = self.output_dir / "dataset_info.json"

        payload = {
            "dataset_name": BASE_DATASET_NAME,
            "experiment_name": self.experiment_name,
            "dataset_identifier": self.dataset_identifier,
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "raw_data_dir": str(self.raw_data_dir),
            "output_dir": str(self.output_dir),
            "task_type": "sleep_staging",
            "task_characteristic": "gradual_states",
            "subjects": aggregated.get("subjects", 0),
            "trials_per_subject": 1,
            "total_segments": aggregated.get("total_segments", 0),
            "sampling_rate_hz": self.target_sfreq,
            "epoch_length_sec": self.window_sec,
            "stride_sec": self.stride_sec,
            "channels": list(SLEEPDATA8_INFO.channels),
            "num_channels": len(SLEEPDATA8_INFO.channels),
            "label_definition": {"0": "Wake", "1": "N1", "2": "N2", "3": "N3", "4": "REM"},
            "preprocessing": {
                "bandpass_hz": [self.filter_low, self.filter_high],
                "notch_hz": self.filter_notch,
                "resample_hz": self.target_sfreq,
                "unit": "uV",
                "windowing": "fixed-length",
            },
            "qc": {
                "rules": [
                    "drop if label is ignore/unscored",
                    "drop if NaN/Inf present",
                    f"drop if max(|uV|) > {self.qc_abs_uv:g}",
                ],
                "drop_reason_counts": aggregated.get("drop_reason_counts", {}),
                "dropped_segments": aggregated.get("dropped_segments", 0),
            },
            "label_histogram": aggregated.get("label_histogram", {}),
            "source": {
                "dataset": "Sleep-EDF Database",
                "url": "https://www.physionet.org/content/sleep-edf/1.0.0/",
            },
            "license": "PhysioNet open license",
            "citation": "Kemp et al., 2000, Sleep, 23(Suppl 3):A248",
            "generated_by": "benchmark SleepData8Builder",
        }

        info_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(info_path)

if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Build Sleep-Data8 dataset into HDF5 format"
    )

    # ========== 必要参数 ==========
    parser.add_argument(
        "raw_data_dir",
        type=str,
        help="Path to raw Sleep-Data8 dataset directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for HDF5 files"
    )

    # ========== 可选控制 ==========
    parser.add_argument(
        "--subjects",
        type=str,
        nargs="*",
        default=None,
        help="Specific subject IDs to process (default: all)"
    )

    # ========== 预处理参数（与组内规范一致） ==========
    parser.add_argument("--target_sfreq", type=float, default=200)
    parser.add_argument("--window_sec", type=float, default=30.0)
    parser.add_argument("--stride_sec", type=float, default=30.0)
    parser.add_argument("--filter_low", type=float, default=0.1)
    parser.add_argument("--filter_high", type=float, default=75.0)
    parser.add_argument("--filter_notch", type=float, default=50.0)
    parser.add_argument("--qc_abs_uv", type=float, default=500.0)

    args = parser.parse_args()

    raw_data_dir = Path(args.raw_data_dir)
    output_dir = Path(args.output_dir)

    builder = SleepData8Builder(
        raw_data_dir=raw_data_dir,
        output_dir=output_dir,
        target_sfreq=args.target_sfreq,
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        filter_low=args.filter_low,
        filter_high=args.filter_high,
        filter_notch=args.filter_notch,
        qc_abs_uv=args.qc_abs_uv,
    )

    # -------- subject 列表 --------
    if args.subjects is None:
        subject_ids = builder.get_subject_ids()
    else:
        subject_ids = args.subjects

    print(f"[SleepData8] Processing {len(subject_ids)} subjects")

    # -------- 主处理循环 --------
    all_stats = []
    for sid in subject_ids:
        print(f"[SleepData8] Building subject {sid}")
        builder.build_subject(sid)
        stats = builder.consume_last_stats()
        if stats is not None:
            all_stats.append(stats)

    # -------- finalize --------
    if hasattr(builder, "finalize"):
        builder.finalize(all_stats)

    print("[SleepData8] Done.")

