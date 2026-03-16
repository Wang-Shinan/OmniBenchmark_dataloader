"""
SEED-VII (SEEDVII) Emotion Dataset Builder (benchmark_dataloader format).

This module converts the SEED-VII preprocessed MATLAB data into the unified
EEG_Bench HDF5 format used by `benchmark_dataloader`, with:

- One HDF5 file per subject
- Root-level subject attributes (`SubjectAttrs`)
- Trial groups (`trial*`) with `TrialAttrs`
- Segment groups (`segment*`) with `SegmentAttrs` and EEG data in µV

Labeling strategy (from `emotion_label_and_stimuli_order.xlsx`):
- 7 explicit emotions:
    0: neutral
    1: sad
    2: anger
    3: happy
    4: disgust
    5: fear
    6: surprise
- 1 extra class:
    7: unknown  (trials whose order entry is a number 21..40 or NaN)

By default we keep the 8-class setup (including `unknown`), but this can be
optionally filtered if desired.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.io

try:
    import mne

    HAS_MNE = True
except ImportError:
    HAS_MNE = False

try:
    from ..schema import SubjectAttrs, TrialAttrs, SegmentAttrs
    from ..hdf5_io import HDF5Writer
    from ..config import DatasetInfo, DatasetTaskType, DownstreamTaskType
except ImportError:
    # Fallback for running as a script
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from schema import SubjectAttrs, TrialAttrs, SegmentAttrs  # type: ignore
    from hdf5_io import HDF5Writer  # type: ignore
    from config import DatasetInfo, DatasetTaskType, DownstreamTaskType  # type: ignore


# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------

# Fixed 62-channel order from the original script
SEEDVII_CHANNELS: List[str] = [
    "FP1",
    "FPZ",
    "FP2",
    "AF3",
    "AF4",
    "F7",
    "F5",
    "F3",
    "F1",
    "FZ",
    "F2",
    "F4",
    "F6",
    "F8",
    "FT7",
    "FC5",
    "FC3",
    "FC1",
    "FCZ",
    "FC2",
    "FC4",
    "FC6",
    "FT8",
    "T7",
    "C5",
    "C3",
    "C1",
    "CZ",
    "C2",
    "C4",
    "C6",
    "T8",
    "TP7",
    "CP5",
    "CP3",
    "CP1",
    "CPZ",
    "CP2",
    "CP4",
    "CP6",
    "TP8",
    "P7",
    "P5",
    "P3",
    "P1",
    "PZ",
    "P2",
    "P4",
    "P6",
    "P8",
    "PO7",
    "PO5",
    "PO3",
    "POZ",
    "PO4",
    "PO6",
    "PO8",
    "CB1",
    "O1",
    "OZ",
    "O2",
    "CB2",
]

SEEDVII_INFO = DatasetInfo(
    dataset_name="SEEDVII_emo",
    task_type=DatasetTaskType.COGNITIVE,
    downstream_task_type=DownstreamTaskType.CLASSIFICATION,
    num_labels=8,
    category_list=[
        "neutral",
        "sad",
        "anger",
        "happy",
        "disgust",
        "fear",
        "surprise",
        "unknown",
    ],
    sampling_rate=200.0,  # target sampling rate
    montage="10_20",
    channels=SEEDVII_CHANNELS,
)

# Default amplitude threshold (µV) for validation
DEFAULT_MAX_AMPLITUDE_UV = 600.0


@dataclass
class TrialLabelMapping:
    """Mapping from trial id to emotion label id."""

    trial_to_label: Dict[int, int]
    name_to_id: Dict[str, int]
    id_to_name: Dict[int, str]


def _build_trial_to_emotion_map(order_xlsx: Path) -> TrialLabelMapping:
    """
    Build mapping: trial_id (1..80) -> emotion id (0..7).

    ORDER_XLSX: columns=['Video index', 1..20]
      - 40 entries are text emotions (Neutral/Sad/Anger/Happy/Disgust/Fear/Surprise)
      - 40 entries are numbers (21..40) = clip/video indices (no emotion mapping yet)

    Strategy:
      - Text emotions -> 7 classes (0..6)
      - Numeric / NaN  -> "unknown" = 7 (8th class)
    """
    order_df = pd.read_excel(order_xlsx)
    print("[SEEDVII] ORDER xlsx columns:", list(order_df.columns))

    # Select columns 1..20
    cols = list(order_df.columns)
    trial_cols: List = []
    for c in cols:
        if (isinstance(c, int) and 1 <= c <= 20) or (
            isinstance(c, str) and c.isdigit() and 1 <= int(c) <= 20
        ):
            trial_cols.append(c)
    trial_cols = sorted(trial_cols, key=lambda x: int(x) if isinstance(x, str) else x)
    if len(trial_cols) != 20:
        raise ValueError(
            f"Expected 20 columns(1..20) in ORDER_XLSX, got {len(trial_cols)}: {trial_cols}"
        )

    mat = order_df[trial_cols].to_numpy()
    flat = mat.reshape(-1)
    if flat.shape[0] < 80:
        raise ValueError(f"Flattened order length <80: got {flat.shape[0]}")
    flat = flat[:80]

    # Recognize numeric values
    def is_number_like(x) -> bool:
        if pd.isna(x):
            return False
        if isinstance(x, (int, np.integer, float, np.floating)):
            return True
        s = str(x).strip()
        return s.replace(".", "", 1).isdigit()

    name2id: Dict[str, int] = {
        "neutral": 0,
        "sad": 1,
        "anger": 2,
        "happy": 3,
        "disgust": 4,
        "fear": 5,
        "surprise": 6,
        "unknown": 7,
    }
    id2name: Dict[int, str] = {v: k for k, v in name2id.items()}

    t2y: Dict[int, int] = {}
    unknown_count = 0
    text_count = 0

    for tid in range(1, 81):
        x = flat[tid - 1]

        if pd.isna(x):
            # NaN -> unknown
            t2y[tid] = name2id["unknown"]
            unknown_count += 1
            continue

        if is_number_like(x):
            # Numbers (21..40) -> unknown
            t2y[tid] = name2id["unknown"]
            unknown_count += 1
            continue

        # Text emotion
        s = str(x).strip().lower()
        if s not in name2id:
            raise KeyError(f"Unknown emotion name in ORDER_XLSX: '{x}' (trial {tid})")
        t2y[tid] = name2id[s]
        text_count += 1

    print(
        f"[SEEDVII] mapped text emotion trials: {text_count}, "
        f"unknown trials: {unknown_count}"
    )
    print(f"[SEEDVII] classes: {len(name2id)} -> {name2id}")

    return TrialLabelMapping(trial_to_label=t2y, name_to_id=name2id, id_to_name=id2name)


class SEEDVIIBuilder:
    """Builder for SEED-VII emotion dataset in benchmark_dataloader format."""

    def __init__(
        self,
        eeg_dir: str,
        order_xlsx: str,
        output_dir: str = "./hdf5",
        target_sfreq: float = 200.0,
        window_sec: float = 2.0,
        stride_sec: float = 2.0,
        filter_low: float = 0.1,
        filter_high: float = 75.0,
        filter_notch: float = 50.0,
        sfreq_in_default: float = 200.0,
        max_amplitude_uv: float = DEFAULT_MAX_AMPLITUDE_UV,
        include_unknown: bool = True,
    ):
        """
        Initialize SEEDVII builder.

        Args:
            eeg_dir: Directory containing subject .mat files (preprocessed EEG).
                     Each file is named like "1.mat", "2.mat", ...
            order_xlsx: Path to `emotion_label_and_stimuli_order.xlsx`.
            output_dir: Output root directory (HDF5 files will be in
                        `<output_dir>/SEEDVII_emo`).
            target_sfreq: Target sampling rate in Hz (default: 200.0).
            window_sec: Window length in seconds (default: 2.0).
            stride_sec: Stride length in seconds (default: 2.0).
            filter_low: Low cutoff frequency for bandpass filter (Hz).
            filter_high: High cutoff frequency for bandpass filter (Hz).
            filter_notch: Notch filter frequency (Hz), 0 to disable.
            sfreq_in_default: Default input sampling rate of .mat data (Hz).
            max_amplitude_uv: Amplitude threshold (µV) for segment validation.
            include_unknown: Whether to keep "unknown" label (class 7). If False,
                             trials mapped to "unknown" are skipped entirely.
        """
        self.eeg_dir = Path(eeg_dir)
        self.order_xlsx = Path(order_xlsx)

        output_path = Path(output_dir)
        if output_path.name == SEEDVII_INFO.dataset_name:
            self.output_dir = output_path
        else:
            self.output_dir = output_path / SEEDVII_INFO.dataset_name

        self.target_sfreq = float(target_sfreq)
        self.window_sec = float(window_sec)
        self.stride_sec = float(stride_sec)
        self.filter_low = float(filter_low)
        self.filter_high = float(filter_high)
        self.filter_notch = float(filter_notch)
        self.sfreq_in_default = float(sfreq_in_default)
        self.max_amplitude_uv = float(max_amplitude_uv)
        self.include_unknown = bool(include_unknown)

        self.window_samples = int(self.window_sec * self.target_sfreq)
        self.stride_samples = int(self.stride_sec * self.target_sfreq)

        if not HAS_MNE:
            raise ImportError("mne is required for SEEDVII preprocessing")

        # Build label mapping from Excel
        mapping = _build_trial_to_emotion_map(self.order_xlsx)
        self.trial_to_label = mapping.trial_to_label
        self.name_to_id = mapping.name_to_id
        self.id_to_name = mapping.id_to_name

        # Validation statistics
        self.total_segments = 0
        self.valid_segments = 0
        self.rejected_segments_amp = 0

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _preprocess_CxT(self, eeg_CxT: np.ndarray, sfreq_in: float) -> np.ndarray:
        """
        Bandpass (0.1–75 Hz) + optional resampling to target_sfreq.

        Input and output are in µV.
        """
        eeg_f = mne.filter.filter_data(
            eeg_CxT.astype(np.float64),
            sfreq=sfreq_in,
            l_freq=self.filter_low,
            h_freq=self.filter_high,
            method="iir",
            iir_params=dict(order=4, ftype="butter"),
            verbose=False,
        )

        if abs(sfreq_in - self.target_sfreq) < 1e-6:
            return eeg_f.astype(np.float32)

        eeg_r = mne.filter.resample(
            eeg_f,
            up=self.target_sfreq,
            down=sfreq_in,
            axis=-1,
            npad="auto",
            verbose=False,
        )
        return eeg_r.astype(np.float32)

    def _load_subject_mat(self, mat_path: Path) -> List[Tuple[int, np.ndarray]]:
        """
        Load subject .mat file.

        Each key '1'..'80' corresponds to a trial array (C x T or T x C).
        Returns list of (trial_id, eeg_CxT) with shape (n_channels, n_samples).
        """
        m = scipy.io.loadmat(str(mat_path))
        keys = [k for k in m.keys() if not k.startswith("__")]

        num_keys: List[int] = []
        for k in keys:
            try:
                if str(k).isdigit():
                    num_keys.append(int(k))
            except Exception:
                continue
        num_keys = sorted(num_keys)

        if len(num_keys) == 0:
            raise ValueError(f"No numeric trial keys found in {mat_path}")

        trials: List[Tuple[int, np.ndarray]] = []
        n_ch_expected = len(SEEDVII_CHANNELS)

        for tid in num_keys:
            arr = np.array(m[str(tid)])
            if arr.ndim != 2:
                raise ValueError(
                    f"Trial {tid} in {mat_path.name} not 2D: shape={arr.shape}"
                )
            # unify to (C, T)
            if arr.shape[0] != n_ch_expected and arr.shape[1] == n_ch_expected:
                arr = arr.T
            if arr.shape[0] != n_ch_expected:
                raise ValueError(
                    f"Trial {tid} channel mismatch: got {arr.shape[0]}, "
                    f"expected {n_ch_expected}"
                )
            trials.append((tid, arr))

        return trials

    def _validate_segment(self, seg_uv: np.ndarray) -> bool:
        """Check if segment amplitude is within acceptable range (µV)."""
        return float(np.abs(seg_uv).max()) <= self.max_amplitude_uv

    # ------------------------------------------------------------------
    # Build functions
    # ------------------------------------------------------------------

    def _get_subject_ids(self) -> List[int]:
        """Infer subject IDs from .mat filenames in eeg_dir."""
        ids: List[int] = []
        for p in self.eeg_dir.glob("*.mat"):
            stem = p.stem
            if stem.isdigit():
                ids.append(int(stem))
        return sorted(ids)

    def _save_dataset_info(self, stats: Dict) -> None:
        """Save dataset info and processing parameters to JSON."""
        info = {
            "dataset": {
                "name": SEEDVII_INFO.dataset_name,
                "description": "SEED-VII emotion EEG dataset (8-class: 7 emotions + unknown)",
                "task_type": str(SEEDVII_INFO.task_type.value),
                "downstream_task": str(SEEDVII_INFO.downstream_task_type.value),
                "num_labels": SEEDVII_INFO.num_labels,
                "category_list": SEEDVII_INFO.category_list,
                "original_sampling_rate": self.sfreq_in_default,
                "channels": SEEDVII_INFO.channels,
                "channel_count": len(SEEDVII_INFO.channels),
                "montage": SEEDVII_INFO.montage,
                "source_url": "https://bcmi.sjtu.edu.cn/~seed/seed-vii.html",
            },
            "processing": {
                "target_sampling_rate": self.target_sfreq,
                "window_sec": self.window_sec,
                "stride_sec": self.stride_sec,
                "filter_low": self.filter_low,
                "filter_high": self.filter_high,
                "filter_notch": self.filter_notch,
                "sfreq_in_default": self.sfreq_in_default,
                "max_amplitude_uv": self.max_amplitude_uv,
                "include_unknown": self.include_unknown,
            },
            "statistics": stats,
            "generated_at": datetime.now().isoformat(),
        }

        self.output_dir.mkdir(parents=True, exist_ok=True)
        json_path = self.output_dir / "dataset_info.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)
        print(f"[SEEDVII] Saved dataset info to {json_path}")

    def build_subject(self, subject_id: int) -> Optional[str]:
        """
        Build HDF5 file for a single subject.

        Returns output path or None if subject file is missing or no valid segments.
        """
        mat_path = self.eeg_dir / f"{subject_id}.mat"
        if not mat_path.exists():
            print(f"[SEEDVII] Subject {subject_id}: {mat_path} not found, skipping.")
            return None

        # Reset validation counters for this subject
        self.total_segments = 0
        self.valid_segments = 0
        self.rejected_segments_amp = 0

        trials = self._load_subject_mat(mat_path)
        print(f"[SEEDVII] Subject {subject_id}: trials={len(trials)} (expect up to 80)")

        all_segments: List[Dict] = []

        for trial_id, eeg_CxT in trials:
            if trial_id not in self.trial_to_label:
                raise KeyError(
                    f"trial_id {trial_id} not in trial_to_label (expect 1..80)."
                )
            label = int(self.trial_to_label[trial_id])
            label_name = self.id_to_name.get(label, "unknown")

            if (not self.include_unknown) and label_name == "unknown":
                # Skip unknown trials entirely
                continue

            # Preprocess (bandpass + resample)
            eeg_proc = self._preprocess_CxT(eeg_CxT, sfreq_in=self.sfreq_in_default)
            n_samples = eeg_proc.shape[-1]

            # Sliding windows
            for i_seg, start in enumerate(
                range(0, n_samples - self.window_samples + 1, self.stride_samples)
            ):
                end = start + self.window_samples
                seg_uv = eeg_proc[:, start:end]  # already in µV

                self.total_segments += 1
                if not self._validate_segment(seg_uv):
                    self.rejected_segments_amp += 1
                    continue

                self.valid_segments += 1

                all_segments.append(
                    {
                        "trial_id": trial_id,
                        "segment_id": i_seg,
                        "start_sample": start,
                        "data_uv": seg_uv,
                        "label": label,
                    }
                )

        if not all_segments:
            print(f"[SEEDVII] Subject {subject_id}: no valid segments, skipping.")
            return None

        # Create subject attributes
        subject_attrs = SubjectAttrs(
            subject_id=subject_id,
            dataset_name=SEEDVII_INFO.dataset_name,
            task_type=SEEDVII_INFO.task_type.value,
            downstream_task_type=SEEDVII_INFO.downstream_task_type.value,
            rsFreq=self.target_sfreq,
            chn_name=SEEDVII_INFO.channels,
            num_labels=SEEDVII_INFO.num_labels,
            category_list=SEEDVII_INFO.category_list,
            chn_pos=None,
            chn_ori=None,
            chn_type="EEG",
            montage=SEEDVII_INFO.montage,
        )

        # Prepare output
        if self.output_dir.exists() and not self.output_dir.is_dir():
            raise FileExistsError(
                f"Output directory path exists as a file: {self.output_dir}"
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        output_path = self.output_dir / f"sub_{subject_id}.h5"

        # Write using HDF5Writer
        with HDF5Writer(str(output_path), subject_attrs) as writer:
            # Group segments by trial_id so that each original trial is a Trial
            from collections import defaultdict

            trial_to_segments: Dict[int, List[Dict]] = defaultdict(list)
            for seg in all_segments:
                trial_to_segments[seg["trial_id"]].append(seg)

            for trial_idx, (trial_id, seg_list) in enumerate(
                sorted(trial_to_segments.items())
            ):
                trial_attrs = TrialAttrs(
                    trial_id=int(trial_id),
                    session_id=0,
                    task_name="emotion",
                )
                trial_name = writer.add_trial(trial_attrs)

                for seg in seg_list:
                    start_sample = seg["start_sample"]
                    seg_start_time = start_sample / self.target_sfreq
                    seg_end_time = seg_start_time + self.window_sec

                    segment_attrs = SegmentAttrs(
                        segment_id=int(seg["segment_id"]),
                        start_time=float(seg_start_time),
                        end_time=float(seg_end_time),
                        time_length=self.window_sec,
                        label=np.array([seg["label"]], dtype=int),
                    )
                    writer.add_segment(trial_name, segment_attrs, seg["data_uv"])

        print(
            f"[SEEDVII] Subject {subject_id}: saved {output_path} "
            f"({self.valid_segments} valid segments, "
            f"{self.rejected_segments_amp} rejected by amplitude)"
        )

        return str(output_path)

    def build_all(self, subject_ids: Optional[List[int]] = None) -> List[str]:
        """
        Build HDF5 files for all subjects.

        Args:
            subject_ids: List of subject IDs to process (None = infer from .mat files)

        Returns:
            List of output file paths
        """
        if subject_ids is None:
            subject_ids = self._get_subject_ids()

        output_paths: List[str] = []
        failed_subjects: List[int] = []

        total_segments_all = 0
        valid_segments_all = 0
        rejected_amp_all = 0

        for sid in subject_ids:
            try:
                path = self.build_subject(sid)
                if path:
                    output_paths.append(path)
                    total_segments_all += self.total_segments
                    valid_segments_all += self.valid_segments
                    rejected_amp_all += self.rejected_segments_amp
            except Exception as e:
                print(f"[SEEDVII] Error processing subject {sid}: {e}")
                failed_subjects.append(sid)
                import traceback

                traceback.print_exc()

        # Summary
        print("\n" + "=" * 60)
        print("SEEDVII PROCESSING SUMMARY")
        print("=" * 60)
        print(f"Total subjects requested: {len(subject_ids)}")
        print(f"Successful subjects: {len(output_paths)}")
        print(f"Failed subjects: {len(failed_subjects)}")
        if failed_subjects:
            print(f"Failed subject IDs: {failed_subjects}")
        print(f"Total segments (before rejection): {total_segments_all}")
        print(f"Valid segments: {valid_segments_all}")
        print(f"Rejected (amplitude): {rejected_amp_all}")
        if total_segments_all > 0:
            print(
                f"Valid rate: {valid_segments_all / total_segments_all * 100:.1f}%"
            )
        print("=" * 60)

        # Save dataset_info.json
        stats = {
            "total_subjects_requested": len(subject_ids),
            "successful_subjects": len(output_paths),
            "failed_subjects": failed_subjects,
            "total_segments": total_segments_all,
            "valid_segments": valid_segments_all,
            "rejected_segments_amp": rejected_amp_all,
        }
        self._save_dataset_info(stats)

        return output_paths


def build_seedvii(
    eeg_dir: str,
    order_xlsx: str,
    output_dir: str = "./hdf5",
    subject_ids: Optional[List[int]] = None,
    **kwargs,
) -> List[str]:
    """
    Convenience function to build SEED-VII emotion dataset.

    Example:
        build_seedvii(
            eeg_dir=\"/mnt/dataset0/qingzhu/EEG_raw/SEED-VII/EEG_preprocessed\",
            order_xlsx=\"/mnt/dataset0/qingzhu/EEG_raw/SEED-VII/emotion_label_and_stimuli_order.xlsx\",
            output_dir=\"/mnt/dataset2/Processed_datasets/EEG_Bench\",
        )
    """
    builder = SEEDVIIBuilder(
        eeg_dir=eeg_dir,
        order_xlsx=order_xlsx,
        output_dir=output_dir,
        **kwargs,
    )
    return builder.build_all(subject_ids)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build SEED-VII (SEEDVII) emotion HDF5 dataset in benchmark_dataloader format"
    )
    parser.add_argument(
        "eeg_dir",
        help="Directory containing SEED-VII preprocessed EEG .mat files (e.g., EEG_preprocessed)",
    )
    parser.add_argument(
        "order_xlsx",
        help="Path to emotion_label_and_stimuli_order.xlsx",
    )
    parser.add_argument(
        "--output_dir",
        default="/mnt/dataset2/Processed_datasets/EEG_Bench/SEEDVII_emo",
        help="Output directory root (HDF5 will be stored under SEEDVII_emo)",
    )
    parser.add_argument(
        "--target_sfreq",
        type=float,
        default=200.0,
        help="Target sampling frequency (Hz)",
    )
    parser.add_argument(
        "--window_sec",
        type=float,
        default=2.0,
        help="Window length in seconds",
    )
    parser.add_argument(
        "--stride_sec",
        type=float,
        default=2.0,
        help="Stride length in seconds",
    )
    parser.add_argument(
        "--filter_low",
        type=float,
        default=0.1,
        help="Low cutoff frequency for bandpass filter (Hz)",
    )
    parser.add_argument(
        "--filter_high",
        type=float,
        default=75.0,
        help="High cutoff frequency for bandpass filter (Hz)",
    )
    parser.add_argument(
        "--filter_notch",
        type=float,
        default=50.0,
        help="Notch filter frequency (Hz, 0 to disable)",
    )
    parser.add_argument(
        "--sfreq_in_default",
        type=float,
        default=200.0,
        help="Default input sampling rate of .mat data (Hz)",
    )
    parser.add_argument(
        "--max_amplitude_uv",
        type=float,
        default=DEFAULT_MAX_AMPLITUDE_UV,
        help="Maximum amplitude threshold in µV",
    )
    parser.add_argument(
        "--include_unknown",
        action="store_true",
        help="Include 'unknown' label (class 7). Default: False (skip unknown trials).",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        type=int,
        help="Subject IDs to process (e.g., 1 2 3). Default: infer all from eeg_dir.",
    )

    args = parser.parse_args()

    build_seedvii(
        eeg_dir=args.eeg_dir,
        order_xlsx=args.order_xlsx,
        output_dir=args.output_dir,
        subject_ids=args.subjects,
        target_sfreq=args.target_sfreq,
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        filter_low=args.filter_low,
        filter_high=args.filter_high,
        filter_notch=args.filter_notch,
        sfreq_in_default=args.sfreq_in_default,
        max_amplitude_uv=args.max_amplitude_uv,
        include_unknown=args.include_unknown,
    )

import re
import numpy as np
import h5py
from pathlib import Path
import pandas as pd
import mne
import scipy.io

# ======================= 路径配置 =======================
ROOT = Path("/mnt/dataset0/qingzhu/EEG_raw/SEED-VII")
EEG_DIR = ROOT / "EEG_preprocessed"
ORDER_XLSX = ROOT / "emotion_label_and_stimuli_order.xlsx"   # 你的表：Video index + 1..20
STIM_XLSX  = ROOT / "SEED-VII_stimulation.xlsx"              # video id -> emotion

OUT_DIR = Path("/mnt/dataset2/benchmark_hdf5/SEEDVII_unified_T=2s_stride=2s")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ======================= 处理参数（和学姐要求一致） =======================
DATASET_NAME = "SEED-VII"
L_FREQ, H_FREQ = 0.1, 75.0

TARGET_SFREQ = 200.0
SFREQ_IN_DEFAULT = 200.0  # SEED 预处理版通常已是 200Hz；若不是请改

SAMPLE_T = 2.0
STRIDE_T = 2.0
SAMPLE_SAMPLES = int(SAMPLE_T * TARGET_SFREQ)   # 400
STRIDE_SAMPLES = int(STRIDE_T * TARGET_SFREQ)   # 400

LABEL_AS_ONEHOT = True  # True: one-hot(7,)  False: int(0..K-1)
MONTAGE = "10_20"       # 不确定也可改 "None"

# ======================= 固定 62 通道顺序（你确认的） =======================
CH_NAMES = [
    "FP1","FPZ","FP2","AF3","AF4","F7","F5","F3","F1","FZ","F2","F4","F6","F8",
    "FT7","FC5","FC3","FC1","FCZ","FC2","FC4","FC6","FT8",
    "T7","C5","C3","C1","CZ","C2","C4","C6","T8",
    "TP7","CP5","CP3","CP1","CPZ","CP2","CP4","CP6","TP8",
    "P7","P5","P3","P1","PZ","P2","P4","P6","P8",
    "PO7","PO5","PO3","POZ","PO4","PO6","PO8",
    "CB1","O1","OZ","O2","CB2"
]
CH_NAMES_H5 = np.array(CH_NAMES, dtype="S")
N_CH = len(CH_NAMES)
assert N_CH == 62, f"Expected 62 channels, got {N_CH}"
print(f"[INFO] Channel names loaded (fixed): {N_CH}")

# ======================= 小工具 =======================
def onehot(y: int, k: int):
    v = np.zeros((k,), dtype=np.uint8)
    v[int(y)] = 1
    return v

# ======================= 预处理：0.1-75Hz + resample到200Hz =======================
def preprocess_CxT(eeg_CxT: np.ndarray, sfreq_in: float):
    eeg_f = mne.filter.filter_data(
        eeg_CxT.astype(np.float64),
        sfreq=sfreq_in,
        l_freq=0.1,
        h_freq=75.0,
        method="iir",
        iir_params=dict(order=4, ftype="butter"),
        verbose=False,
    )
    if abs(sfreq_in - 200.0) < 1e-6:
        return eeg_f.astype(np.float32)

    eeg_r = mne.filter.resample(
        eeg_f, up=200.0, down=sfreq_in,
        axis=-1, npad="auto", verbose=False
    )
    return eeg_r.astype(np.float32)
        

# ======================= 读取 subject mat：keys 是 '1'..'80' =======================
def load_subject_mat(mat_path: Path):
    m = scipy.io.loadmat(str(mat_path))
    keys = [k for k in m.keys() if not k.startswith("__")]

    num_keys = []
    for k in keys:
        if re.fullmatch(r"\d+", str(k)):
            num_keys.append(int(k))
    num_keys = sorted(num_keys)

    if len(num_keys) == 0:
        raise ValueError(f"No numeric trial keys found in {mat_path}")

    trials = []
    for tid in num_keys:
        arr = np.array(m[str(tid)])
        if arr.ndim != 2:
            raise ValueError(f"Trial {tid} in {mat_path.name} not 2D: shape={arr.shape}")
        # unify to (C,T)
        if arr.shape[0] != N_CH and arr.shape[1] == N_CH:
            arr = arr.T
        if arr.shape[0] != N_CH:
            raise ValueError(f"Trial {tid} channel mismatch: got {arr.shape[0]}, expected {N_CH}")
        trials.append((tid, arr))
    return trials

# ======================= label 映射（order xlsx + stimulation xlsx） =======================


def build_trial_to_emotion_map():
    """
    ORDER_XLSX: columns=['Video index', 1..20]
      - 40 个是文字情绪（Neutral/Sad/Anger/Happy/Disgust/Fear/Surprise）
      - 40 个是数字（21..40）= clip/video 编号（目前无法映射到情绪）
    策略：
      - 文字情绪 -> 7类 (0..6)
      - 数字编号 -> unknown=7 (第8类)
    输出：
      trial_id(1..80) -> label_id(0..7)
      id2name/name2id
    """
    order_df = pd.read_excel(ORDER_XLSX)
    print("[INFO] order xlsx columns:", list(order_df.columns))

    # 取 1..20 列
    cols = list(order_df.columns)
    trial_cols = []
    for c in cols:
        if (isinstance(c, int) and 1 <= c <= 20) or (isinstance(c, str) and c.isdigit() and 1 <= int(c) <= 20):
            trial_cols.append(c)
    trial_cols = sorted(trial_cols, key=lambda x: int(x) if isinstance(x, str) else x)
    if len(trial_cols) != 20:
        raise ValueError(f"Expected 20 columns(1..20) in ORDER_XLSX, got {len(trial_cols)}: {trial_cols}")

    mat = order_df[trial_cols].to_numpy()
    flat = mat.reshape(-1)
    if flat.shape[0] < 80:
        raise ValueError(f"Flattened order length <80: got {flat.shape[0]}")
    flat = flat[:80]

    # 识别数字
    def is_number_like(x):
        if pd.isna(x): return False
        if isinstance(x, (int, np.integer, float, np.floating)): return True
        s = str(x).strip()
        return s.replace(".", "", 1).isdigit()

    # 你统计出来的 7 类情绪（固定顺序）
    name2id = {
        "neutral": 0,
        "sad": 1,
        "anger": 2,
        "happy": 3,
        "disgust": 4,
        "fear": 5,
        "surprise": 6,
        "unknown": 7,
    }
    id2name = {v: k for k, v in name2id.items()}

    t2y = {}
    unknown_count = 0
    text_count = 0

    for tid in range(1, 81):
        x = flat[tid - 1]

        if pd.isna(x):
            # 空值也当 unknown
            t2y[tid] = name2id["unknown"]
            unknown_count += 1
            continue

        if is_number_like(x):
            # 数字（21..40）没有映射表 -> unknown
            t2y[tid] = name2id["unknown"]
            unknown_count += 1
            continue

        # 文字情绪
        s = str(x).strip().lower()
        if s not in name2id:
            raise KeyError(f"Unknown emotion name in ORDER_XLSX: '{x}' (trial {tid})")
        t2y[tid] = name2id[s]
        text_count += 1

    print(f"[INFO] mapped text emotion trials: {text_count}, unknown trials: {unknown_count}")
    print(f"[INFO] classes: {len(name2id)} -> {name2id}")

    return t2y, name2id







TRIAL2EMO, EMO2ID = build_trial_to_emotion_map()
N_CLASS = len(EMO2ID)

# ======================= 写 H5（完全对齐学姐 emoeeg 结构/attrs） =======================
def build_one_subject(mat_path: Path):
    sid = int(mat_path.stem)  # 9.mat -> subject_id=9
    out_path = OUT_DIR / f"sub_{sid}.h5"
    if out_path.exists():
        print(f"[SKIP] exists: {out_path}")
        return

    trials = load_subject_mat(mat_path)
    print(f"[INFO] subject {sid}: trials={len(trials)} (expect 80)")

    with h5py.File(out_path, "w") as f:
        for trial_id, eeg_CxT in trials:
            # label（trial-level）
            if trial_id not in TRIAL2EMO:
                raise KeyError(f"trial_id {trial_id} not in TRIAL2EMO (expect 1..80).")
            y = TRIAL2EMO[trial_id]
            y_store = onehot(y, N_CLASS) if LABEL_AS_ONEHOT else int(y)

            # 预处理
            eeg_CxT = preprocess_CxT(eeg_CxT, sfreq_in=SFREQ_IN_DEFAULT)
            n_samples = eeg_CxT.shape[-1]

            trial_grp = f.create_group(f"trial{trial_id}")

            # 2s window, stride 2s
            for i_slice, start in enumerate(range(0, n_samples - SAMPLE_SAMPLES + 1, STRIDE_SAMPLES)):
                end = start + SAMPLE_SAMPLES
                slice_data = eeg_CxT[:, start:end]  # (62, 400)

                slice_grp = trial_grp.create_group(f"sample{i_slice}")
                dset = slice_grp.create_dataset("eeg", data=slice_data, compression="gzip")

                # ===== attrs：键名必须和学姐一致 =====
                dset.attrs["rsFreq"] = int(TARGET_SFREQ)
                dset.attrs["label"] = y_store

                dset.attrs["subject_id"] = int(sid)
                dset.attrs["trial_id"] = int(trial_id)
                dset.attrs["session_id"] = 0
                dset.attrs["segment_id"] = int(i_slice)
                dset.attrs["time_length"] = float(SAMPLE_T)
                dset.attrs["dataset_name"] = DATASET_NAME

                dset.attrs["chn_name"] = CH_NAMES_H5
                dset.attrs["chn_pos"] = "None"
                dset.attrs["chn_ori"] = "None"
                dset.attrs["chn_type"] = "EEG"
                dset.attrs["montage"] = MONTAGE
                dset.attrs["start_sample"] = int(start)

    print(f"[OK] saved: {out_path}")

def main():
    mats = sorted(
        [p for p in EEG_DIR.glob("*.mat") if p.stem.isdigit()],
        key=lambda p: int(p.stem)
    )
    print("[INFO] mat files:", len(mats), "example:", [p.name for p in mats[:10]])

    for mp in mats:
        build_one_subject(mp)

if __name__ == "__main__":
    main()