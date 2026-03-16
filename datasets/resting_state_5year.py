"""
Resting_State_5year Dataset Builder.

Dataset: Longitudinal EEG Test-Retest Reliability in Healthy Individuals
- OpenNeuro ID: ds005385 (5-year longitudinal resting-state EEG)
- BrainVision format (.vhdr/.eeg/.vmrk), BIDS layout
- Multiple sessions per subject (e.g., V0, V1, V2, V3, V4)
- Two resting conditions per session:
  - CE: Eyes closed
  - OE: Eyes open

This builder:
- Reads raw resting EEG from `sub-*/ses-*/eeg/sub-*_ses-*_task-*_eeg.vhdr`
- Bandpass: 0.5–70 Hz, 60 Hz notch (PowerLineFrequency=60 in dataset)
- Resamples to 200 Hz
- Segments into fixed-length windows
- Writes HDF5 with per-segment labels: CE vs OE
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import mne
    HAS_MNE = True
except ImportError:
    HAS_MNE = False

try:
    from ..schema import SubjectAttrs, TrialAttrs, SegmentAttrs
    from ..hdf5_io import HDF5Writer
    from ..config import DatasetInfo, DatasetTaskType, DownstreamTaskType
except ImportError:  # pragma: no cover - fallback for direct execution
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
    from schema import SubjectAttrs, TrialAttrs, SegmentAttrs  # type: ignore
    from hdf5_io import HDF5Writer  # type: ignore
    from config import DatasetInfo, DatasetTaskType, DownstreamTaskType  # type: ignore


DEFAULT_MAX_AMPLITUDE_UV = 800.0


# Base dataset info (resting-state, subject-as-label; details filled at runtime)
RESTING5Y_INFO = DatasetInfo(
    dataset_name="Resting_State_5year",
    task_type=DatasetTaskType.RESTING_STATE,
    downstream_task_type=DownstreamTaskType.CLASSIFICATION,
    num_labels=0,           # will be overridden with number of subjects
    category_list=[],       # will be overridden with list of subject IDs
    sampling_rate=200.0,
    montage="10_20",
    channels=[],  # Will be taken from Raw.ch_names
)


def detect_unit_and_convert_to_volts(data: np.ndarray) -> Tuple[np.ndarray, str]:
    """
    Robustly detect unit (V/mV/µV) and convert to Volts.
    """
    abs_data = np.abs(data)
    # Use robust percentiles to avoid artifacts
    robust_max = np.percentile(abs_data, 99.0)
    max_amp = max(robust_max, np.percentile(abs_data, 95.0))

    mad = np.median(abs_data)
    if mad > 0:
        mad_based = 3 * mad
        max_amp = max(max_amp, mad_based)

    if max_amp > 1e-2:  # > 0.01, likely µV
        return data / 1e6, "µV"
    if max_amp > 1e-5:  # > 0.00001, likely mV
        return data / 1e3, "mV"
    return data, "V"


@dataclass
class SessionFile:
    """Container for one session-condition file."""

    session: str  # e.g., "ses-V0"
    condition: str  # "CE" or "OE"
    vhdr_path: Path


class Resting5YearBuilder:
    """
    Builder for Resting_State_5year dataset (ds005385-like longitudinal resting EEG).
    """

    def __init__(
        self,
        raw_root: str,
        output_root: str = "./hdf5",
        target_sfreq: float = 200.0,
        window_sec: float = 2.0,
        stride_sec: float = 2.0,
        filter_low: float = 0.5,
        filter_high: float = 70.0,
        filter_notch: float = 60.0,
        max_amplitude_uv: float = DEFAULT_MAX_AMPLITUDE_UV,
    ):
        """
        Args:
            raw_root: Path to BIDS root, e.g.
                `/mnt/dataset2/novelDS/Longitudinal EEG Reliability`
            output_root: Base output directory for HDF5 files
        """
        self.raw_root = Path(raw_root)
        self.output_root = Path(output_root) / RESTING5Y_INFO.dataset_name

        self.target_sfreq = target_sfreq
        self.window_sec = window_sec
        self.stride_sec = stride_sec
        self.filter_low = filter_low
        self.filter_high = filter_high
        self.filter_notch = filter_notch
        self.max_amplitude_uv = max_amplitude_uv

        self.window_samples = int(window_sec * target_sfreq)
        self.stride_samples = int(stride_sec * target_sfreq)

        # Track validation statistics
        self.total_segments: int = 0
        self.valid_segments: int = 0
        self.rejected_segments: int = 0

        # Subject-as-label configuration
        # subject_label_map: {"sub-XXX": label_index}
        self.subject_label_map: Dict[str, int] = {}
        # label_categories: ordered list of subject IDs, index = label id
        self.label_categories: List[str] = []

        # Global channel list (set from first successfully read subject)
        self.channels: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # Subject & file discovery
    # ------------------------------------------------------------------
    def get_subject_ids(self) -> List[str]:
        """Return list of BIDS subject IDs with EEG data (e.g., 'sub-001').

        For the Resting_State_5year dataset, raw EEG is stored as EDF files:
        sub-XXX/ses-Y/eeg/sub-XXX_ses-Y_task-*_eeg.edf
        """
        subjects: List[str] = []
        if not self.raw_root.exists():
            raise FileNotFoundError(f"Raw root not found: {self.raw_root}")

        for sub_dir in sorted(self.raw_root.glob("sub-*")):
            # Check if there is at least one EDF file in any session
            has_eeg = False
            for ses_dir in sub_dir.glob("ses-*"):
                eeg_dir = ses_dir / "eeg"
                if eeg_dir.exists() and list(eeg_dir.glob("*_eeg.edf")):
                    has_eeg = True
                    break
            if has_eeg:
                subjects.append(sub_dir.name)
        return subjects

    def _find_session_files(self, bids_id: str) -> List[SessionFile]:
        """
        Find all EyesClosed/EyesOpen EDF files for a subject.

        File pattern (BIDS-style):
        sub-XXX/ses-Y/eeg/sub-XXX_ses-Y_task-EyesClosed_acq-*_eeg.edf
        sub-XXX_ses-Y_task-EyesOpen_acq-*_eeg.edf

        Returns:
            List of SessionFile entries.
        """
        files: List[SessionFile] = []
        sub_dir = self.raw_root / bids_id
        if not sub_dir.exists():
            print(f"⚠️  Warning: subject directory not found: {sub_dir}")
            return files

        for ses_dir in sorted(sub_dir.glob("ses-*")):
            session = ses_dir.name  # e.g., "ses-1"
            eeg_dir = ses_dir / "eeg"
            if not eeg_dir.exists():
                continue

            # Eyes closed
            for edf in sorted(
                eeg_dir.glob(f"{bids_id}_{session}_task-EyesClosed*_eeg.edf")
            ):
                files.append(
                    SessionFile(session=session, condition="EyesClosed", vhdr_path=edf)
                )
                break

            # Eyes open
            for edf in sorted(
                eeg_dir.glob(f"{bids_id}_{session}_task-EyesOpen*_eeg.edf")
            ):
                files.append(
                    SessionFile(session=session, condition="EyesOpen", vhdr_path=edf)
                )
                break

        return files

    # ------------------------------------------------------------------
    # Reading & preprocessing
    # ------------------------------------------------------------------
    def _read_raw_brainvision(self, vhdr_path: Path) -> mne.io.BaseRaw:
        """Read an EDF file and apply basic preprocessing."""
        if not HAS_MNE:
            raise ImportError("MNE is required for Resting_State_5year reader")

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            # Dataset uses EDF, despite the helper name
            raw = mne.io.read_raw_edf(str(vhdr_path), preload=True, verbose=False)

        # Drop non-EEG channels (EOG, ECG, etc.) heuristically by type
        eeg_picks = mne.pick_types(raw.info, eeg=True, meg=False, stim=False)
        raw.pick(eeg_picks)

        # Ensure units are Volts
        data = raw.get_data()
        data_v, unit = detect_unit_and_convert_to_volts(data)
        if unit != "V":
            raw._data = data_v

        # Bandpass + notch + resample
        if self.filter_notch > 0:
            raw.notch_filter(freqs=self.filter_notch, verbose=False)
        raw.filter(
            l_freq=self.filter_low,
            h_freq=self.filter_high,
            verbose=False,
        )
        if raw.info["sfreq"] != self.target_sfreq:
            raw.resample(self.target_sfreq, verbose=False)

        return raw

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _validate_segment(self, seg_uv: np.ndarray) -> bool:
        """Amplitude-based validation in µV."""
        return float(np.abs(seg_uv).max()) <= self.max_amplitude_uv

    # ------------------------------------------------------------------
    # Build per-subject HDF5
    # ------------------------------------------------------------------
    def build_subject(self, bids_id: str, label_idx: int) -> Optional[str]:
        """
        Build one HDF5 file for a given subject, containing all sessions & conditions.

        Returns:
            Path to HDF5 file, or None if no valid segments.
        """
        session_files = self._find_session_files(bids_id)
        if not session_files:
            print(f"[skip] {bids_id}: no EEG files found")
            return None

        print(f"\nSubject {bids_id}: {len(session_files)} session-condition files")

        all_segments: List[Tuple[np.ndarray, float, int, str, str]] = []
        ch_names: Optional[List[str]] = None

        for sf in session_files:
            print(f"  Reading {sf.vhdr_path.name} ({sf.session}, {sf.condition})")
            try:
                raw = self._read_raw_brainvision(sf.vhdr_path)
            except Exception as e:  # pragma: no cover - robust to bad subjects
                print(f"  [error] failed to read {sf.vhdr_path}: {e}")
                continue

            if ch_names is None:
                ch_names = raw.ch_names
                # Also cache globally for JSON summary
                if self.channels is None:
                    self.channels = ch_names

            data_v = raw.get_data()
            data_uv = data_v * 1e6  # V → µV
            n_ch, n_samp = data_uv.shape

            # Label: subject-as-label (global subject id index)
            label = label_idx

            # Windowing over entire continuous recording
            for start in range(0, n_samp - self.window_samples + 1, self.stride_samples):
                end = start + self.window_samples
                if end > n_samp:
                    break
                seg = data_uv[:, start:end]
                if seg.shape[1] != self.window_samples:
                    continue

                self.total_segments += 1
                if not self._validate_segment(seg):
                    self.rejected_segments += 1
                    continue

                self.valid_segments += 1
                all_segments.append(
                    (
                        seg,
                        self.window_sec,
                        label,
                        sf.session,
                        sf.condition,
                    )
                )

        if not all_segments or ch_names is None:
            print(f"  [warn] {bids_id}: no valid segments")
            return None

        # Subject & dataset info (subject-as-label, shared across all subjects)
        num_labels = len(self.label_categories) if self.label_categories else RESTING5Y_INFO.num_labels
        category_list = self.label_categories if self.label_categories else RESTING5Y_INFO.category_list
        channels = self.channels or ch_names

        dataset_info = DatasetInfo(
            dataset_name=RESTING5Y_INFO.dataset_name,
            task_type=RESTING5Y_INFO.task_type,
            downstream_task_type=RESTING5Y_INFO.downstream_task_type,
            num_labels=num_labels,
            category_list=category_list,
            sampling_rate=self.target_sfreq,
            montage=RESTING5Y_INFO.montage,
            channels=channels,
        )

        subject_attrs = SubjectAttrs(
            subject_id=bids_id,
            dataset_name=dataset_info.dataset_name,
            task_type=dataset_info.task_type.value,
            downstream_task_type=dataset_info.downstream_task_type.value,
            rsFreq=self.target_sfreq,
            chn_name=channels,
            num_labels=num_labels,
            category_list=category_list,
            chn_type="EEG",
            montage=dataset_info.montage,
        )

        # Output path
        self.output_root.mkdir(parents=True, exist_ok=True)
        out_path = self.output_root / f"{bids_id}.h5"

        # Single trial with many 2s segments
        with HDF5Writer(str(out_path), subject_attrs) as writer:
            trial_attrs = TrialAttrs(trial_id=1, session_id=1)
            trial_name = writer.add_trial(trial_attrs)

            for seg_id, (seg_data, time_len, label, session, condition) in enumerate(all_segments):
                label_arr = np.array([label], dtype=np.int64)
                seg_attrs = SegmentAttrs(
                    segment_id=seg_id,
                    start_time=seg_id * self.window_sec,
                    end_time=(seg_id + 1) * self.window_sec,
                    time_length=time_len,
                    label=label_arr,
                )
                writer.add_segment(trial_name, seg_attrs, seg_data)

        print(
            f"  [ok] {bids_id}: segments={len(all_segments)}, "
            f"out='{out_path}'"
        )
        return str(out_path)

    # ------------------------------------------------------------------
    # Summary & dataset info
    # ------------------------------------------------------------------
    def _save_dataset_info(self, stats: dict) -> None:
        """Save dataset info and processing parameters to JSON."""
        info = {
            "dataset": {
                "name": RESTING5Y_INFO.dataset_name,
                "description": (
                    "Longitudinal resting-state EEG (5-year follow-up) "
                    "with subject-as-label (one label id per subject)."
                ),
                "task_type": str(RESTING5Y_INFO.task_type.value),
                "downstream_task": str(RESTING5Y_INFO.downstream_task_type.value),
                "num_labels": len(self.label_categories),
                "category_list": self.label_categories,
                "original_sampling_rate": 1000.0,
                "channels": self.channels or [],
                "montage": RESTING5Y_INFO.montage,
                "source_url": "https://openneuro.org/datasets/ds005385",
            },
            "processing": {
                "target_sampling_rate": self.target_sfreq,
                "window_sec": self.window_sec,
                "stride_sec": self.stride_sec,
                "filter_low": self.filter_low,
                "filter_high": self.filter_high,
                "filter_notch": self.filter_notch,
                "max_amplitude_uv": self.max_amplitude_uv,
            },
            "statistics": stats,
            "generated_at": datetime.now().isoformat(),
        }

        self.output_root.mkdir(parents=True, exist_ok=True)
        json_path = self.output_root / "dataset_info.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)
        print(f"Saved dataset info to {json_path}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build_all(self, subject_ids: Optional[List[str]] = None) -> List[str]:
        """
        Build HDF5s for all subjects.

        Args:
            subject_ids: Optional list of BIDS subject IDs (e.g., ['sub-CTR007']).
                          If None, auto-discover all subjects.
        """
        if subject_ids is None:
            subject_ids = self.get_subject_ids()

        # Build global subject→label mapping (subject-as-label)
        subject_ids = sorted(subject_ids)
        self.subject_label_map = {bid: idx for idx, bid in enumerate(subject_ids)}
        self.label_categories = list(self.subject_label_map.keys())

        outputs: List[str] = []
        failed: List[str] = []

        # Reset stats
        self.total_segments = 0
        self.valid_segments = 0
        self.rejected_segments = 0

        print(f"Found {len(subject_ids)} subjects with EEG data.")

        for bids_id in subject_ids:
            try:
                label_idx = self.subject_label_map[bids_id]
                out = self.build_subject(bids_id, label_idx)
                if out is not None:
                    outputs.append(out)
            except Exception as e:  # pragma: no cover
                print(f"[error] failed to process {bids_id}: {e}")
                failed.append(bids_id)

        # Summary report
        print("\n" + "=" * 50)
        print("Resting_State_5year PROCESSING SUMMARY")
        print("=" * 50)
        print(f"Total subjects: {len(subject_ids)}")
        print(f"Successful: {len(outputs)}")
        print(f"Failed: {len(failed)}")
        if failed:
            print(f"Failed subjects: {failed}")
        print(f"Total segments: {self.total_segments}")
        print(f"Valid segments: {self.valid_segments}")
        print(f"Rejected segments: {self.rejected_segments}")
        if self.total_segments > 0:
            rej_rate = self.rejected_segments / self.total_segments * 100.0
            print(f"Rejection rate: {rej_rate:.1f}%")
        print("=" * 50)

        stats = {
            "total_subjects": len(subject_ids),
            "successful_subjects": len(outputs),
            "failed_subjects": len(failed),
            "total_segments": self.total_segments,
            "valid_segments": self.valid_segments,
            "rejected_segments": self.rejected_segments,
            "rejection_rate": (
                f"{self.rejected_segments / self.total_segments * 100.0:.1f}%"
                if self.total_segments > 0
                else "0%"
            ),
        }
        self._save_dataset_info(stats)

        return outputs


def build_resting_state_5year(
    raw_root: str,
    output_root: str = "./hdf5",
    target_sfreq: float = 200.0,
    window_sec: float = 2.0,
    stride_sec: float = 2.0,
    filter_low: float = 0.5,
    filter_high: float = 70.0,
    filter_notch: float = 60.0,
    max_amplitude_uv: float = DEFAULT_MAX_AMPLITUDE_UV,
    subject_ids: Optional[List[str]] = None,
) -> List[str]:
    """
    Convenience function to build Resting_State_5year dataset.
    """
    builder = Resting5YearBuilder(
        raw_root=raw_root,
        output_root=output_root,
        target_sfreq=target_sfreq,
        window_sec=window_sec,
        stride_sec=stride_sec,
        filter_low=filter_low,
        filter_high=filter_high,
        filter_notch=filter_notch,
        max_amplitude_uv=max_amplitude_uv,
    )
    return builder.build_all(subject_ids)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build Resting_State_5year HDF5 dataset from BIDS-formatted "
            "longitudinal resting-state EEG (CE vs OE)."
        )
    )
    parser.add_argument(
        "--raw_root",
        required=True,
        help=(
            "Path to BIDS root, e.g. "
            "/mnt/dataset2/novelDS/Longitudinal EEG Reliability"
        ),
    )
    parser.add_argument(
        "--output_root",
        default="./hdf5",
        help="Output root directory for HDF5 files (default: ./hdf5)",
    )
    parser.add_argument(
        "--target_sfreq",
        type=float,
        default=200.0,
        help="Target sampling rate (default: 200.0 Hz)",
    )
    parser.add_argument(
        "--window_sec",
        type=float,
        default=2.0,
        help="Window length in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--stride_sec",
        type=float,
        default=2.0,
        help="Stride length in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--filter_low",
        type=float,
        default=0.5,
        help="Low cutoff frequency for bandpass (default: 0.5 Hz)",
    )
    parser.add_argument(
        "--filter_high",
        type=float,
        default=70.0,
        help="High cutoff frequency for bandpass (default: 70.0 Hz)",
    )
    parser.add_argument(
        "--filter_notch",
        type=float,
        default=60.0,
        help="Notch frequency (default: 60.0 Hz; dataset collected in Americas)",
    )
    parser.add_argument(
        "--max_amplitude_uv",
        type=float,
        default=DEFAULT_MAX_AMPLITUDE_UV,
        help=(
            "Amplitude threshold in µV for segment rejection "
            f"(default: {DEFAULT_MAX_AMPLITUDE_UV})"
        ),
    )
    args = parser.parse_args()

    build_resting_state_5year(
        raw_root=args.raw_root,
        output_root=args.output_root,
        target_sfreq=args.target_sfreq,
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        filter_low=args.filter_low,
        filter_high=args.filter_high,
        filter_notch=args.filter_notch,
        max_amplitude_uv=args.max_amplitude_uv,
    )


if __name__ == "__main__":
    main()

