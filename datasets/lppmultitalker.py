"""LPPMulti-talker (OpenNeuro ds005345) EEG Dataset Builder.

Converts raw BrainVision EEG files:
  sub-XX/eeg/sub-XX_task-multitalker_eeg.vhdr
into one HDF5 file per subject.

Trial strategy: 1 trial per subject (whole continuous recording).
Segmentation: fixed windows after unified preprocessing.

Preprocessing defaults are aligned with the lab unified setting:
- bandpass: 0.1–75 Hz
- notch: 50 Hz
- resample: 200 Hz
- window/stride: 1s / 1s

Output:
- output_dir/LPPMultiTalker/sub-XX.h5
- subject attrs: num_labels=0 (unlabeled)
- trial attrs: trial0, session_id=0, task_name="multitalker"
- segment eeg stored in microvolts (uV)
"""

from __future__ import annotations

from pathlib import Path
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
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from schema import SubjectAttrs, TrialAttrs, SegmentAttrs
    from hdf5_io import HDF5Writer
    from config import DatasetInfo, DatasetTaskType, DownstreamTaskType


LPPMULTITALKER_INFO = DatasetInfo(
    dataset_name="LPPMultiTalker",
    task_type=DatasetTaskType.COGNITIVE,
    downstream_task_type=DownstreamTaskType.CLASSIFICATION,
    num_labels=0,
    category_list=[],
    sampling_rate=200.0,
    montage="standard_1020",
    channels=[],
)


class LPPMultiTalkerBuilder:
    """Builder for LPPMulti-talker EEG (raw BrainVision)."""

    def __init__(
        self,
        raw_data_dir: str,
        output_dir: str = "/mnt/dataset2/hdf5_datasets",
        target_sfreq: float = 200.0,
        window_sec: float = 1.0,
        stride_sec: float = 1.0,
        filter_low: float = 0.1,
        filter_high: float = 75.0,
        filter_notch: float = 50.0,
        max_subjects: int | None = None,
    ):
        if not HAS_MNE:
            raise ImportError("MNE is required for LPPMultiTalkerBuilder")

        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir) / LPPMULTITALKER_INFO.dataset_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.target_sfreq = float(target_sfreq)
        self.window_sec = float(window_sec)
        self.stride_sec = float(stride_sec)
        self.filter_low = float(filter_low)
        self.filter_high = float(filter_high)
        self.filter_notch = float(filter_notch)
        self.max_subjects = max_subjects

        self.window_samples = int(round(self.window_sec * self.target_sfreq))
        self.stride_samples = int(round(self.stride_sec * self.target_sfreq))

        if self.window_samples <= 0 or self.stride_samples <= 0:
            raise ValueError("window_sec/stride_sec must be positive")

    # --------------------------- discovery ---------------------------
    def get_subject_ids(self) -> list[str]:
        """Return subject IDs like ['sub-01', 'sub-02', ...]."""
        subs = sorted([p.name for p in self.raw_data_dir.glob("sub-*") if p.is_dir()])
        if self.max_subjects is not None:
            subs = subs[: int(self.max_subjects)]
        return subs

    def _vhdr_path(self, subject_id: str) -> Path:
        eeg_dir = self.raw_data_dir / subject_id / "eeg"
        if not eeg_dir.exists():
            raise FileNotFoundError(f"Missing eeg directory: {eeg_dir}")

        # Prefer BIDS standard name if present
        cand = eeg_dir / f"{subject_id}_task-multitalker_eeg.vhdr"
        if cand.exists():
            return cand

        # Fallback: any .vhdr in eeg dir
        vhdrs = list(eeg_dir.glob("*.vhdr"))
        if len(vhdrs) == 1:
            return vhdrs[0]
        if not vhdrs:
            raise FileNotFoundError(f"No .vhdr found under {eeg_dir}")
        raise FileNotFoundError(f"Multiple .vhdr found under {eeg_dir}: {[p.name for p in vhdrs]}")

    # ------------------------- preprocessing -------------------------
    def _preprocess_raw(self, raw: "mne.io.BaseRaw") -> "mne.io.BaseRaw":
        """Apply unified preprocessing to an MNE Raw object."""
        # Pick EEG channels only (drop stim/misc by default)
        try:
            picks = mne.pick_types(raw.info, eeg=True, eog=False, ecg=False, emg=False, stim=False, misc=False)
            if len(picks) > 0:
                raw.pick(picks)
        except Exception:
            # If type info is missing, keep all channels
            pass

        # Ensure data is loaded for filtering/resampling
        if not raw.preload:
            raw.load_data()

        # Notch first
        if self.filter_notch and self.filter_notch > 0:
            raw.notch_filter(freqs=[self.filter_notch], verbose=False)

        # Bandpass
        raw.filter(l_freq=self.filter_low, h_freq=self.filter_high, verbose=False)

        # Resample
        if float(raw.info["sfreq"]) != self.target_sfreq:
            raw.resample(self.target_sfreq, npad="auto", verbose=False)

        return raw

    # --------------------------- build ---------------------------
    def build_subject(self, subject_id: str) -> str:
        """Build one subject into a single HDF5 file."""
        out_file = self.output_dir / f"{subject_id}.h5"
        if out_file.exists():
            return str(out_file)

        vhdr_path = self._vhdr_path(subject_id)
        print(f"[{LPPMULTITALKER_INFO.dataset_name}] Building {subject_id} from {vhdr_path}")

        raw = mne.io.read_raw_brainvision(str(vhdr_path), preload=True, verbose="ERROR")
        raw = self._preprocess_raw(raw)

        # Data in MNE is Volts; convert to microvolts for HDF5
        data_uv = raw.get_data().astype(np.float32) * 1e6  # (n_ch, n_samples)
        ch_names = list(raw.ch_names)

        # Subject-level attrs (unlabeled)
        subject_attrs = SubjectAttrs(
            subject_id=subject_id,
            dataset_name=LPPMULTITALKER_INFO.dataset_name,
            task_type=LPPMULTITALKER_INFO.task_type.value,
            downstream_task_type=LPPMULTITALKER_INFO.downstream_task_type.value,
            rsFreq=self.target_sfreq,
            chn_name=ch_names,
            num_labels=LPPMULTITALKER_INFO.num_labels,
            category_list=LPPMULTITALKER_INFO.category_list,
            chn_type="EEG",
            montage=LPPMULTITALKER_INFO.montage,
        )

        # One trial per subject
        trial_attrs = TrialAttrs(trial_id=0, session_id=0, task_name="multitalker")

        n_samples = data_uv.shape[1]
        if n_samples < self.window_samples:
            raise RuntimeError(
                f"Recording too short for one window: samples={n_samples}, window_samples={self.window_samples}"
            )

        starts = list(range(0, n_samples - self.window_samples + 1, self.stride_samples))
        print(
            f"  sfreq={self.target_sfreq}Hz, channels={data_uv.shape[0]}, samples={n_samples}, "
            f"segments={len(starts)} (window={self.window_sec}s, stride={self.stride_sec}s)"
        )

        with HDF5Writer(str(out_file), subject_attrs) as writer:
            trial_name = writer.add_trial(trial_attrs)

            for seg_id, start in enumerate(starts):
                end = start + self.window_samples
                seg = data_uv[:, start:end]

                start_time = start / self.target_sfreq
                end_time = end / self.target_sfreq

                segment_attrs = SegmentAttrs(
                    segment_id=seg_id,
                    start_time=float(start_time),
                    end_time=float(end_time),
                    time_length=self.window_sec,
                    label=np.array([], dtype=np.int64),  # unlabeled
                    task_label="",
                )
                writer.add_segment(trial_name, segment_attrs, seg)

        print(f"  ✓ Wrote {out_file}")
        return str(out_file)

    def build(self, subjects: list[str] | None = None) -> list[str]:
        """Convenience method to build multiple subjects."""
        if subjects is None:
            subjects = self.get_subject_ids()
        outputs = []
        for sid in subjects:
            try:
                outputs.append(self.build_subject(sid))
            except Exception as e:
                print(f"  ! Failed {sid}: {e}")
        return outputs
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build LPPMultiTalker HDF5 dataset")
    parser.add_argument("raw_data_dir", type=str, help="Raw dataset root (contains sub-*/eeg/*.vhdr)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for HDF5 files")
    parser.add_argument("--subjects", nargs="*", type=str, default=None, help="Subject IDs like sub-01 sub-02 ...")
    parser.add_argument("--max_subjects", type=int, default=None, help="Limit number of subjects (for debugging)")

    args = parser.parse_args()

    builder = LPPMultiTalkerBuilder(
        raw_data_dir=args.raw_data_dir,
        output_dir=args.output_dir,
        # 预处理统一规范：写死即可（你文件顶部也说明了 0.1-75 / 50 / 200 / 1s1s）
        target_sfreq=200.0,
        window_sec=1.0,
        stride_sec=1.0,
        filter_low=0.1,
        filter_high=75.0,
        filter_notch=50.0,
        max_subjects=args.max_subjects,
    )

    if args.subjects is None or len(args.subjects) == 0:
        outs = builder.build()
    else:
        outs = builder.build(args.subjects)

    print(f"Done. Built {len(outs)} subjects.")

