"""
Process a single TUAB EDF file and write HDF5 segments.

Usage:
    python -m benchmark_dataloader.datasets.tuab_single /path/to/file.edf --output_dir ./hdf5

This script reads an EDF via MNE, applies notch/bandpass/resample,
segments into fixed windows and writes to an HDF5 using HDF5Writer.
"""
from pathlib import Path
import argparse
import sys
import numpy as np

try:
    import mne
    HAS_MNE = True
except Exception:
    HAS_MNE = False

# Ensure imports work when running as a module or script
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "train_dataloader"))

try:
    from benchmark_dataloader.schema import SubjectAttrs, TrialAttrs, SegmentAttrs
    from train_dataloader.hdf5_io import HDF5Writer
except Exception:
    try:
        from train_dataloader.schema import SubjectAttrs, TrialAttrs, SegmentAttrs
        from train_dataloader.hdf5_io import HDF5Writer
    except Exception:
        try:
            from schema import SubjectAttrs, TrialAttrs, SegmentAttrs
            from hdf5_io import HDF5Writer
        except Exception:
            raise


def detect_unit_and_convert_to_volts(data: np.ndarray) -> tuple[np.ndarray, str]:
    max_amp = np.abs(data).max()
    if max_amp > 1e-2:
        return data / 1e6, "µV"
    elif max_amp > 1e-5:
        return data / 1e3, "mV"
    else:
        return data, "V"


class TUABSingleBuilder:
    def __init__(
        self,
        edf_path: str,
        output_dir: str = "./hdf5",
        target_sfreq: float = 100.0,
        window_sec: float = 4.0,
        stride_sec: float = 4.0,
        filter_low: float = 0.5,
        filter_high: float = 70.0,
        filter_notch: float = 60.0,
        max_amplitude_uv: float = 600.0,
    ):
        self.edf_path = Path(edf_path)
        self.output_dir = Path(output_dir)
        self.target_sfreq = float(target_sfreq)
        self.window_sec = float(window_sec)
        self.stride_sec = float(stride_sec)
        self.filter_low = filter_low
        self.filter_high = filter_high
        self.filter_notch = filter_notch
        self.window_samples = int(self.window_sec * self.target_sfreq)
        self.stride_samples = int(self.stride_sec * self.target_sfreq)
        self.max_amplitude_uv = max_amplitude_uv

    def _read_raw(self):
        if not HAS_MNE:
            raise ImportError("MNE is required to read EDF files")

        raw = mne.io.read_raw_edf(str(self.edf_path), preload=True, verbose=False)
        return raw

    def _preprocess(self, raw):
        # Drop common non-EEG channels if present
        drop_chs = [ch for ch in raw.ch_names if ch.lower().startswith(("ekg", "emg", "eog", "resp"))]
        if drop_chs:
            raw.drop_channels(drop_chs)

        if self.filter_notch and self.filter_notch > 0:
            try:
                raw.notch_filter(freqs=self.filter_notch, verbose=False)
            except Exception:
                pass

        try:
            raw.filter(l_freq=self.filter_low, h_freq=self.filter_high, verbose=False)
        except Exception:
            pass

        if raw.info["sfreq"] != self.target_sfreq:
            try:
                raw.resample(self.target_sfreq, verbose=False)
            except Exception:
                pass

        return raw

    def build(self) -> str:
        raw = self._read_raw()
        raw = self._preprocess(raw)

        ch_names = raw.ch_names

        # Get data in µV for trial-level validation and writing
        data_volts = raw.get_data()  # MNE returns Volts
        data_uv = data_volts * 1e6

        # Single 'trial' covering the whole recording
        trial_id = 1
        session_id = 0

        subject_id = 0
        try:
            stem = self.edf_path.stem
            subject_id = int(''.join(filter(str.isdigit, stem))[:4]) if any(c.isdigit() for c in stem) else 0
        except Exception:
            subject_id = 0

        subject_attrs = SubjectAttrs(
            subject_id=int(subject_id),
            dataset_name="TUAB_single",
            task_type="seizure_detection",
            downstream_task_type="detection",
            rsFreq=float(self.target_sfreq),
            chn_name=ch_names,
            num_labels=0,
            category_list=[],
            chn_pos=None,
            chn_ori=None,
            chn_type="EEG",
            montage="unknown",
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.output_dir / f"{self.edf_path.stem}.h5"

        with HDF5Writer(str(out_path), subject_attrs) as writer:
            trial_attrs = TrialAttrs(trial_id=trial_id, session_id=session_id)
            trial_name = writer.add_trial(trial_attrs)

            n_samples = data_uv.shape[1]
            seg_id = 0
            for start in range(0, n_samples - self.window_samples + 1, self.stride_samples):
                end = start + self.window_samples
                slice_data = data_uv[:, start:end]

                # Amplitude validation
                if np.abs(slice_data).max() > self.max_amplitude_uv:
                    seg_id += 1
                    continue

                segment_attrs = SegmentAttrs(
                    segment_id=seg_id,
                    start_time=float(start / self.target_sfreq),
                    end_time=float(end / self.target_sfreq),
                    time_length=float(self.window_sec),
                    label=np.array([0]),
                )
                writer.add_segment(trial_name, segment_attrs, slice_data)
                seg_id += 1

        print(f"Saved {out_path} with {seg_id} segments")
        return str(out_path)


def main():
    parser = argparse.ArgumentParser(description="Process a single TUAB EDF file to HDF5 segments")
    parser.add_argument("edf", help="Path to EDF file")
    parser.add_argument("--output_dir", default="./hdf5", help="Output directory")
    parser.add_argument("--sfreq", type=float, default=100.0, help="Target sampling rate (Hz)")
    parser.add_argument("--window", type=float, default=4.0, help="Window length (s)")
    parser.add_argument("--stride", type=float, default=4.0, help="Window stride (s)")
    parser.add_argument("--notch", type=float, default=60.0, help="Notch frequency")
    parser.add_argument("--max_amp_uv", type=float, default=600.0, help="Max amplitude µV")
    args = parser.parse_args()

    builder = TUABSingleBuilder(
        args.edf,
        output_dir=args.output_dir,
        target_sfreq=args.sfreq,
        window_sec=args.window,
        stride_sec=args.stride,
        filter_notch=args.notch,
        max_amplitude_uv=args.max_amp_uv,
    )
    builder.build()


if __name__ == "__main__":
    main()
