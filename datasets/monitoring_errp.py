"""
Monitoring_Errp-8Class: Brief description.
- 6 subjects
- 2 sessions per subject
- 10 trials per session
- 8 classes: 
- 64 channels
- link to dataset: https://lampx.tugraz.at/~bci/database/013-2015/description.pdf
"""

from pathlib import Path
from datetime import datetime
import json
import numpy as np

try:
    import mne
    from scipy.io import loadmat
    HAS_MNE = True
except ImportError:
    HAS_MNE = False

try:
    from ..schema import SubjectAttrs, TrialAttrs, SegmentAttrs
    from ..hdf5_io import HDF5Writer
    from ..config import DatasetInfo, DatasetTaskType, DownstreamTaskType
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from schema import SubjectAttrs, TrialAttrs, SegmentAttrs
    from hdf5_io import HDF5Writer
    from config import DatasetInfo, DatasetTaskType, DownstreamTaskType


# Dataset configuration
MERRP_INFO = DatasetInfo(
    dataset_name="Monitoring_Errp-8Class",
    task_type=DatasetTaskType.COGNITIVE,
    downstream_task_type=DownstreamTaskType.CLASSIFICATION,
    num_labels=8,
    category_list=["1:Target located in the left ","2:Target located in the right",
                   "4:Cursor movement to the left ","5:Target located in the left and Cursor movement to the left",
                   "6:Target located in the right and Cursor movement to the left","8:Cursor movement to the right",
                   "9:Target located in the left and Cursor movement to the right","10:Target located in the right and Cursor movement to the right"],
    sampling_rate=512.0,
    montage="extended 10_20",
    channels=['Fp1','AF7','AF3','F1','F3','F5','F7','FT7','FC5','FC3',
              'FC1','C1','C3','C5','T7','TP7','CP5','CP3','CP1','P1',
              'P3','P5','P7','P9','PO7','PO3','O1','Iz','Oz','POz',
              'Pz','CPz','Fpz','Fp2','AF8','AF4','AFz','Fz','F2','F4',
              'F6','F8','FT8','FC6','FC4','FC2','FCz','Cz','C2','C4',
              'C6','T8','TP8','CP6','CP4','CP2','P2','P4','P6','P8',
              'P10','PO8','PO4','O2'],
)

# ## TODO: define label mapping or metadata if needed
YOUR_DATASET_LABELS = {
    '1': 'Target located in the left',
    '2': 'Target located in the right',
    '4': 'Cursor movement to the left',
    '5': 'Target located in the left and Cursor movement to the left',
    '6': 'Target located in the right and Cursor movement to the left',
    '8': 'Cursor movement to the right',
    '9': 'Target located in the left and Cursor movement to the right',
    '10': 'Target located in the right and Cursor movement to the right',
}

# ## TODO: list reference channels to remove, empty list if none
REMOVE_CHANNELS = []

# Default amplitude threshold (uV), check the final report after building dataset, if rejected segments are too many, let's DISCUSS and adjust this value
DEFAULT_MAX_AMPLITUDE_UV = 600.0


def detect_unit_and_convert_to_volts(data: np.ndarray) -> tuple[np.ndarray, str]:
    """Auto-detect data unit and convert to Volts for MNE."""
    max_amp = np.abs(data).max()
    if max_amp > 1e-3:  # likely uV
        return data / 1e6, "uV"

    return data, "V"


class MERRPBuilder:
    """Builder for MERRP."""

    def __init__(
        self,
        raw_data_dir: str,
        output_dir: str = "./hdf5",
        target_sfreq: float = 200.0,
        window_sec: float = 1.0,
        stride_sec: float = 1.0,
        filter_low: float = 0.1,
        filter_high: float = 75.0,
        filter_notch: float = 50.0,
        max_amplitude_uv: float = DEFAULT_MAX_AMPLITUDE_UV,
    ):
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir) / "Monitoring_Errp_test" 
        self.target_sfreq = target_sfreq
        self.orig_sfreq = 512.0
        self.window_sec = window_sec
        self.stride_sec = stride_sec
        self.filter_low = filter_low
        self.filter_high = filter_high
        self.filter_notch = filter_notch
        self.max_amplitude_uv = max_amplitude_uv

        self.window_samples = int(window_sec * target_sfreq)
        self.stride_samples = int(stride_sec * target_sfreq)

        # Validation statistics
        self.total_trials = 0
        self.valid_trials = 0
        self.rejected_trials = 0

    def get_subject_ids(self) -> list[int]:
        """Get list of subject IDs."""
        return list(range(1, 7))

    def _find_files(self, subject_id: int):
        """Find all files for a subject."""
        sub_prefix = f'Subject{subject_id:02d}'
        files = [
            self.raw_data_dir / f'{sub_prefix}_s1.mat',
            self.raw_data_dir / f'{sub_prefix}_s2.mat'
        ]
        for fp in files:
            if not fp.is_file():
                raise FileNotFoundError(fp)
            else:
                print(fp)

        return files

    def _read_raw(self, file_path: Path):
        """Read raw EEG file and convert to MNE Raw object."""
        data_chunks = []          
        onset_sec_all = [] 
        offset_sec_all=[]       
        desc_all = []             
        sfreq = None
        ch_names = None
        cum_samples = 0 

        for sess_id, path in enumerate(file_path, 1):
            mat = loadmat(path, simplify_cells=True)
            runs = mat['run']
            n_run = len(runs)               # 10 runs
            for run_idx, run in enumerate(runs):
                eeg = run['eeg'].T          # (n_ch, n_samples)
                data_chunks.append(eeg)
                if sfreq is None:
                    hdr = run['header']
                    sfreq = float(hdr['SampleRate'])
                    ch_names = MERRP_INFO.channels
                header=run['header']
                evt = header['EVENT']
                onset_sec = evt['POS']+cum_samples
                onset_sec_all.extend(onset_sec/sfreq)
                offset_sec = np.concatenate([onset_sec[1:],[cum_samples + eeg.shape[1]]])
                offset_sec_all.extend(offset_sec/sfreq)
                desc_all.extend([c for c in np.atleast_1d(evt['TYP'])])
                cum_samples+=eeg.shape[1]
                print(onset_sec)

        data = np.concatenate(data_chunks, axis=1)  # (n_ch, total_samples)
        data, unit = detect_unit_and_convert_to_volts(data)

        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
        info['description'] = f'Concatenated 2 sessions, converted from {unit} to V'

        raw = mne.io.RawArray(data, info, first_samp=0)

        if onset_sec_all:
            raw.set_annotations(
                mne.Annotations(onset=onset_sec_all,duration=0.0,description=desc_all)
            )
        return raw

    def _validate_trial(self, trial_data: np.ndarray) -> bool:
        """Check if trial amplitude is within acceptable range."""
        return np.abs(trial_data).max() <= self.max_amplitude_uv

    def _report_validation_stats(self, subject_id: int, is_resting: bool = True):
        """Report validation statistics."""
        unit = "segments" if is_resting else "trials"
        valid_pct = (self.valid_trials / self.total_trials * 100) if self.total_trials > 0 else 0
        print(f"Subject {subject_id} validation report:")
        print(f"  Amplitude threshold: {self.max_amplitude_uv} uV")
        print(f"  Total {unit}: {self.total_trials}")
        print(f"  Valid {unit}: {self.valid_trials} ({valid_pct:.1f}%)")
        print(f"  Rejected {unit}: {self.rejected_trials} ({100 - valid_pct:.1f}%)")

    def _preprocess(self, raw):
        """Apply preprocessing to raw data."""
        # Drop target remove channels if needed, check with the dataset documentation
        if REMOVE_CHANNELS:
            raw.drop_channels(REMOVE_CHANNELS)
        # Notch filter
        if self.filter_notch > 0:
            raw.notch_filter(freqs=self.filter_notch, verbose=False)

        # Bandpass filter
        raw.filter(l_freq=self.filter_low, h_freq=self.filter_high, verbose=False)

        # Resample if needed
        if raw.info["sfreq"] != self.target_sfreq:
            raw.resample(self.target_sfreq, verbose=False)

        return raw
    def _extract_trials(self,raw):
        """
        annotation.onset is the start of trial 
        label annotation.description。
        """
        trials = []
        annot = raw.annotations
        sfreq = raw.info['sfreq']
        data = raw.get_data()                      # (n_ch, n_times)
        max_samp = data.shape[1]

        for idx, (onset, desc) in enumerate(zip(annot.onset, annot.description)):
            # onset
            start_samp = int(onset * self.target_sfreq)
            # offset
            if idx + 1 < len(annot):
                end_samp = int(annot.onset[idx + 1] * self.target_sfreq)
            else:
                end_samp = max_samp
            if start_samp >= max_samp:            
                continue

            trial_data = data[:, start_samp:end_samp]
            # calculate offset_time
            if idx + 1 < len(annot):
                offset_time = annot.onset[idx + 1]
            else:
                offset_time = end_samp / self.target_sfreq
            trials.append({
                'data': trial_data,
                'label': desc,                    
                'trial_id': idx,
                'onset_time': onset,
                'offset_time': offset_time,
            })
        return trials
    def _save_dataset_info(self, stats: dict) -> None:
        """Save dataset info and processing parameters to JSON."""
        info = {
            "dataset": {
                "name": MERRP_INFO.dataset_name,
                "task_type": str(MERRP_INFO.task_type.value),
                "downstream_task": str(MERRP_INFO.downstream_task_type.value),
                "num_labels": MERRP_INFO.num_labels,
                "category_list": MERRP_INFO.category_list,
                "label_explanation":["Bit Description",
                                     "0 Target located in the left", 
                                     "1 Target located in the right", 
                                     "2 Cursor movement to the left", 
                                     "3 Cursor movement to the right",
                                     "label: 1 2 4 5 6 8 9 10",
                                     "5 and 10 are correct movements",
                                     "6 and 9 are erroneous movements"],
                "original_sampling_rate": MERRP_INFO.sampling_rate,
                "channels": MERRP_INFO.channels,
                "montage": MERRP_INFO.montage,
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

        json_path = self.output_dir / "dataset_info.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)
        print(f"Saved dataset info to {json_path}")

    def build_subject(self, subject_id: int) -> str:
        """Build HDF5 file for a single subject."""
        if not HAS_MNE:
            raise ImportError("MNE is required to build this dataset")

        self.total_trials = 0
        self.valid_trials = 0
        self.rejected_trials = 0

        files = self._find_files(subject_id)
        if not files:
            raise FileNotFoundError(f"No files found for subject {subject_id}")

        all_trials = []
        ch_names = ['Fp1','AF7','AF3','F1','F3','F5','F7','FT7','FC5','FC3',
              'FC1','C1','C3','C5','T7','TP7','CP5','CP3','CP1','P1',
              'P3','P5','P7','P9','PO7','PO3','O1','Iz','Oz','POz',
              'Pz','CPz','Fpz','Fp2','AF8','AF4','AFz','Fz','F2','F4',
              'F6','F8','FT8','FC6','FC4','FC2','FCz','Cz','C2','C4',
              'C6','T8','TP8','CP6','CP4','CP2','P2','P4','P6','P8',
              'P10','PO8','PO4','O2']
       
        raw = self._read_raw(files)
        raw = self._preprocess(raw)
        all_trials=self._extract_trials(raw)

        if not all_trials:
            raise ValueError(f"No valid trials extracted for subject {subject_id}")

        subject_attrs = SubjectAttrs(
            subject_id=subject_id,
            dataset_name=MERRP_INFO.dataset_name,
            task_type=MERRP_INFO.task_type.value,
            downstream_task_type=MERRP_INFO.downstream_task_type.value,
            rsFreq=self.target_sfreq,
            chn_name=ch_names,
            num_labels=MERRP_INFO.num_labels,
            category_list=MERRP_INFO.category_list,
            chn_pos=None,
            chn_ori=None,
            chn_type="EEG",
            montage=MERRP_INFO.montage,
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / f"sub_{subject_id}.h5"

        with HDF5Writer(str(output_path), subject_attrs) as writer:
            for trial in all_trials:
                trial_data = trial["data"]
                trial_data= trial_data*1e6
                # For non-resting-state tasks, validate at trial level
                if MERRP_INFO.task_type != DatasetTaskType.RESTING_STATE:
                    self.total_trials += 1
                    if not self._validate_trial(trial_data):
                        self.rejected_trials += 1
                        continue
                    self.valid_trials += 1

                trial_attrs = TrialAttrs(
                    trial_id=trial["trial_id"],
                    session_id=0,
                )
                trial_name = writer.add_trial(trial_attrs)

                n_samples = trial_data.shape[1]
                for i_slice, start in enumerate(
                    range(0, n_samples - self.window_samples + 1, self.stride_samples)
                ):
                    end = start + self.window_samples
                    slice_data = trial_data[:, start:end]

                    if MERRP_INFO.task_type == DatasetTaskType.RESTING_STATE:
                        self.total_trials += 1
                        if not self._validate_trial(slice_data):
                            self.rejected_trials += 1
                            continue
                        self.valid_trials += 1

                    segment_attrs = SegmentAttrs(
                        segment_id=i_slice,
                        start_time=trial.get("onset_time", 0.0) + start / self.target_sfreq,
                        end_time=trial.get("onset_time", 0.0) + end / self.target_sfreq,
                        time_length=self.window_sec,
                        label=int(trial["label"]),
                    )
                    writer.add_segment(trial_name, segment_attrs, slice_data)

        is_resting = MERRP_INFO.task_type == DatasetTaskType.RESTING_STATE
        self._report_validation_stats(subject_id, is_resting)
        print(f"Saved {output_path}")
        return str(output_path)

    def build_all(self, subject_ids: list[int] | None = None) -> list[str]:
        """Build HDF5 files for all subjects."""
        if subject_ids is None:
            subject_ids = self.get_subject_ids()

        output_paths = []
        failed_subjects = []
        all_total_trials = 0
        all_valid_trials = 0
        all_rejected_trials = 0

        for subject_id in subject_ids:
            try:
                output_path = self.build_subject(subject_id)
                output_paths.append(output_path)
                all_total_trials += self.total_trials
                all_valid_trials += self.valid_trials
                all_rejected_trials += self.rejected_trials
            except Exception as exc:
                print(f"Error processing subject {subject_id}: {exc}")
                failed_subjects.append(subject_id)

        stats = {
            "total_subjects": len(subject_ids),
            "successful": len(output_paths),
            "failed": len(failed_subjects),
            "failed_subject_ids": failed_subjects,
            "total_trials_or_segments": all_total_trials,
            "valid_trials_or_segments": all_valid_trials,
            "rejected_trials_or_segments": all_rejected_trials,
        }
        self._save_dataset_info(stats)

        return output_paths


def build_your_dataset(
    raw_data_dir: str,
    output_dir: str = "./hdf5",
    subject_ids: list[int] | None = None,
    **kwargs,
) -> list[str]:
    """Convenience function to build YOUR_DATASET."""
    builder = MERRPBuilder(raw_data_dir, output_dir, **kwargs)
    return builder.build_all(subject_ids)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build Monitoring Errp HDF5 dataset")
    parser.add_argument("raw_data_dir", help="Directory containing raw files")
    parser.add_argument("--output_dir", default="/mnt/dataset2/Processed_datasets/EEG_Bench", help="Output directory")
    parser.add_argument("--subjects", nargs="+", type=int, help="Subject IDs to process")
    args = parser.parse_args()

    build_your_dataset(args.raw_data_dir, args.output_dir, args.subjects)
