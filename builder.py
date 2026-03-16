"""
Base builder class for EEG dataset preprocessing.

Provides abstract base class for building HDF5 datasets from raw EEG files.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
import numpy as np

try:
    import mne
    HAS_MNE = True
except ImportError:
    HAS_MNE = False

try:
    from .schema import SubjectAttrs, TrialAttrs, SampleAttrs
    from .hdf5_io import HDF5Writer
    from .config import PreprocConfig, DatasetInfo
except ImportError:
    # Direct import (when run as script or imported from script)
    from schema import SubjectAttrs, TrialAttrs, SampleAttrs
    from hdf5_io import HDF5Writer
    from config import PreprocConfig, DatasetInfo


class EEGDatasetBuilder(ABC):
    """Abstract base class for building EEG datasets."""

    def __init__(
        self,
        dataset_info: DatasetInfo,
        preproc_config: PreprocConfig,
        raw_data_dir: str,
    ):
        """
        Initialize the dataset builder.

        Args:
            dataset_info: Dataset-level information
            preproc_config: Preprocessing configuration
            raw_data_dir: Path to raw data directory
        """
        self.dataset_info = dataset_info
        self.preproc_config = preproc_config
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(preproc_config.output_dir) / dataset_info.dataset_name

    @abstractmethod
    def get_subject_ids(self) -> list:
        """Get list of subject IDs in the dataset."""
        pass

    @abstractmethod
    def get_raw_file_path(self, subject_id) -> Path:
        """Get path to raw EEG file for a subject."""
        pass

    @abstractmethod
    def get_trial_info(self, subject_id) -> list[dict]:
        """
        Get trial information for a subject.

        Returns:
            List of dicts with keys: trial_id, session_id, start_sec, end_sec, label
        """
        pass

    def preprocess_raw(self, raw) -> np.ndarray:
        """
        Apply preprocessing to raw MNE object.

        Args:
            raw: MNE Raw object

        Returns:
            Preprocessed data array (n_channels, n_timepoints)
        """
        if not HAS_MNE:
            raise ImportError("MNE is required for preprocessing")

        # Notch filter
        if self.preproc_config.filter_notch > 0:
            raw.notch_filter(freqs=self.preproc_config.filter_notch, verbose=False)

        # Bandpass filter
        raw.filter(
            l_freq=self.preproc_config.filter_low,
            h_freq=self.preproc_config.filter_high,
            verbose=False,
        )

        # Resample
        if raw.info['sfreq'] != self.preproc_config.target_sfreq:
            raw.resample(self.preproc_config.target_sfreq, verbose=False)

        return raw.get_data()

    def segment_trial(
        self,
        data: np.ndarray,
        start_sec: float,
        end_sec: float,
    ) -> list[tuple[np.ndarray, float]]:
        """
        Segment a trial into windows.

        Args:
            data: Full recording data (n_channels, n_timepoints)
            start_sec: Trial start time in seconds
            end_sec: Trial end time in seconds

        Returns:
            List of (segment_data, time_length) tuples
        """
        sfreq = self.preproc_config.target_sfreq
        window_samples = int(self.preproc_config.window_sec * sfreq)
        stride_samples = int(self.preproc_config.stride_sec * sfreq)

        start_sample = int(start_sec * sfreq)
        end_sample = int(end_sec * sfreq)

        trial_data = data[:, start_sample:end_sample]
        n_samples = trial_data.shape[1]

        segments = []
        for start in range(0, n_samples - window_samples + 1, stride_samples):
            end = start + window_samples
            segment = trial_data[:, start:end]
            segments.append((segment, self.preproc_config.window_sec))

        return segments

    def build_subject(self, subject_id) -> Optional[str]:
        """
        Build HDF5 file for a single subject.

        Args:
            subject_id: Subject identifier

        Returns:
            Path to output file, or None if failed
        """
        if not HAS_MNE:
            raise ImportError("MNE is required for building datasets")

        raw_path = self.get_raw_file_path(subject_id)
        if not raw_path.exists():
            print(f"Raw file not found: {raw_path}")
            return None

        # Load and preprocess
        raw = mne.io.read_raw(str(raw_path), preload=True, verbose=False)
        ch_names = raw.ch_names
        data = self.preprocess_raw(raw)

        # Create subject attributes
        subject_attrs = SubjectAttrs(
            subject_id=subject_id,
            dataset_name=self.dataset_info.dataset_name,
            task_type=self.dataset_info.task_type.value,
            rsFreq=self.preproc_config.target_sfreq,
            chn_name=ch_names,
            chn_pos=None,
            chn_ori=None,
            chn_type="EEG",
            montage=self.dataset_info.montage,
        )

        # Create output file
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / f"sub_{subject_id}.h5"

        with HDF5Writer(str(output_path), subject_attrs) as writer:
            trial_infos = self.get_trial_info(subject_id)

            for trial_info in trial_infos:
                trial_attrs = TrialAttrs(
                    trial_id=trial_info['trial_id'],
                    session_id=trial_info['session_id'],
                )
                trial_name = writer.add_trial(trial_attrs)

                # Segment trial
                segments = self.segment_trial(
                    data,
                    trial_info['start_sec'],
                    trial_info['end_sec'],
                )

                for segment_id, (segment_data, time_length) in enumerate(segments):
                    sample_attrs = SampleAttrs(
                        segment_id=segment_id,
                        time_length=time_length,
                        label=np.array(trial_info['label']),
                    )
                    writer.add_sample(trial_name, sample_attrs, segment_data)

        return str(output_path)

    def build_all(self) -> list[str]:
        """
        Build HDF5 files for all subjects.

        Returns:
            List of output file paths
        """
        output_paths = []
        subject_ids = self.get_subject_ids()

        for subject_id in subject_ids:
            print(f"Processing subject {subject_id}...")
            output_path = self.build_subject(subject_id)
            if output_path:
                output_paths.append(output_path)

        return output_paths
