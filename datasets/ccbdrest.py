"""
CCBDRest Dataset Builder.

CCBDRest Dataset
- Resting state EEG data
- EEGLAB .set file format
- 1000 Hz original sampling rate
- 32 channels (10-20 montage)
"""

import sys
import json
import warnings
from pathlib import Path
from datetime import datetime
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
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from schema import SubjectAttrs, TrialAttrs, SegmentAttrs
    from hdf5_io import HDF5Writer
    from config import DatasetInfo, DatasetTaskType, DownstreamTaskType

CCBDREST_INFO = DatasetInfo(
    dataset_name="CCBDRest",
    task_type=DatasetTaskType.RESTING_STATE,
    downstream_task_type=DownstreamTaskType.CLASSIFICATION,
    num_labels=1,  # Resting state - single label
    category_list=["resting"],
    sampling_rate=1000.0,  # Original sampling rate
    montage="10_20",
    channels=["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", 
              "F7", "F8", "T7", "T8", "P7", "P8", "Fz", "Cz", "Pz", "Oz",
              "FC1", "FC2", "CP1", "CP2", "FC5", "FC6", "CP5", "CP6", 
              "PO7", "PO8", "M1", "M2"],
)

# Default amplitude threshold (µV)
DEFAULT_MAX_AMPLITUDE_UV = 600.0

# Reference channels (M1, M2) - will be kept as EEG channels for now
REFERENCE_CHANNELS = ['M1', 'M2']


class CCBDRestBuilder:
    """Builder for CCBDRest dataset."""

    def __init__(
        self,
        raw_data_dir: str,
        output_dir: str = "./hdf5",
        target_sfreq: float = 200.0,
        window_sec: float = 2.0,
        stride_sec: float = 2.0,
        filter_low: float = 0.1,
        filter_high: float = 75.0,
        filter_notch: float = 50.0,
        max_amplitude_uv: float = DEFAULT_MAX_AMPLITUDE_UV,
    ):
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir) / "CCBDRest"
        self.target_sfreq = target_sfreq
        self.orig_sfreq = 1000.0
        self.window_sec = window_sec
        self.stride_sec = stride_sec
        self.filter_low = filter_low
        self.filter_high = filter_high
        self.filter_notch = filter_notch
        self.max_amplitude_uv = max_amplitude_uv

        self.window_samples = int(window_sec * target_sfreq)
        self.stride_samples = int(stride_sec * target_sfreq)

        # Validation statistics
        self.total_segments = 0
        self.valid_segments = 0
        self.rejected_segments = 0

    def _validate_segment(self, segment_data: np.ndarray) -> bool:
        """Check if segment amplitude is within acceptable range."""
        return np.abs(segment_data).max() <= self.max_amplitude_uv

    def _report_validation_stats(self, subject_id: str):
        """Report validation statistics."""
        valid_pct = (self.valid_segments / self.total_segments * 100) if self.total_segments > 0 else 0
        print(f"Subject {subject_id} validation report:")
        print(f"  Amplitude threshold: {self.max_amplitude_uv} µV")
        print(f"  Total segments: {self.total_segments}")
        print(f"  Valid segments: {self.valid_segments} ({valid_pct:.1f}%)")
        print(f"  Rejected segments: {self.rejected_segments} ({100-valid_pct:.1f}%)")

    def _save_dataset_info(self, stats: dict):
        """Save dataset info and processing parameters to JSON."""
        info = {
            "dataset": {
                "name": CCBDREST_INFO.dataset_name,
                "description": "CCBDRest Dataset - Resting State EEG",
                "task_type": str(CCBDREST_INFO.task_type.value),
                "downstream_task": str(CCBDREST_INFO.downstream_task_type.value),
                "num_labels": CCBDREST_INFO.num_labels,
                "category_list": CCBDREST_INFO.category_list,
                "original_sampling_rate": CCBDREST_INFO.sampling_rate,
                "channels": CCBDREST_INFO.channels,
                "montage": CCBDREST_INFO.montage,
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
        with open(json_path, "w") as f:
            json.dump(info, f, indent=2)
        print(f"Saved dataset info to {json_path}")

    def get_subject_ids(self) -> list[str]:
        """Get list of subject IDs by scanning directory."""
        subject_ids = []
        if not self.raw_data_dir.exists():
            return subject_ids
        
        # Scan for directories containing .set files
        # Expected structure: raw_data_dir/{subject_id}/{subject_id}.set
        for item in self.raw_data_dir.iterdir():
            if item.is_dir():
                # Check if this directory contains a .set file with matching name
                set_files = list(item.glob("*.set"))
                if set_files:
                    # Use directory name as subject_id
                    subject_ids.append(item.name)
        
        return sorted(subject_ids)

    def _find_files(self, subject_id: str) -> Path:
        """
        Find .set file for a subject.
        Expected structure: raw_data_dir/{subject_id}/{subject_id}.set
        """
        subject_dir = self.raw_data_dir / subject_id
        if not subject_dir.exists():
            raise FileNotFoundError(f"Subject directory not found: {subject_dir}")
        
        # Try exact match first: {subject_id}.set
        set_file = subject_dir / f"{subject_id}.set"
        if set_file.exists():
            return set_file
        
        # Try to find any .set file in the directory
        set_files = list(subject_dir.glob("*.set"))
        if set_files:
            return set_files[0]
        
        raise FileNotFoundError(f"No .set file found for subject {subject_id} in {subject_dir}")

    def _read_set(self, file_path: Path):
        """Read raw .set file."""
        if not HAS_MNE:
            raise ImportError("MNE is required for reading EEGLAB .set files")
        
        raw = mne.io.read_raw_eeglab(str(file_path), preload=True, verbose=False)

        # Print channel info
        print(f"  Channels ({len(raw.ch_names)}): {raw.ch_names}")

        # Check for reference channels (M1, M2) - keep them as EEG channels
        ref_chs = [ch for ch in REFERENCE_CHANNELS if ch in raw.ch_names]
        if ref_chs:
            print(f"  Reference channels found: {ref_chs} (keeping as EEG channels)")
        
        # Always use common average reference
        raw.set_eeg_reference('average', verbose=False)

        # Auto-detect unit and convert to Volts for MNE
        max_amp = np.abs(raw._data).max()
        if max_amp > 1e-3:  # > 0.001, likely microvolts
            raw._data = raw._data / 1e6
            detected_unit = "µV"
        else:  # likely already Volts
            detected_unit = "V"
        print(f"  Detected unit: {detected_unit}, max amplitude: {max_amp:.2e}")

        return raw

    def _preprocess(self, raw):
        """Apply preprocessing to raw data."""
        # Notch filter
        if self.filter_notch > 0:
            try:
                raw.notch_filter(freqs=self.filter_notch, verbose=False)
            except Exception as e:
                print(f"  Warning: Notch filter failed: {e}")
        
        # Bandpass filter
        raw.filter(l_freq=self.filter_low, h_freq=self.filter_high, verbose=False)
        
        # Resample if needed
        if abs(raw.info['sfreq'] - self.target_sfreq) > 0.1:
            raw.resample(self.target_sfreq, verbose=False)
        
        return raw

    def build_subject(self, subject_id: str) -> str:
        """
        Build HDF5 file for a single subject.

        Args:
            subject_id: Subject identifier (string, e.g., "5x5add_Resting-prepro")

        Returns:
            Path to output HDF5 file
        """
        if not HAS_MNE:
            raise ImportError("MNE is required for building CCBDRest dataset")

        # Reset validation counters
        self.total_segments = 0
        self.valid_segments = 0
        self.rejected_segments = 0

        print(f"\n{'='*50}")
        print(f"Processing Subject {subject_id}")

        # Find .set file
        set_file = self._find_files(subject_id)
        print(f"  Found file: {set_file}")

        # Read and preprocess
        raw = self._read_set(set_file)
        raw = self._preprocess(raw)

        # Get channel names
        ch_names = raw.ch_names

        # Get data (in Volts from MNE)
        data = raw.get_data()  # Shape: (n_channels, n_samples)
        
        # Convert to µV
        data_uv = data * 1e6

        # Create subject attributes
        subject_attrs = SubjectAttrs(
            subject_id=subject_id,
            dataset_name=CCBDREST_INFO.dataset_name,
            task_type=str(CCBDREST_INFO.task_type.value),
            downstream_task_type=str(CCBDREST_INFO.downstream_task_type.value),
            rsFreq=self.target_sfreq,
            chn_name=ch_names,
            num_labels=CCBDREST_INFO.num_labels,
            category_list=CCBDREST_INFO.category_list,
            chn_pos=None,
            chn_ori=None,
            chn_type="EEG",
            montage=CCBDREST_INFO.montage,
        )

        # Create output file
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / f"sub_{subject_id}.h5"

        # Window data and write segments
        n_channels, total_samples = data_uv.shape
        sfreq = self.target_sfreq

        with HDF5Writer(str(output_path), subject_attrs) as writer:
            # Single trial for resting state
            trial_attrs = TrialAttrs(trial_id=0, session_id=0)
            trial_name = writer.add_trial(trial_attrs)

            start_sample = 0
            seg_id = 0

            while start_sample + self.window_samples <= total_samples:
                end_sample = start_sample + self.window_samples
                seg_data = data_uv[:, start_sample:end_sample]

                # Validate amplitude
                self.total_segments += 1
                if not self._validate_segment(seg_data):
                    self.rejected_segments += 1
                    start_sample += self.stride_samples
                    continue
                self.valid_segments += 1

                # Calculate time
                seg_start_time = start_sample / sfreq
                seg_end_time = seg_start_time + self.window_sec

                segment_attrs = SegmentAttrs(
                    segment_id=seg_id,
                    start_time=seg_start_time,
                    end_time=seg_end_time,
                    time_length=self.window_sec,
                    label=np.array([0]),  # Single label for resting state
                )
                writer.add_segment(trial_name, segment_attrs, seg_data)

                start_sample += self.stride_samples
                seg_id += 1

        # Report validation statistics
        self._report_validation_stats(subject_id)
        print(f"Saved {output_path} ({self.valid_segments} valid segments)")
        return str(output_path)

    def build_all(self, subject_ids: list[str] = None) -> list[str]:
        """
        Build HDF5 files for all subjects.

        Args:
            subject_ids: List of subject IDs to process (None = all)

        Returns:
            List of output file paths
        """
        if subject_ids is None:
            subject_ids = self.get_subject_ids()

        if not subject_ids:
            print("Warning: No subjects found!")
            return []

        print(f"Found {len(subject_ids)} subjects: {subject_ids[:5]}..." if len(subject_ids) > 5 else f"Found {len(subject_ids)} subjects: {subject_ids}")

        output_paths = []
        failed_subjects = []
        all_total_segments = 0
        all_valid_segments = 0
        all_rejected_segments = 0

        for subject_id in subject_ids:
            try:
                output_path = self.build_subject(subject_id)
                output_paths.append(output_path)
                all_total_segments += self.total_segments
                all_valid_segments += self.valid_segments
                all_rejected_segments += self.rejected_segments
            except Exception as e:
                print(f"Error processing subject {subject_id}: {e}")
                failed_subjects.append(subject_id)
                import traceback
                traceback.print_exc()

        # Summary report
        print("\n" + "=" * 50)
        print("PROCESSING SUMMARY")
        print("=" * 50)
        print(f"Total subjects: {len(subject_ids)}")
        print(f"Successful: {len(output_paths)}")
        print(f"Failed: {len(failed_subjects)}")
        if failed_subjects:
            print(f"Failed subject IDs: {failed_subjects}")
        print(f"\nTotal segments: {all_total_segments}")
        print(f"Valid segments: {all_valid_segments}")
        print(f"Rejected segments: {all_rejected_segments}")
        if all_total_segments > 0:
            print(f"Rejection rate: {all_rejected_segments / all_total_segments * 100:.1f}%")
        print("=" * 50)

        # Save dataset info JSON
        stats = {
            "total_subjects": len(subject_ids),
            "successful_subjects": len(output_paths),
            "failed_subjects": failed_subjects,
            "total_segments": all_total_segments,
            "valid_segments": all_valid_segments,
            "rejected_segments": all_rejected_segments,
            "rejection_rate": f"{all_rejected_segments / all_total_segments * 100:.1f}%" if all_total_segments > 0 else "0%",
        }
"""
CCBDRest Dataset Builder.

CCBDRest Dataset
- Resting state EEG data
- EEGLAB .set file format
- 1000 Hz original sampling rate
- 32 channels (10-20 montage)
"""

import sys
import json
import warnings
from pathlib import Path
from datetime import datetime
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
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from schema import SubjectAttrs, TrialAttrs, SegmentAttrs
    from hdf5_io import HDF5Writer
    from config import DatasetInfo, DatasetTaskType, DownstreamTaskType

CCBDREST_INFO = DatasetInfo(
    dataset_name="CCBDRest",
    task_type=DatasetTaskType.RESTING_STATE,
    downstream_task_type=DownstreamTaskType.CLASSIFICATION,
    num_labels=1,  # Resting state - single label
    category_list=["resting"],
    sampling_rate=1000.0,  # Original sampling rate
    montage="10_20",
    channels=["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", 
              "F7", "F8", "T7", "T8", "P7", "P8", "Fz", "Cz", "Pz", "Oz",
              "FC1", "FC2", "CP1", "CP2", "FC5", "FC6", "CP5", "CP6", 
              "PO7", "PO8", "M1", "M2"],
)

# Default amplitude threshold (µV)
DEFAULT_MAX_AMPLITUDE_UV = 600.0

# Reference channels (M1, M2) - will be kept as EEG channels for now
REFERENCE_CHANNELS = ['M1', 'M2']


class CCBDRestBuilder:
    """Builder for CCBDRest dataset."""

    def __init__(
        self,
        raw_data_dir: str,
        output_dir: str = "./hdf5",
        target_sfreq: float = 200.0,
        window_sec: float = 2.0,
        stride_sec: float = 2.0,
        filter_low: float = 0.1,
        filter_high: float = 75.0,
        filter_notch: float = 50.0,
        max_amplitude_uv: float = DEFAULT_MAX_AMPLITUDE_UV,
    ):
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir) / "CCBDRest"
        self.target_sfreq = target_sfreq
        self.orig_sfreq = 1000.0
        self.window_sec = window_sec
        self.stride_sec = stride_sec
        self.filter_low = filter_low
        self.filter_high = filter_high
        self.filter_notch = filter_notch
        self.max_amplitude_uv = max_amplitude_uv

        self.window_samples = int(window_sec * target_sfreq)
        self.stride_samples = int(stride_sec * target_sfreq)

        # Validation statistics
        self.total_segments = 0
        self.valid_segments = 0
        self.rejected_segments = 0

    def _validate_segment(self, segment_data: np.ndarray) -> bool:
        """Check if segment amplitude is within acceptable range."""
        return np.abs(segment_data).max() <= self.max_amplitude_uv

    def _report_validation_stats(self, subject_id: str):
        """Report validation statistics."""
        valid_pct = (self.valid_segments / self.total_segments * 100) if self.total_segments > 0 else 0
        print(f"Subject {subject_id} validation report:")
        print(f"  Amplitude threshold: {self.max_amplitude_uv} µV")
        print(f"  Total segments: {self.total_segments}")
        print(f"  Valid segments: {self.valid_segments} ({valid_pct:.1f}%)")
        print(f"  Rejected segments: {self.rejected_segments} ({100-valid_pct:.1f}%)")

    def _save_dataset_info(self, stats: dict):
        """Save dataset info and processing parameters to JSON."""
        info = {
            "dataset": {
                "name": CCBDREST_INFO.dataset_name,
                "description": "CCBDRest Dataset - Resting State EEG",
                "task_type": str(CCBDREST_INFO.task_type.value),
                "downstream_task": str(CCBDREST_INFO.downstream_task_type.value),
                "num_labels": CCBDREST_INFO.num_labels,
                "category_list": CCBDREST_INFO.category_list,
                "original_sampling_rate": CCBDREST_INFO.sampling_rate,
                "channels": CCBDREST_INFO.channels,
                "montage": CCBDREST_INFO.montage,
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
        with open(json_path, "w") as f:
            json.dump(info, f, indent=2)
        print(f"Saved dataset info to {json_path}")

    def get_subject_ids(self) -> list[str]:
        """Get list of subject IDs by scanning directory."""
        subject_ids = []
        if not self.raw_data_dir.exists():
            return subject_ids
        
        # Scan for sub-* directories
        # Expected structure: raw_data_dir/sub-xxxxxxx/ses-visit1/*.set
        for item in self.raw_data_dir.iterdir():
            if item.is_dir() and item.name.startswith("sub-"):
                # Extract subject ID from directory name (sub-xxxxxxx -> xxxxxxx)
                subject_id = item.name[4:]  # Remove "sub-" prefix
                # Check if this directory contains ses-* subdirectories with .set files
                ses_dirs = list(item.glob("ses-*"))
                for ses_dir in ses_dirs:
                    set_files = list(ses_dir.glob("*.set"))
                    if set_files:
                        subject_ids.append(subject_id)
                        break  # Found at least one session with .set file
        
        return sorted(subject_ids)

    def _find_files(self, subject_id: str) -> Path:
        """
        Find .set file for a subject.
        Expected structure: raw_data_dir/sub-{subject_id}/ses-visit1/*.set
        """
        subject_dir = self.raw_data_dir / f"sub-{subject_id}"
        if not subject_dir.exists():
            raise FileNotFoundError(f"Subject directory not found: {subject_dir}")
        
        # Look for ses-* subdirectories
        ses_dirs = sorted(subject_dir.glob("ses-*"))
        if not ses_dirs:
            raise FileNotFoundError(f"No session directory found for subject {subject_id} in {subject_dir}")
        
        # Try to find .set file in session directories (prefer ses-visit1)
        for ses_dir in ses_dirs:
            # Try pattern sub_{subject_id}.set first (new naming)
            set_file = ses_dir / f"sub_{subject_id}.set"
            if set_file.exists():
                return set_file
            
            # Try old pattern {subject_id}_Resting-prepro.set
            set_file = ses_dir / f"{subject_id}_Resting-prepro.set"
            if set_file.exists():
                return set_file
            
            # Try any .set file in this session directory
            set_files = list(ses_dir.glob("*.set"))
            if set_files:
                return set_files[0]
        
        raise FileNotFoundError(f"No .set file found for subject {subject_id} in {subject_dir}")

    def _read_set(self, file_path: Path):
        """Read raw .set file."""
        if not HAS_MNE:
            raise ImportError("MNE is required for reading EEGLAB .set files")
        
        raw = mne.io.read_raw_eeglab(str(file_path), preload=True, verbose=False)

        # Print channel info
        print(f"  Channels ({len(raw.ch_names)}): {raw.ch_names}")

        # Check for reference channels (M1, M2) - keep them as EEG channels
        ref_chs = [ch for ch in REFERENCE_CHANNELS if ch in raw.ch_names]
        if ref_chs:
            print(f"  Reference channels found: {ref_chs} (keeping as EEG channels)")
        
        # Always use common average reference
        raw.set_eeg_reference('average', verbose=False)

        # Auto-detect unit and convert to Volts for MNE
        max_amp = np.abs(raw._data).max()
        if max_amp > 1e-3:  # > 0.001, likely microvolts
            raw._data = raw._data / 1e6
            detected_unit = "µV"
        else:  # likely already Volts
            detected_unit = "V"
        print(f"  Detected unit: {detected_unit}, max amplitude: {max_amp:.2e}")

        return raw

    def _preprocess(self, raw):
        """Apply preprocessing to raw data."""
        # Notch filter
        if self.filter_notch > 0:
            try:
                raw.notch_filter(freqs=self.filter_notch, verbose=False)
            except Exception as e:
                print(f"  Warning: Notch filter failed: {e}")
        
        # Bandpass filter
        raw.filter(l_freq=self.filter_low, h_freq=self.filter_high, verbose=False)
        
        # Resample if needed
        if abs(raw.info['sfreq'] - self.target_sfreq) > 0.1:
            raw.resample(self.target_sfreq, verbose=False)
        
        return raw

    def build_subject(self, subject_id: str) -> str:
        """
        Build HDF5 file for a single subject.

        Args:
            subject_id: Subject identifier (string, e.g., "5x5add_Resting-prepro")

        Returns:
            Path to output HDF5 file
        """
        if not HAS_MNE:
            raise ImportError("MNE is required for building CCBDRest dataset")

        # Reset validation counters
        self.total_segments = 0
        self.valid_segments = 0
        self.rejected_segments = 0

        print(f"\n{'='*50}")
        print(f"Processing Subject {subject_id}")

        # Find .set file
        set_file = self._find_files(subject_id)
        print(f"  Found file: {set_file}")

        # Read and preprocess
        raw = self._read_set(set_file)
        raw = self._preprocess(raw)

        # Get channel names
        ch_names = raw.ch_names

        # Get data (in Volts from MNE)
        data = raw.get_data()  # Shape: (n_channels, n_samples)
        
        # Convert to µV
        data_uv = data * 1e6

        # Create subject attributes
        subject_attrs = SubjectAttrs(
            subject_id=subject_id,
            dataset_name=CCBDREST_INFO.dataset_name,
            task_type=str(CCBDREST_INFO.task_type.value),
            downstream_task_type=str(CCBDREST_INFO.downstream_task_type.value),
            rsFreq=self.target_sfreq,
            chn_name=ch_names,
            num_labels=CCBDREST_INFO.num_labels,
            category_list=CCBDREST_INFO.category_list,
            chn_pos=None,
            chn_ori=None,
            chn_type="EEG",
            montage=CCBDREST_INFO.montage,
        )

        # Create output file
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / f"sub_{subject_id}.h5"

        # Window data and write segments
        n_channels, total_samples = data_uv.shape
        sfreq = self.target_sfreq

        with HDF5Writer(str(output_path), subject_attrs) as writer:
            # Single trial for resting state
            trial_attrs = TrialAttrs(trial_id=0, session_id=0)
            trial_name = writer.add_trial(trial_attrs)

            start_sample = 0
            seg_id = 0

            while start_sample + self.window_samples <= total_samples:
                end_sample = start_sample + self.window_samples
                seg_data = data_uv[:, start_sample:end_sample]

                # Validate amplitude
                self.total_segments += 1
                if not self._validate_segment(seg_data):
                    self.rejected_segments += 1
                    start_sample += self.stride_samples
                    continue
                self.valid_segments += 1

                # Calculate time
                seg_start_time = start_sample / sfreq
                seg_end_time = seg_start_time + self.window_sec

                segment_attrs = SegmentAttrs(
                    segment_id=seg_id,
                    start_time=seg_start_time,
                    end_time=seg_end_time,
                    time_length=self.window_sec,
                    label=np.array([0]),  # Single label for resting state
                )
                writer.add_segment(trial_name, segment_attrs, seg_data)

                start_sample += self.stride_samples
                seg_id += 1

        # Report validation statistics
        self._report_validation_stats(subject_id)
        print(f"Saved {output_path} ({self.valid_segments} valid segments)")
        return str(output_path)

    def build_all(self, subject_ids: list[str] = None) -> list[str]:
        """
        Build HDF5 files for all subjects.

        Args:
            subject_ids: List of subject IDs to process (None = all)

        Returns:
            List of output file paths
        """
        if subject_ids is None:
            subject_ids = self.get_subject_ids()

        if not subject_ids:
            print("Warning: No subjects found!")
            return []

        print(f"Found {len(subject_ids)} subjects: {subject_ids[:5]}..." if len(subject_ids) > 5 else f"Found {len(subject_ids)} subjects: {subject_ids}")

        output_paths = []
        failed_subjects = []
        all_total_segments = 0
        all_valid_segments = 0
        all_rejected_segments = 0

        for subject_id in subject_ids:
            try:
                output_path = self.build_subject(subject_id)
                output_paths.append(output_path)
                all_total_segments += self.total_segments
                all_valid_segments += self.valid_segments
                all_rejected_segments += self.rejected_segments
            except Exception as e:
                print(f"Error processing subject {subject_id}: {e}")
                failed_subjects.append(subject_id)
                import traceback
                traceback.print_exc()

        # Summary report
        print("\n" + "=" * 50)
        print("PROCESSING SUMMARY")
        print("=" * 50)
        print(f"Total subjects: {len(subject_ids)}")
        print(f"Successful: {len(output_paths)}")
        print(f"Failed: {len(failed_subjects)}")
        if failed_subjects:
            print(f"Failed subject IDs: {failed_subjects}")
        print(f"\nTotal segments: {all_total_segments}")
        print(f"Valid segments: {all_valid_segments}")
        print(f"Rejected segments: {all_rejected_segments}")
        if all_total_segments > 0:
            print(f"Rejection rate: {all_rejected_segments / all_total_segments * 100:.1f}%")
        print("=" * 50)

        # Save dataset info JSON
        stats = {
            "total_subjects": len(subject_ids),
            "successful_subjects": len(output_paths),
            "failed_subjects": failed_subjects,
            "total_segments": all_total_segments,
            "valid_segments": all_valid_segments,
            "rejected_segments": all_rejected_segments,
            "rejection_rate": f"{all_rejected_segments / all_total_segments * 100:.1f}%" if all_total_segments > 0 else "0%",
        }
        self._save_dataset_info(stats)

        return output_paths


def build_ccbdrest(
    raw_data_dir: str,
    output_dir: str = "./hdf5",
    subject_ids: list[str] = None,
    **kwargs,
) -> list[str]:
    """
    Convenience function to build CCBDRest dataset.

    Args:
        raw_data_dir: Directory containing raw .set files
        output_dir: Output directory for HDF5 files
        subject_ids: List of subject IDs to process (None = all)
        **kwargs: Additional arguments for CCBDRestBuilder

    Returns:
        List of output file paths
    """
    builder = CCBDRestBuilder(raw_data_dir, output_dir, **kwargs)
    return builder.build_all(subject_ids)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build CCBDRest HDF5 dataset")
    parser.add_argument("raw_data_dir", help="Directory containing raw .set files")
    parser.add_argument(
        "--output_dir",
        default="./hdf5",
        help="Output directory",
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
        "--max_amplitude_uv",
        type=float,
        default=DEFAULT_MAX_AMPLITUDE_UV,
        help="Maximum amplitude threshold in µV",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        help="Subject IDs to process (e.g., 5x5add_Resting-prepro). Default: all subjects.",
    )

    args = parser.parse_args()

    builder = CCBDRestBuilder(
        raw_data_dir=args.raw_data_dir,
        output_dir=args.output_dir,
        target_sfreq=args.target_sfreq,
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        filter_low=args.filter_low,
        filter_high=args.filter_high,
        filter_notch=args.filter_notch,
        max_amplitude_uv=args.max_amplitude_uv,
    )

    builder.build_all(args.subjects)
