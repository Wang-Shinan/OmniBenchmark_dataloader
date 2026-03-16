"""
HDF5 I/O utilities for EEG data.

Provides HDF5Writer and HDF5Reader classes for storing and loading
EEG segments in the hierarchical format:
- Subject-level attributes at file root
- Trial groups with trial-level attributes
- Segment groups with EEG data and segment-level attributes
TODO: discuss about the current setting
"""

from pathlib import Path
from typing import Iterator, Optional
import h5py
import numpy as np
import json

try:
    from .schema import SubjectAttrs, TrialAttrs, SegmentAttrs, EEGSegment
except ImportError:
    from schema import SubjectAttrs, TrialAttrs, SegmentAttrs, EEGSegment


class HDF5Writer:
    """Write EEG segments to HDF5 format."""

    def __init__(
        self,
        filepath: str,
        subject_attrs: SubjectAttrs,
    ):
        """
        Initialize HDF5 writer.

        Args:
            filepath: Path to output HDF5 file
            subject_attrs: Subject-level attributes to store at file root
        """
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.file = h5py.File(self.filepath, "w")
        self._write_subject_attrs(subject_attrs)
        self._current_trial: Optional[str] = None

    def _write_subject_attrs(self, attrs: SubjectAttrs) -> None:
        """Write subject-level attributes to file root."""
        self.file.attrs["subject_id"] = attrs.subject_id
        self.file.attrs["dataset_name"] = attrs.dataset_name
        self.file.attrs["task_type"] = attrs.task_type
        self.file.attrs["downstream_task_type"] = attrs.downstream_task_type
        self.file.attrs["rsFreq"] = attrs.rsFreq
        self.file.attrs["chn_name"] = attrs.chn_name
        self.file.attrs["num_labels"] = attrs.num_labels
        # Convert category_list to bytes array for HDF5 compatibility
        if attrs.category_list:
            self.file.attrs["category_list"] = [cat.encode('utf-8') for cat in attrs.category_list]
        else:
            self.file.attrs["category_list"] = []
        self.file.attrs["chn_pos"] = attrs.chn_pos if attrs.chn_pos is not None else "None"
        self.file.attrs["chn_ori"] = attrs.chn_ori if attrs.chn_ori is not None else "None"
        self.file.attrs["chn_type"] = attrs.chn_type
        self.file.attrs["montage"] = attrs.montage

    def add_trial(self, trial_attrs: TrialAttrs) -> str:
        """
        Create a new trial group.

        Args:
            trial_attrs: Trial-level attributes

        Returns:
            Trial group name
        """
        trial_name = f"trial{trial_attrs.trial_id}"
        trial_grp = self.file.create_group(trial_name)
        trial_grp.attrs["trial_id"] = trial_attrs.trial_id
        trial_grp.attrs["session_id"] = trial_attrs.session_id
        trial_grp.attrs["task_name"] = trial_attrs.task_name or ""
        
        # Add report and metadata if present
        if hasattr(trial_attrs, "report"):
            trial_grp.attrs["report"] = trial_attrs.report
        
        if hasattr(trial_attrs, "clinical_metadata"):
            # Serialize dict to JSON string
            meta_json = json.dumps(trial_attrs.clinical_metadata)
            trial_grp.attrs["clinical_metadata"] = meta_json
            
        self._current_trial = trial_name
        return trial_name

    def add_segment(
        self,
        trial_name: str,
        segment_attrs: SegmentAttrs,
        eeg_data: np.ndarray,
    ) -> None:
        """
        Add an EEG segment to a trial.

        Args:
            trial_name: Name of the trial group
            segment_attrs: Segment-level attributes
            eeg_data: EEG data array (n_channels, n_timepoints)
        """
        trial_grp = self.file[trial_name]
        segment_name = f"segment{segment_attrs.segment_id}"
        segment_grp = trial_grp.create_group(segment_name)

        dset = segment_grp.create_dataset("eeg", data=eeg_data.astype(np.float32))
        dset.attrs["segment_id"] = segment_attrs.segment_id
        dset.attrs["start_time"] = segment_attrs.start_time
        dset.attrs["end_time"] = segment_attrs.end_time
        dset.attrs["time_length"] = segment_attrs.time_length
        dset.attrs["label"] = segment_attrs.label
        dset.attrs["task_label"] = segment_attrs.task_label or ""

    def close(self) -> None:
        """Close the HDF5 file."""
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class HDF5Reader:
    """Read EEG segments from HDF5 format."""

    def __init__(self, filepath: str):
        """
        Initialize HDF5 reader.

        Args:
            filepath: Path to HDF5 file
        """
        self.filepath = Path(filepath)
        self.file = h5py.File(self.filepath, "r")
        self._subject_attrs = self._read_subject_attrs()

    def _read_subject_attrs(self) -> SubjectAttrs:
        """Read subject-level attributes from file root."""
        chn_pos = self.file.attrs["chn_pos"]
        chn_ori = self.file.attrs["chn_ori"]

        # Handle subject_id as either int or str
        raw_subject_id = self.file.attrs["subject_id"]
        if isinstance(raw_subject_id, bytes):
            subject_id = raw_subject_id.decode("utf-8")
        elif isinstance(raw_subject_id, str):
            subject_id = raw_subject_id
        else:
            subject_id = int(raw_subject_id)

        # Read num_labels and category_list if they exist
        num_labels = int(self.file.attrs.get("num_labels", 0))
        category_list_raw = self.file.attrs.get("category_list", [])
        # Handle category_list: may be bytes array or string array
        if isinstance(category_list_raw, np.ndarray):
            if category_list_raw.dtype.type == np.bytes_:
                category_list = [item.decode('utf-8') for item in category_list_raw]
            else:
                category_list = [str(item) for item in category_list_raw]
        elif isinstance(category_list_raw, (list, tuple)):
            category_list = [item.decode('utf-8') if isinstance(item, bytes) else str(item) for item in category_list_raw]
        else:
            category_list = []

        return SubjectAttrs(
            subject_id=subject_id,
            dataset_name=str(self.file.attrs["dataset_name"]),
            task_type=str(self.file.attrs["task_type"]),
            downstream_task_type=str(self.file.attrs.get("downstream_task_type", "")),
            rsFreq=float(self.file.attrs["rsFreq"]),
            chn_name=list(self.file.attrs["chn_name"]),
            num_labels=num_labels,
            category_list=category_list,
            chn_pos=None if isinstance(chn_pos, str) and chn_pos == "None" else np.array(chn_pos),
            chn_ori=None if isinstance(chn_ori, str) and chn_ori == "None" else np.array(chn_ori),
            chn_type=str(self.file.attrs["chn_type"]),
            montage=str(self.file.attrs["montage"]),
        )

    @property
    def subject_attrs(self) -> SubjectAttrs:
        """Get subject-level attributes."""
        return self._subject_attrs

    def get_trial_names(self) -> list[str]:
        """Get list of trial group names."""
        return [k for k in self.file.keys() if k.startswith("trial")]

    def get_trial_attrs(self, trial_name: str) -> TrialAttrs:
        """Get trial-level attributes."""
        trial_grp = self.file[trial_name]
        raw_task_name = trial_grp.attrs.get("task_name", "")
        if raw_task_name is None:
            task_name = ""
        elif isinstance(raw_task_name, bytes):
            task_name = raw_task_name.decode("utf-8")
        else:
            task_name = str(raw_task_name)
            
        # Read report
        report = trial_grp.attrs.get("report", "")
        if isinstance(report, bytes):
            report = report.decode("utf-8")
            
        # Read clinical metadata
        clinical_metadata = {}
        if "clinical_metadata" in trial_grp.attrs:
            try:
                meta_json = trial_grp.attrs["clinical_metadata"]
                if isinstance(meta_json, bytes):
                    meta_json = meta_json.decode("utf-8")
                clinical_metadata = json.loads(meta_json)
            except Exception as e:
                print(f"Warning: Failed to parse clinical_metadata: {e}")
                
        return TrialAttrs(
            trial_id=int(trial_grp.attrs["trial_id"]),
            session_id=int(trial_grp.attrs["session_id"]),
            task_name=task_name,
            report=report,
            clinical_metadata=clinical_metadata
        )

    def get_segment_names(self, trial_name: str) -> list[str]:
        """Get list of segment group names in a trial."""
        trial_grp = self.file[trial_name]
        return [k for k in trial_grp.keys() if k.startswith("segment")]

    def get_segment(self, trial_name: str, segment_name: str) -> EEGSegment:
        """
        Get a single EEG segment.

        Args:
            trial_name: Name of the trial group
            segment_name: Name of the segment group

        Returns:
            EEGSegment with all attributes
        """
        trial_grp = self.file[trial_name]
        segment_grp = trial_grp[segment_name]
        dset = segment_grp["eeg"]

        trial_attrs = self.get_trial_attrs(trial_name)
        raw_task_label = dset.attrs.get("task_label", "")
        if raw_task_label is None:
            task_label = ""
        elif isinstance(raw_task_label, bytes):
            task_label = raw_task_label.decode("utf-8")
        else:
            task_label = str(raw_task_label)
        segment_attrs = SegmentAttrs(
            segment_id=int(dset.attrs["segment_id"]),
            start_time=float(dset.attrs.get("start_time", 0.0)),
            end_time=float(dset.attrs.get("end_time", 0.0)),
            time_length=float(dset.attrs["time_length"]),
            label=np.array(dset.attrs["label"]),
            task_label=task_label,
        )

        return EEGSegment(
            data=np.array(dset),
            subject=self._subject_attrs,
            trial=trial_attrs,
            segment=segment_attrs,
        )

    def iter_segments(self) -> Iterator[EEGSegment]:
        """Iterate over all segments in the file."""
        for trial_name in self.get_trial_names():
            for segment_name in self.get_segment_names(trial_name):
                yield self.get_segment(trial_name, segment_name)

    def __len__(self) -> int:
        """Get total number of segments."""
        count = 0
        for trial_name in self.get_trial_names():
            count += len(self.get_segment_names(trial_name))
        return count

    def close(self) -> None:
        """Close the HDF5 file."""
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
