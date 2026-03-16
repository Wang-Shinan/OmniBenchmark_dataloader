"""
EEG Data Schema for HDF5 storage format.

Hierarchical structure:
- Subject-level: Common attributes for all segments from a subject
- Trial-level: Attributes specific to each trial
- Segment-level: Attributes for each EEG segment
TODO: discuss about the current setting
"""

from dataclasses import dataclass, field
from typing import Optional, Union
import numpy as np


@dataclass
class SubjectAttrs:
    """Subject-level attributes (stored at HDF5 file root)."""
    subject_id: Union[int, str]
    dataset_name: str  # e.g., "SEED_3class" (concat dataset and experiment)
    task_type: str
    downstream_task_type: str  # downstream task type
    rsFreq: float
    chn_name: list[str]
    num_labels: int = 0  # number of classes
    category_list: list[str] = field(default_factory=list)  # label names, index = label number
    chn_pos: Optional[np.ndarray] = None  # (n_channels, 3) or None
    chn_ori: Optional[np.ndarray] = None  # (n_channels, 3) or None
    chn_type: str = "EEG"
    montage: str = "10_20"
    report: str = ""  # Clinical report text
    clinical_metadata: dict = field(default_factory=dict)  # Structured metadata from Excel


@dataclass
class TrialAttrs:
    """Trial-level attributes."""
    trial_id: int
    session_id: int
    task_name: Optional[str] = ""
    report: str = ""  # Clinical report text for this session
    clinical_metadata: dict = field(default_factory=dict)  # Structured metadata from Excel


@dataclass
class SegmentAttrs:
    """Segment-level attributes."""
    segment_id: int
    start_time: float  # start time in seconds
    end_time: float  # end time in seconds
    time_length: float
    label: np.ndarray
    task_label: Optional[str] = ""


@dataclass
class EEGSegment:
    """Complete EEG segment with inherited attributes from subject/trial."""
    data: np.ndarray  # (n_channels, n_timepoints)
    subject: SubjectAttrs
    trial: TrialAttrs
    segment: SegmentAttrs


@dataclass
class DatasetConfig:
    """Dataset-level configuration."""
    dataset_name: str
    experiment_name: str
    num_labels: int
    category_list: list[str] = field(default_factory=list)
    sampling_rate: float = 256.0
