"""
Configuration classes for EEG data preprocessing.
TODO: discuss about the current setting
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class DatasetTaskType(str, Enum):
    """Task types for EEG datasets."""
    SEIZURE = "seizure"
    EMOTION = "emotion"
    MOTOR_IMAGINARY = "motor_imaginary"
    SLEEP = "sleep"
    COGNITIVE = "cognitive"
    RESTING_STATE = "resting_state"
    EOG = "eog"
    MIX= "mix"
    OTHER = "other"


class DownstreamTaskType(str, Enum):
    """Downstream task types for EEG classification."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    DETECTION = "detection"


@dataclass
class PreprocConfig:
    """Preprocessing configuration for EEG data."""
    # Filtering
    filter_low: float = 0.1
    filter_high: float = 75.0
    filter_notch: float = 50.0

    # Resampling
    target_sfreq: float = 200.0

    # Windowing
    window_sec: float = 2.0
    stride_sec: float = 2.0

    # Output
    output_dir: str = "./hdf5"

    # Processing
    num_workers: int = 4


@dataclass
class DatasetInfo:
    """Dataset-level information."""
    dataset_name: str  # e.g., "SEED_3class" (concat dataset and experiment)
    task_type: DatasetTaskType
    downstream_task_type: DownstreamTaskType = DownstreamTaskType.CLASSIFICATION
    num_labels: int = 0
    category_list: list[str] = field(default_factory=list)
    sampling_rate: float = 200.0
    montage: str = "10_20"
    channels: list[str] = field(default_factory=list)
