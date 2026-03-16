"""
Benchmark Dataloader for EEG Foundation Model Benchmarking.

HDF5-based data infrastructure for EEG datasets.
"""

from .schema import (
    SubjectAttrs,
    TrialAttrs,
    SegmentAttrs,
    EEGSegment,
    DatasetConfig,
)
from .config import PreprocConfig, DatasetInfo, DatasetTaskType, DownstreamTaskType
from .hdf5_io import HDF5Writer, HDF5Reader
from .loader import EEGDataset, MultiDatasetLoader, load_dataset, split_dataset, SplitStrategy

__all__ = [
    # Schema
    "SubjectAttrs",
    "TrialAttrs",
    "SegmentAttrs",
    "EEGSegment",
    "DatasetConfig",
    # Config
    "PreprocConfig",
    "DatasetInfo",
    "DatasetTaskType",
    "DownstreamTaskType",
    # HDF5 I/O
    "HDF5Writer",
    "HDF5Reader",
    # Loader
    "EEGDataset",
    "MultiDatasetLoader",
    "load_dataset",
    "split_dataset",
    "SplitStrategy",
]
