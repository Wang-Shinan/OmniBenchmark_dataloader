"""
Base class for Metadata Extractors.
"""

from abc import ABC, abstractmethod
from typing import List
from pathlib import Path
from ..schema import SubjectMetadata

class BaseMetadataExtractor(ABC):
    """
    Abstract base class for dataset-specific metadata extractors.
    """

    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)

    @abstractmethod
    def parse(self) -> List[SubjectMetadata]:
        """
        Parse the raw dataset metadata files and return a list of SubjectMetadata.
        
        Returns:
            List[SubjectMetadata]: List of standardized metadata objects.
        """
        pass

    def validate(self, metadata_list: List[SubjectMetadata]):
        """
        Optional validation logic.
        """
        if not metadata_list:
            print("Warning: No metadata extracted.")
        else:
            print(f"Successfully extracted metadata for {len(metadata_list)} subjects.")
