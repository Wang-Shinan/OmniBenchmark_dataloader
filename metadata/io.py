"""
HDF5 I/O Utilities for Metadata.

Handles writing and reading SubjectMetadata objects to/from HDF5 files.
"""

import h5py
import json
import os
from pathlib import Path
from typing import List, Optional, Union
from .schema import SubjectMetadata, Demographics, RecordingContext

class MetadataHDF5Writer:
    """Writes subject metadata to HDF5 file."""
    
    def __init__(self, file_path: Union[str, Path], mode: str = "w"):
        self.file_path = Path(file_path)
        # Ensure directory exists
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.mode = mode

    def write(self, metadata_list: List[SubjectMetadata]):
        """Write a list of subject metadata to the file."""
        with h5py.File(self.file_path, self.mode) as f:
            # Create subjects group if not exists
            if "subjects" not in f:
                subjects_grp = f.create_group("subjects")
            else:
                subjects_grp = f["subjects"]

            for meta in metadata_list:
                # Group name: sub_XXXX (e.g., sub_0001)
                sub_name = f"sub_{meta.subject_id:04d}"
                
                if sub_name in subjects_grp:
                    del subjects_grp[sub_name]  # Overwrite
                
                sub_grp = subjects_grp.create_group(sub_name)
                
                # Write basic info
                sub_grp.attrs["subject_id"] = meta.subject_id
                sub_grp.attrs["dataset_name"] = meta.dataset_name
                sub_grp.attrs["raw_source_record"] = json.dumps(meta.raw_source_record)

                # Write Demographics
                demo_grp = sub_grp.create_group("demographics")
                for k, v in meta.demographics.to_dict().items():
                    if v is not None:
                        demo_grp.attrs[k] = v

                # Write Context
                ctx_grp = sub_grp.create_group("context")
                for k, v in meta.context.to_dict().items():
                    if v is not None:
                        ctx_grp.attrs[k] = v

class MetadataHDF5Reader:
    """Reads subject metadata from HDF5 file."""
    
    def __init__(self, file_path: Union[str, Path]):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.file_path}")

    def read_subject(self, subject_id: int) -> Optional[SubjectMetadata]:
        """Read metadata for a specific subject."""
        sub_name = f"sub_{subject_id:04d}"
        
        with h5py.File(self.file_path, "r") as f:
            if "subjects" not in f or sub_name not in f["subjects"]:
                return None
            
            sub_grp = f["subjects"][sub_name]
            
            # Load raw source
            raw_source = {}
            if "raw_source_record" in sub_grp.attrs:
                try:
                    raw_source = json.loads(sub_grp.attrs["raw_source_record"])
                except:
                    pass

            # Load Demographics
            demo_data = {}
            if "demographics" in sub_grp:
                demo_grp = sub_grp["demographics"]
                for k in demo_grp.attrs:
                    demo_data[k] = demo_grp.attrs[k]
            
            demographics = Demographics.from_dict(demo_data)

            # Load Context
            ctx_data = {}
            if "context" in sub_grp:
                ctx_grp = sub_grp["context"]
                for k in ctx_grp.attrs:
                    ctx_data[k] = ctx_grp.attrs[k]
            
            context = RecordingContext.from_dict(ctx_data)

            return SubjectMetadata(
                subject_id=sub_grp.attrs["subject_id"],
                dataset_name=sub_grp.attrs["dataset_name"],
                demographics=demographics,
                context=context,
                raw_source_record=raw_source
            )

    def read_all(self) -> List[SubjectMetadata]:
        """Read all subjects in the file."""
        metadata_list = []
        with h5py.File(self.file_path, "r") as f:
            if "subjects" not in f:
                return []
            
            # Sort by subject ID for consistency
            sub_names = sorted(f["subjects"].keys())
            
            # We can't use read_subject inside the loop efficiently because we want to keep file open
            # But for simplicity/safety, let's just collect IDs and call read_subject
            # Or re-implement reading logic here to avoid re-opening file
            
            subjects_grp = f["subjects"]
            for sub_name in sub_names:
                sub_grp = subjects_grp[sub_name]
                
                # ... (Same reading logic as above) ...
                # To avoid code duplication, I'll rely on read_subject logic but implemented inline
                
                raw_source = {}
                if "raw_source_record" in sub_grp.attrs:
                    try:
                        raw_source = json.loads(sub_grp.attrs["raw_source_record"])
                    except:
                        pass

                demo_data = {}
                if "demographics" in sub_grp:
                    demo_grp = sub_grp["demographics"]
                    for k in demo_grp.attrs:
                        demo_data[k] = demo_grp.attrs[k]
                demographics = Demographics.from_dict(demo_data)

                ctx_data = {}
                if "context" in sub_grp:
                    ctx_grp = sub_grp["context"]
                    for k in ctx_grp.attrs:
                        ctx_data[k] = ctx_grp.attrs[k]
                context = RecordingContext.from_dict(ctx_data)
                
                metadata_list.append(SubjectMetadata(
                    subject_id=sub_grp.attrs["subject_id"],
                    dataset_name=sub_grp.attrs["dataset_name"],
                    demographics=demographics,
                    context=context,
                    raw_source_record=raw_source
                ))
                
        return metadata_list
