"""
Metadata Schema for EEG Benchmark Dataloader.

Defines standardized data structures for subject demographics and recording context,
with support for flexible extension via key-value pairs.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum
import json

class Gender(Enum):
    MALE = "M"
    FEMALE = "F"
    OTHER = "O"
    UNKNOWN = "U"

class Handedness(Enum):
    RIGHT = "R"
    LEFT = "L"
    AMBIDEXTROUS = "A"
    UNKNOWN = "U"

@dataclass
class Demographics:
    """
    Standardized demographics information.
    
    Attributes:
        age: Subject age in years.
        gender: Standardized gender.
        handedness: Subject handedness.
        group: Primary grouping/label (e.g., "Patient", "Control").
        extra_attributes: Dictionary for dataset-specific attributes (e.g., {"iq": 120}).
    """
    age: Optional[float] = None
    gender: Gender = Gender.UNKNOWN
    handedness: Handedness = Handedness.UNKNOWN
    group: Optional[str] = None
    extra_attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "age": self.age,
            "gender": self.gender.value,
            "handedness": self.handedness.value,
            "group": self.group,
            "extra_attributes": json.dumps(self.extra_attributes)  # JSON serialize dict
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Demographics':
        """Load from dictionary."""
        extra = data.get("extra_attributes", "{}")
        if isinstance(extra, str):
            try:
                extra = json.loads(extra)
            except json.JSONDecodeError:
                extra = {}
                
        return cls(
            age=data.get("age"),
            gender=Gender(data.get("gender", "U")),
            handedness=Handedness(data.get("handedness", "U")),
            group=data.get("group"),
            extra_attributes=extra
        )

@dataclass
class RecordingContext:
    """
    Contextual information about the recording session.
    """
    device_name: Optional[str] = None
    location: Optional[str] = None
    date: Optional[str] = None  # ISO format preferred
    notes: Optional[str] = None
    extra_attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device_name": self.device_name,
            "location": self.location,
            "date": self.date,
            "notes": self.notes,
            "extra_attributes": json.dumps(self.extra_attributes)
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RecordingContext':
        extra = data.get("extra_attributes", "{}")
        if isinstance(extra, str):
            try:
                extra = json.loads(extra)
            except json.JSONDecodeError:
                extra = {}

        return cls(
            device_name=data.get("device_name"),
            location=data.get("location"),
            date=data.get("date"),
            notes=data.get("notes"),
            extra_attributes=extra
        )

@dataclass
class SubjectMetadata:
    """
    Complete metadata for a single subject.
    """
    subject_id: int
    dataset_name: str
    
    demographics: Demographics = field(default_factory=Demographics)
    context: RecordingContext = field(default_factory=RecordingContext)
    
    # Raw source data for traceability
    raw_source_record: Dict[str, Any] = field(default_factory=dict)
