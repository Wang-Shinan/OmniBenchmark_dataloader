"""
SleepEDF Metadata Extractor.

Parses SC-subjects.xls and ST-subjects.xls to extract demographics.
"""

import pandas as pd
from typing import List, Dict, Any
from pathlib import Path
from .base import BaseMetadataExtractor
from ..schema import SubjectMetadata, Demographics, Gender, RecordingContext

class SleepEDFExtractor(BaseMetadataExtractor):
    def parse(self) -> List[SubjectMetadata]:
        metadata_list = []
        
        # 1. Parse SC (Sleep Cassette) subjects
        sc_file = self.dataset_path / "SC-subjects.xls"
        if sc_file.exists():
            metadata_list.extend(self._parse_sc_file(sc_file))
            
        # 2. Parse ST (Sleep Telemetry) subjects
        st_file = self.dataset_path / "ST-subjects.xls"
        if st_file.exists():
            metadata_list.extend(self._parse_st_file(st_file))
            
        return metadata_list

    def _parse_gender(self, gender_str: Any) -> Gender:
        if pd.isna(gender_str):
            return Gender.UNKNOWN
        g = str(gender_str).strip().upper()
        if g.startswith('M') or g == '1':
            return Gender.MALE
        if g.startswith('F') or g == '2':
            return Gender.FEMALE
        return Gender.UNKNOWN

    def _parse_sc_file(self, file_path: Path) -> List[SubjectMetadata]:
        """
        Parse SC-subjects.xls
        Expected columns: subject, age, sex, lights off
        """
        try:
            # Read header from line 2 (index 1) based on raw file preview
            # But usually pandas read_excel can auto-detect. 
            # The raw preview showed "subject" around line 3. Let's try header=0 first.
            df = pd.read_excel(file_path)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return []

        # Normalize column names
        df.columns = [str(c).lower().strip() for c in df.columns]
        
        results = []
        for _, row in df.iterrows():
            if pd.isna(row.get('subject')):
                continue
                
            try:
                # SC subject IDs are usually integers like 0, 1, 2...
                # But in file naming SC4001, '4' is study, '00' is subject.
                # Let's assume the excel 'subject' column matches the middle part of filename
                sub_id = int(row['subject'])
                
                # Create Demographics
                age = float(row['age']) if not pd.isna(row.get('age')) else None
                gender = self._parse_gender(row.get('sex (f=1)')) # Header might vary, adjust dynamically
                
                # Check for alternative gender headers
                if gender == Gender.UNKNOWN and 'sex' in row:
                     gender = self._parse_gender(row['sex'])
                if gender == Gender.UNKNOWN and 'm1/f2' in row:
                     gender = self._parse_gender(row['m1/f2'])

                # Extra attributes
                extra = {}
                if 'lights off' in row:
                    extra['lights_off_time'] = str(row['lights off'])
                
                demographics = Demographics(
                    age=age,
                    gender=gender,
                    group="Sleep-Cassette", # Cohort
                    extra_attributes=extra
                )
                
                meta = SubjectMetadata(
                    subject_id=sub_id, # Note: This might collide with ST subjects if IDs overlap
                    dataset_name="SleepEDF-SC",
                    demographics=demographics,
                    raw_source_record=row.to_dict()
                )
                results.append(meta)
                
            except ValueError:
                continue
                
        return results

    def _parse_st_file(self, file_path: Path) -> List[SubjectMetadata]:
        """
        Parse ST-subjects.xls
        """
        try:
            df = pd.read_excel(file_path)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return []

        df.columns = [str(c).lower().strip() for c in df.columns]
        results = []
        
        for _, row in df.iterrows():
            if pd.isna(row.get('subject')):
                continue
                
            try:
                sub_id = int(row['subject'])
                # Avoid ID collision with SC: maybe add offset? 
                # For now, let's keep original ID but distinct dataset_name
                
                age = float(row['age']) if not pd.isna(row.get('age')) else None
                
                # ST file often has 'sex (f=1)' or similar
                gender = Gender.UNKNOWN
                if 'sex' in row:
                    gender = self._parse_gender(row['sex'])
                elif 'gender' in row:
                    gender = self._parse_gender(row['gender'])
                
                extra = {}
                # Capture all other columns as extra
                for k, v in row.items():
                    if k not in ['subject', 'age', 'sex', 'gender'] and not pd.isna(v):
                        extra[k] = v

                demographics = Demographics(
                    age=age,
                    gender=gender,
                    group="Sleep-Telemetry",
                    extra_attributes=extra
                )
                
                meta = SubjectMetadata(
                    subject_id=sub_id,
                    dataset_name="SleepEDF-ST",
                    demographics=demographics,
                    raw_source_record=row.to_dict()
                )
                results.append(meta)
            except ValueError:
                continue
                
        return results
