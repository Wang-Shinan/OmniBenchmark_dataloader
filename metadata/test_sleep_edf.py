"""
Test script for SleepEDF metadata extraction.
"""

from benchmark_dataloader.metadata.extractors import SleepEDFExtractor
from benchmark_dataloader.metadata.io import MetadataHDF5Writer, MetadataHDF5Reader
import os

def test_sleep_edf():
    # Path to dataset
    dataset_path = "Z:\\Datasets\\SleepEDFxDataset"
    output_file = "Z:\\benchmark_dataloader\\metadata\\output\\sleep_edf_metadata.h5"
    
    print(f"Extracting metadata from {dataset_path}...")
    extractor = SleepEDFExtractor(dataset_path)
    metadata_list = extractor.parse()
    
    extractor.validate(metadata_list)
    
    if metadata_list:
        print(f"\nSample Metadata (Subject {metadata_list[0].subject_id}):")
        print(f"  Dataset: {metadata_list[0].dataset_name}")
        print(f"  Age: {metadata_list[0].demographics.age}")
        print(f"  Gender: {metadata_list[0].demographics.gender}")
        print(f"  Extra: {metadata_list[0].demographics.extra_attributes}")
        
        # Save to HDF5
        print(f"\nSaving to {output_file}...")
        writer = MetadataHDF5Writer(output_file)
        writer.write(metadata_list)
        
        # Verify read
        print("\nVerifying read back...")
        reader = MetadataHDF5Reader(output_file)
        read_meta = reader.read_subject(metadata_list[0].subject_id)
        
        if read_meta:
            print(f"  Read Success! Subject {read_meta.subject_id} found.")
            print(f"  Restored Extra: {read_meta.demographics.extra_attributes}")
        else:
            print("  Read Failed!")

if __name__ == "__main__":
    test_sleep_edf()
