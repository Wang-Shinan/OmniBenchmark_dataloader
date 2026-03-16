#!/usr/bin/env python3
"""
Unified script to run dataset builders.

Usage:
    python run_dataset.py <dataset_name> <raw_data_dir> [options]

Examples:
    # Process TUH dataset
    python run_dataset.py tuh /mnt/dataset2/Datasets/TUH/tuh_eeg/tuh_eeg/v2.0.1 \
        --montage_filter 01_tcp_ar \
        --output_dir /mnt/dataset2/hdf5_datasets

    # Process ADHD dataset
    python run_dataset.py adhd /mnt/dataset2/Datasets/ADHD \
        --output_dir /mnt/dataset2/hdf5_datasets

    # Process specific subjects
    python run_dataset.py tuh /mnt/dataset2/Datasets/TUH/tuh_eeg/tuh_eeg/v2.0.1 \
        --subjects aaaaaqtl aaaaaaljr \
        --output_dir /mnt/dataset2/hdf5_datasets
"""

import argparse
import sys
from pathlib import Path

# Import dataset builders
from datasets.tuh import TUHBuilder
from datasets.adhd import ADHDBuilder
from datasets.sleep_edf import SleepEDFBuilder
from datasets.tuev import TUEVBuilder
from datasets.workload import WorkloadBuilder
from datasets.cire import CIREBuilder
from datasets.bcic_2a import BCIC2ABuilder

# Note: deap.py contains BCICIV2ABuilder, not DEAPBuilder
# If DEAP dataset is needed, it should be implemented separately

# Dataset mapping
DATASET_MAP = {
    'tuh': TUHBuilder,
    'adhd': ADHDBuilder,
    'sleep_edf': SleepEDFBuilder,
    'sleepedf': SleepEDFBuilder,  # Alias
    'tuev': TUEVBuilder,
    'workload': WorkloadBuilder,
    'cire': CIREBuilder,
    'bcic_2a': BCIC2ABuilder,
    'bcic2a': BCIC2ABuilder,  # Alias
}


def main():
    parser = argparse.ArgumentParser(
        description='Build HDF5 datasets from raw data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        'dataset',
        type=str,
        choices=list(DATASET_MAP.keys()),
        help='Dataset name'
    )
    
    parser.add_argument(
        'raw_data_dir',
        type=str,
        help='Path to raw data directory'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/mnt/dataset2/hdf5_datasets',
        help='Output directory for HDF5 files (default: /mnt/dataset2/hdf5_datasets)'
    )
    
    parser.add_argument(
        '--subjects',
        type=str,
        nargs='+',
        help='Specific subject IDs to process (default: all subjects)'
    )
    
    # TUH-specific arguments
    parser.add_argument(
        '--montage_filter',
        type=str,
        default=None,
        help='Filter by montage for TUH dataset (e.g., "01_tcp_ar")'
    )
    
    parser.add_argument(
        '--max_subjects',
        type=int,
        default=None,
        help='Maximum number of subjects to process (for testing)'
    )
    
    # Common preprocessing arguments
    parser.add_argument(
        '--target_sfreq',
        type=float,
        default=200.0,
        help='Target sampling frequency after resampling (default: 200.0 Hz)'
    )
    
    parser.add_argument(
        '--window_sec',
        type=float,
        default=2.0,
        help='Window size in seconds (default: 2.0)'
    )
    
    parser.add_argument(
        '--stride_sec',
        type=float,
        default=2.0,
        help='Stride size in seconds (default: 2.0)'
    )
    
    parser.add_argument(
        '--filter_notch',
        type=float,
        default=60.0,
        help='Notch filter frequency (default: 60.0 Hz, use 50.0 for EU)'
    )
    
    parser.add_argument(
        '--filter_low',
        type=float,
        default=1.0,
        help='Low-pass filter frequency (default: 1.0 Hz)'
    )
    
    parser.add_argument(
        '--filter_high',
        type=float,
        default=40.0,
        help='High-pass filter frequency (default: 40.0 Hz)'
    )
    
    args = parser.parse_args()
    
    # Get builder class
    builder_class = DATASET_MAP[args.dataset]
    
    # Build arguments for builder
    builder_kwargs = {
        'raw_data_dir': args.raw_data_dir,
        'output_dir': args.output_dir,
        'target_sfreq': args.target_sfreq,
        'window_sec': args.window_sec,
        'stride_sec': args.stride_sec,
    }
    
    # Add TUH-specific arguments
    if args.dataset == 'tuh':
        builder_kwargs['montage_filter'] = args.montage_filter
        builder_kwargs['max_subjects'] = args.max_subjects
        builder_kwargs['filter_notch'] = args.filter_notch
        builder_kwargs['filter_low'] = args.filter_low
        builder_kwargs['filter_high'] = args.filter_high
    
    # Create builder instance
    try:
        builder = builder_class(**builder_kwargs)
    except Exception as e:
        print(f"❌ Failed to create builder: {e}")
        sys.exit(1)
    
    # Get subject IDs
    try:
        all_subject_ids = builder.get_subject_ids()
    except Exception as e:
        print(f"❌ Failed to get subject IDs: {e}")
        sys.exit(1)
    
    # Filter subjects if specified
    if args.subjects:
        subject_ids = [sid for sid in args.subjects if sid in all_subject_ids]
        if not subject_ids:
            print(f"❌ No valid subjects found in {args.subjects}")
            print(f"   Available subjects: {all_subject_ids[:10]}...")
            sys.exit(1)
    else:
        subject_ids = all_subject_ids
    
    print(f"\n{'='*60}")
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Raw data dir: {args.raw_data_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Total subjects: {len(all_subject_ids)}")
    print(f"Processing: {len(subject_ids)} subjects")
    if args.dataset == 'tuh':
        if args.montage_filter:
            print(f"Montage filter: {args.montage_filter}")
        if args.max_subjects:
            print(f"Max subjects: {args.max_subjects}")
    print(f"{'='*60}\n")
    
    # Process subjects
    success_count = 0
    skip_count = 0
    error_count = 0
    
    for i, subject_id in enumerate(subject_ids, 1):
        print(f"[{i}/{len(subject_ids)}] Processing subject: {subject_id}")
        
        try:
            # Check if output file already exists (for skip counting)
            # TUH uses "TUH" as output subdirectory, others use dataset name
            if args.dataset == 'tuh':
                output_file = Path(args.output_dir) / "TUH" / f"sub_{subject_id}.h5"
            else:
                output_file = Path(args.output_dir) / builder_class.__name__.replace('Builder', '') / f"sub_{subject_id}.h5"
            was_skipped = output_file.exists()
            
            output_path = builder.build_subject(subject_id)
            
            if output_path:
                if was_skipped:
                    skip_count += 1
                else:
                    success_count += 1
            else:
                error_count += 1
                print(f"  ⚠️  Warning: build_subject returned empty path")
        except KeyboardInterrupt:
            print(f"\n\n⚠️  Interrupted by user")
            print(f"   Processed: {success_count} subjects")
            print(f"   Skipped: {skip_count} subjects")
            print(f"   Errors: {error_count} subjects")
            sys.exit(1)
        except Exception as e:
            error_count += 1
            print(f"  ❌ Error processing subject {subject_id}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"  ✓ Success: {success_count}")
    print(f"  ⏭️  Skipped: {skip_count}")
    print(f"  ❌ Errors: {error_count}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()

