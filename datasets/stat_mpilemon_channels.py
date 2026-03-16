"""
Statistics script for MPI-LEMON dataset channel analysis.

This script analyzes:
1. Standard channel (10-20 system) missing patterns
2. Non-standard channel distribution
"""

from pathlib import Path
from collections import defaultdict, Counter
import warnings
import sys

try:
    import mne
    HAS_MNE = True
except ImportError:
    HAS_MNE = False
    print("Warning: MNE not available, cannot read EEG files")

try:
    from ..utils import ElectrodeSet
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from utils import ElectrodeSet

# Standard 10-20 System Channels (21 channels)
STANDARD_CHANNELS = ElectrodeSet.Standard_10_20

def clean_channel_name(name: str) -> str:
    """Clean and normalize channel name."""
    name = name.upper().strip()
    # Remove common prefixes/suffixes
    name = name.replace("EEG ", "").replace("-REF", "").replace("REF", "")
    return name

def standardize_channel_name(name: str, electrode_set: ElectrodeSet) -> str:
    """Standardize channel name using ElectrodeSet."""
    clean = clean_channel_name(name)
    return electrode_set.standardize_name(clean)

def get_channels_from_file(file_path: Path, electrode_set: ElectrodeSet) -> tuple[list[str], bool]:
    """
    Read channels from a BrainVision or EEGLAB file.
    
    Returns:
        (channel_names, success): List of channel names and success flag
    """
    if not HAS_MNE:
        return [], False
    
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            
            if file_path.suffix.lower() == '.vhdr':
                raw = mne.io.read_raw_brainvision(str(file_path), preload=False, verbose=False)
            elif file_path.suffix.lower() == '.set':
                raw = mne.io.read_raw_eeglab(str(file_path), preload=False, verbose=False)
            else:
                return [], False
            
            # Get only EEG channels
            eeg_picks = mne.pick_types(raw.info, eeg=True, meg=False, stim=False)
            ch_names = [raw.ch_names[i] for i in eeg_picks]
            
            return ch_names, True
            
    except Exception as e:
        return [], False

def analyze_subject_channels(
    raw_root: Path,
    subject_id: str,
    electrode_set: ElectrodeSet
) -> dict:
    """
    Analyze channels for a single subject.
    
    Returns:
        dict with keys: 'channels', 'standard_missing', 'non_standard', 'file_type'
    """
    sub_dir = raw_root / subject_id / "RSEEG"
    
    # Try BrainVision files first
    vhdr_files = list(sub_dir.glob("*.vhdr")) if sub_dir.exists() else []
    
    if vhdr_files:
        ch_names, success = get_channels_from_file(vhdr_files[0], electrode_set)
        if success:
            return {
                'channels': ch_names,
                'file_type': 'vhdr',
                'file_path': str(vhdr_files[0])
            }
    
    # Fallback to preprocessed EEGLAB files
    preproc_root = raw_root.parent / "EEG_Preprocessed_BIDS_ID" / "EEG_Preprocessed" / subject_id
    if preproc_root.exists():
        set_files = list(preproc_root.glob("*_EC.set")) + list(preproc_root.glob("*_EO.set"))
        if set_files:
            ch_names, success = get_channels_from_file(set_files[0], electrode_set)
            if success:
                return {
                    'channels': ch_names,
                    'file_type': 'set',
                    'file_path': str(set_files[0])
                }
    
    return {
        'channels': [],
        'file_type': 'none',
        'file_path': None
    }

def main(raw_root: str):
    """
    Main function to analyze MPI-LEMON channel statistics.
    
    Args:
        raw_root: Path to EEG_Raw_BIDS_ID directory
    """
    raw_root = Path(raw_root)
    electrode_set = ElectrodeSet()
    
    # Find all subjects
    subject_dirs = sorted(raw_root.glob("sub-*"))
    print(f"Found {len(subject_dirs)} subjects")
    
    # Statistics
    total_subjects = 0
    successful_reads = 0
    failed_reads = []
    
    # Standard channel missing statistics
    missing_channels_count = Counter()  # Count how many subjects miss each standard channel
    subjects_missing_channels = defaultdict(list)  # Which subjects miss which channels
    
    # Non-standard channel statistics
    non_standard_channels_count = Counter()  # Count occurrences of non-standard channels
    channel_count_distribution = Counter()  # Distribution of total channel counts
    
    # File type statistics
    file_type_count = Counter()
    
    print("\nProcessing subjects...")
    for i, sub_dir in enumerate(subject_dirs):
        subject_id = sub_dir.name
        total_subjects += 1
        
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(subject_dirs)} subjects...")
        
        result = analyze_subject_channels(raw_root, subject_id, electrode_set)
        
        if not result['channels']:
            failed_reads.append(subject_id)
            continue
        
        successful_reads += 1
        file_type_count[result['file_type']] += 1
        
        ch_names = result['channels']
        channel_count_distribution[len(ch_names)] += 1
        
        # Clean and standardize channel names
        cleaned_channels = [clean_channel_name(ch) for ch in ch_names]
        standardized_channels = [standardize_channel_name(ch, electrode_set) for ch in cleaned_channels]
        
        # Find missing standard channels
        present_standard = set()
        missing_standard = []
        
        for std_ch in STANDARD_CHANNELS:
            # Check if standard channel is present (direct match or alias)
            found = False
            for cleaned, std in zip(cleaned_channels, standardized_channels):
                if std == std_ch or cleaned == std_ch:
                    present_standard.add(std_ch)
                    found = True
                    break
            
            if not found:
                missing_standard.append(std_ch)
                missing_channels_count[std_ch] += 1
                subjects_missing_channels[std_ch].append(subject_id)
        
        # Find non-standard channels
        for cleaned, std in zip(cleaned_channels, standardized_channels):
            if std not in STANDARD_CHANNELS:
                non_standard_channels_count[cleaned] += 1
    
    # Print statistics
    print("\n" + "=" * 70)
    print("MPI-LEMON Channel Statistics")
    print("=" * 70)
    
    print(f"\nOverall Statistics:")
    print(f"  Total subjects: {total_subjects}")
    print(f"  Successfully read: {successful_reads}")
    print(f"  Failed to read: {len(failed_reads)}")
    if failed_reads:
        print(f"  Failed subjects (first 10): {failed_reads[:10]}")
    
    print(f"\nFile Type Distribution:")
    for file_type, count in file_type_count.most_common():
        print(f"  {file_type}: {count} subjects ({count/successful_reads*100:.1f}%)")
    
    print(f"\nChannel Count Distribution:")
    for count, num_subjects in sorted(channel_count_distribution.items()):
        print(f"  {count} channels: {num_subjects} subjects ({num_subjects/successful_reads*100:.1f}%)")
    
    print(f"\nStandard Channel (10-20) Missing Statistics:")
    print(f"  Total standard channels: {len(STANDARD_CHANNELS)}")
    print(f"  Channels missing in at least one subject:")
    if missing_channels_count:
        for ch, count in missing_channels_count.most_common():
            percentage = count / successful_reads * 100
            print(f"    {ch}: {count} subjects ({percentage:.1f}%)")
            if count <= 5:  # Show subjects for rarely missing channels
                print(f"      Subjects: {subjects_missing_channels[ch]}")
    else:
        print("    None - all standard channels present in all subjects")
    
    # Channels present in all subjects
    always_present = [ch for ch in STANDARD_CHANNELS if ch not in missing_channels_count]
    print(f"\n  Channels present in ALL subjects ({len(always_present)}):")
    print(f"    {', '.join(always_present)}")
    
    print(f"\nNon-Standard Channel Distribution:")
    if non_standard_channels_count:
        print(f"  Top 20 most common non-standard channels:")
        for ch, count in non_standard_channels_count.most_common(20):
            percentage = count / successful_reads * 100
            print(f"    {ch}: {count} subjects ({percentage:.1f}%)")
    else:
        print("  No non-standard channels found")
    
    print("\n" + "=" * 70)
    
    # Summary by channel count
    print("\nSummary by Channel Count:")
    for count in sorted(channel_count_distribution.keys()):
        num_subjects = channel_count_distribution[count]
        print(f"\n  {count} channels ({num_subjects} subjects):")
        
        # Sample a few subjects with this channel count
        sample_subjects = []
        for sub_dir in subject_dirs:
            subject_id = sub_dir.name
            result = analyze_subject_channels(raw_root, subject_id, electrode_set)
            if len(result['channels']) == count:
                sample_subjects.append(subject_id)
                if len(sample_subjects) >= 3:
                    break
        
        if sample_subjects:
            for sub_id in sample_subjects:
                result = analyze_subject_channels(raw_root, sub_id, electrode_set)
                ch_names = result['channels']
                cleaned = [clean_channel_name(ch) for ch in ch_names]
                print(f"    Example ({sub_id}): {', '.join(cleaned[:10])}{'...' if len(cleaned) > 10 else ''}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analyze MPI-LEMON dataset channel statistics"
    )
    parser.add_argument(
        "raw_root",
        help="Path to EEG_Raw_BIDS_ID directory, e.g., "
        "/mnt/dataset2/Datasets/MPI-LEMON/EEG_MPILMBB_LEMON/EEG_Raw_BIDS_ID"
    )
    
    args = parser.parse_args()
    main(args.raw_root)
