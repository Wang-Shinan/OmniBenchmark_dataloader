"""
Utility functions and classes for EEG data processing.
TODO: discuss about the current setting
"""

import random
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def set_seed(seed: int, deterministic: bool = False):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)

    if HAS_TORCH:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


class ElectrodeSet:
    """
    A class to represent a set of EEG electrodes based on the 10-10 system.
    https://www.fieldtriptoolbox.org/assets/img/template/layout/eeg1010.lay.png
    """
    Layout = '10-10'
    Count = 90
    Electrodes = [
                                   'FP1', 'FPZ', 'FP2',
       'AF9', 'AF7', 'AF5', 'AF3', 'AF1', 'AFZ', 'AF2', 'AF4', 'AF6', 'AF8', 'AF10',
      'T1', 'F9', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'F10', 'T2',
       'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10',
      'A1', 'T9', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'T10', 'A2',
       'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10',
            'P9', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'P10',
       'PO9', 'PO7', 'PO5', 'PO3', 'PO1', 'POZ', 'PO2', 'PO4', 'PO6', 'PO8', 'PO10',
                                    'O1', 'OZ', 'O2',
                                    'I1', 'IZ', 'I2',
    ]

    # Standard 10-20 subset (21 channels)
    Standard_10_20 = [
        'FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8',
        'T7', 'C3', 'CZ', 'C4', 'T8',
        'P7', 'P3', 'PZ', 'P4', 'P8',
        'O1', 'O2',
        'A1', 'A2',
    ]

    def __init__(self):
        self.electrode_dict = {electrode: i for i, electrode in enumerate(self.Electrodes)}
        self.index_dict = {i: electrode for i, electrode in enumerate(self.Electrodes)}

    def __len__(self):
        return self.Count

    def get_electrodes_index(self, electrodes: list[str]) -> np.ndarray:
        """Get indices for a list of electrode names."""
        return np.array([self.electrode_dict[electrode] for electrode in electrodes], dtype=np.int32)

    def get_electrodes_name(self, electrodes: list[int]) -> list[str]:
        """Get electrode names for a list of indices."""
        return [self.index_dict[electrode] for electrode in electrodes]

    def is_valid_electrode(self, electrode: str) -> bool:
        """Check if an electrode name is valid."""
        return electrode.upper() in self.electrode_dict

    def standardize_name(self, name: str) -> str:
        """Standardize electrode name to uppercase."""
        name = name.upper()
        # Common aliases
        aliases = {
            'T3': 'T7',
            'T4': 'T8',
            'T5': 'P7',
            'T6': 'P8',
        }
        return aliases.get(name, name)


def get_mne_montage_positions(montage_name: str = "standard_1020") -> dict:
    """
    Get 3D electrode positions from MNE standard montages.

    Args:
        montage_name: Name of the montage (e.g., 'standard_1020', 'standard_1010')

    Returns:
        Dictionary mapping electrode names to (x, y, z) positions
    """
    try:
        import mne
        montage = mne.channels.make_standard_montage(montage_name)
        positions = montage.get_positions()
        ch_pos = positions['ch_pos']
        return {name.upper(): pos for name, pos in ch_pos.items()}
    except ImportError:
        return {}
