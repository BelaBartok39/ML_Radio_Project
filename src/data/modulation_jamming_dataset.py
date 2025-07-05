import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

class ModulationJammingDataset(Dataset):
    """
    HDF5-based dataset for multi-task modulation classification and jamming detection.
    """
    def __init__(self, h5_path, split='train'):
        self.file = h5py.File(h5_path, 'r')
        # Build label maps from all splits to cover all classes
        all_mods = []
        all_jam_types = []
        for grp_name in self.file.keys():
            grp = self.file[grp_name]
            if 'modulation' in grp:
                all_mods.extend(grp['modulation'][:])
            if 'jamming_type' in grp:
                all_jam_types.extend(grp['jamming_type'][:])
        # Unique sorted classes
        self.mod_classes = sorted({m.decode() for m in all_mods})
        self.mod_to_idx = {c: i for i, c in enumerate(self.mod_classes)}
        self.jam_type_classes = sorted({j.decode() for j in all_jam_types})
        self.jam_type_to_idx = {c: i for i, c in enumerate(self.jam_type_classes)}
        # Now load the specific split group
        grp = self.file[split]
        self.signals = grp['signals']
        self.mod_labels = grp['modulation']
        self.jam_labels = grp['jammed']
        self.jam_type_labels = grp['jamming_type']

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = self.signals[idx]
        x = np.stack([signal.real, signal.imag], axis=0)
        x = torch.tensor(x, dtype=torch.float32)

        mod = self.mod_to_idx[self.mod_labels[idx].decode()]
        jammed = int(self.jam_labels[idx])
        jam_type = self.jam_type_to_idx[self.jam_type_labels[idx].decode()] if jammed else -1
        return x, mod, jammed, jam_type

    def close(self):
        self.file.close()
