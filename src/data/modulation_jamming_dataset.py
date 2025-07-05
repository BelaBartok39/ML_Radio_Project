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
        grp = self.file[split]
        self.signals = grp['signals']
        self.mod_labels = grp['modulation']
        self.jam_labels = grp['jammed']
        self.jam_type_labels = grp['jamming_type']

        # Build label maps
        self.mod_classes = sorted({l.decode() for l in self.mod_labels[:]})
        self.mod_to_idx = {c: i for i, c in enumerate(self.mod_classes)}
        self.jam_type_classes = sorted({l.decode() for l in self.jam_type_labels[:]})
        self.jam_type_to_idx = {c: i for i, c in enumerate(self.jam_type_classes)}

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
