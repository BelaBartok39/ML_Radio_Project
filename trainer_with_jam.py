import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from typing import NamedTuple
from torch.optim import Adam
from sklearn.metrics import accuracy_score

# --- Dataset Class ---
class JammingDataset(Dataset):
    def __init__(self, h5_path, split='train'):
        self.h5_path = h5_path
        self.split = split
        self.file = h5py.File(self.h5_path, 'r')
        self.signals = self.file[split]['signals']
        self.mod_labels = self.file[split]['modulation']
        self.jam_labels = self.file[split]['jammed']
        self.jam_type_labels = self.file[split]['jamming_type']
        self.length = len(self.signals)

        # Label dictionaries
        self.mod_classes = sorted(set([l.decode() for l in self.mod_labels[:]]))
        self.mod_to_idx = {c: i for i, c in enumerate(self.mod_classes)}

        self.jam_type_classes = sorted(set([l.decode() for l in self.jam_type_labels[:]]))
        self.jam_type_to_idx = {c: i for i, c in enumerate(self.jam_type_classes)}

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        signal = self.signals[idx]
        x = torch.tensor(np.stack([signal.real, signal.imag], axis=0), dtype=torch.float32)

        mod = self.mod_to_idx[self.mod_labels[idx].decode()]
        jammed = int(self.jam_labels[idx])
        jam_type = self.jam_type_to_idx[self.jam_type_labels[idx].decode()] if jammed else -1

        return x, mod, jammed, jam_type

    def close(self):
        self.file.close()

class MultiTaskOutput(NamedTuple):
    mod: torch.Tensor
    jam: torch.Tensor
    jam_type: torch.Tensor

# --- Model ---
class MultiTaskCNN(nn.Module):
    def __init__(self, num_mod_classes, num_jam_types):
        super().__init__()
        self.conv1 = nn.Conv1d(2, 32, 7, padding=3)
        self.conv2 = nn.Conv1d(32, 64, 5, padding=2)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128 * 128, 256)

        self.classifier_mod = nn.Linear(256, num_mod_classes)
        self.classifier_jam = nn.Linear(256, 2)  # binary
        self.classifier_jam_type = nn.Linear(256, num_jam_types)

    def features(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        return x

    def forward(self, x):
        feat = self.features(x)
        return MultiTaskOutput(
            mod=self.classifier_mod(feat),
            jam=self.classifier_jam(feat),
            jam_type=self.classifier_jam_type(feat),
        )

# --- Training function ---
def train(h5_path, epochs=20, batch_size=64, lr=1e-3, model_path='multitask_cnn.pt'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_dataset = JammingDataset(h5_path, 'train')
    val_dataset = JammingDataset(h5_path, 'val')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = MultiTaskCNN(len(train_dataset.mod_classes), len(train_dataset.jam_type_classes)).to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0
        for x, mod, jammed, jam_type in train_loader:
            x, mod, jammed, jam_type = x.to(device), mod.to(device), jammed.to(device), jam_type.to(device)
            out = model(x)

            # Calculate losses
            loss_mod = loss_fn(out.mod, mod)
            loss_jam = loss_fn(out.jam, jammed)

            # Calculate jamming type loss only for jammed samples
            jammed_mask = (jammed == 1)
            if jammed_mask.any():
                loss_jam_type = loss_fn(out.jam_type[jammed_mask], jam_type[jammed_mask])
            else:
                # zero tensor on correct device and dtype
                loss_jam_type = torch.tensor(0.0, device=device, dtype=torch.float32)

            loss = loss_mod + loss_jam + loss_jam_type

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}")

        # Optional: validation
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for x, mod, jammed, jam_type in val_loader:
                x, mod, jammed, jam_type = x.to(device), mod.to(device), jammed.to(device), jam_type.to(device)
                out = model(x)
                v_loss_mod = loss_fn(out.mod, mod)
                v_loss_jam = loss_fn(out.jam, jammed)
                v_jammed_mask = (jammed == 1)
                if v_jammed_mask.any():
                    v_loss_jam_type = loss_fn(out.jam_type[v_jammed_mask], jam_type[v_jammed_mask])
                else:
                    v_loss_jam_type = torch.tensor(0.0, device=device, dtype=torch.float32)
                val_loss += (v_loss_mod + v_loss_jam + v_loss_jam_type).item()
            val_loss /= len(val_loader)
        print(f"Epoch {epoch}/{epochs} - Validation Loss: {val_loss:.4f}")

    # Save TorchScript model (use strict=False because of dict outputs)
    model.eval()
    example_input = torch.randn(1, 2, 1024).to(device)
    traced = torch.jit.trace(model, example_input, strict=False)
    traced.save(model_path)
    print(f"Model saved to {model_path}")

    train_dataset.close()
    val_dataset.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train multi-task modulation + jamming classifier")
    parser.add_argument('--data', type=str, default='gnuradio_jamming_dataset.h5', help='HDF5 dataset path')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--model_path', type=str, default='multitask_cnn.pt')
    args = parser.parse_args()

    train(args.data, args.epochs, args.batch_size, args.lr, args.model_path)
