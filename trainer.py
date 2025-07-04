import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import accuracy_score

# --- Dataset Class ---
class JammingDataset(Dataset):
    def __init__(self, h5_path, split='train'):
        self.h5_path = h5_path
        self.split = split
        self.file = h5py.File(self.h5_path, 'r')
        self.signals = self.file[split]['signals']
        self.labels = self.file[split]['modulation']
        self.length = len(self.signals)

        self.classes = sorted(set([l.decode() for l in self.labels[:]]))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        signal = self.signals[idx]
        x = torch.tensor(np.stack([signal.real, signal.imag], axis=0), dtype=torch.float32)
        label_str = self.labels[idx].decode()
        label = self.class_to_idx[label_str]
        return x, label

    def close(self):
        self.file.close()

# --- Model ---
class ModulationCNN(nn.Module):
    def __init__(self, num_classes):
        super(ModulationCNN, self).__init__()
        self.conv1 = nn.Conv1d(2, 32, 7, padding=3)
        self.conv2 = nn.Conv1d(32, 64, 5, padding=2)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128 * 128, 256)  # 1024 down to 128 after 3 poolings
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # (B,32,512)
        x = self.pool(F.relu(self.conv2(x)))  # (B,64,256)
        x = self.pool(F.relu(self.conv3(x)))  # (B,128,128)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# --- Training function ---
def train(h5_path, epochs=20, batch_size=64, lr=1e-3, model_path='modulation_cnn.pt', classes_path='classes.txt'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_dataset = JammingDataset(h5_path, 'train')
    val_dataset = JammingDataset(h5_path, 'val')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = ModulationCNN(num_classes=len(train_dataset.classes)).to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs+1):
        model.train()
        losses = []
        preds = []
        labels = []

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            preds.extend(outputs.argmax(dim=1).cpu().numpy())
            labels.extend(y.cpu().numpy())

        train_acc = accuracy_score(labels, preds)
        print(f"Epoch {epoch}/{epochs} - Train Loss: {np.mean(losses):.4f} - Train Acc: {train_acc:.4f}")

        # Validation
        model.eval()
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                val_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                val_labels.extend(y.cpu().numpy())
        val_acc = accuracy_score(val_labels, val_preds)
        print(f"Epoch {epoch}/{epochs} - Val Acc: {val_acc:.4f}")

    # Save TorchScript model
    model.eval()
    example_input = torch.randn(1, 2, 1024).to(device)
    traced = torch.jit.trace(model, example_input)
    traced.save(model_path)
    print(f"Model saved to {model_path}")

    # Save class labels
    with open(classes_path, 'w') as f:
        for c in train_dataset.classes:
            f.write(c + '\n')
    print(f"Class labels saved to {classes_path}")

    train_dataset.close()
    val_dataset.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train modulation classifier on 5G jamming dataset")
    parser.add_argument('--data', type=str, default='gnuradio_jamming_dataset.h5', help='HDF5 dataset path')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--model_path', type=str, default='modulation_cnn.pt')
    parser.add_argument('--classes_path', type=str, default='classes.txt')
    args = parser.parse_args()

    train(args.data, args.epochs, args.batch_size, args.lr, args.model_path, args.classes_path)
