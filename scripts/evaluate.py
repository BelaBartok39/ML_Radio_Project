#!/usr/bin/env python3
# ensure project root on PYTHONPATH before imports
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
"""
Evaluation and debugging utilities for RFML model:
- Console reports: accuracy, F1-score
- Confusion matrices for modulation and jamming
- Model confidence visualization
- Optional test-time augmentation (signal reversal)
"""
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from src.data.modulation_jamming_dataset import ModulationJammingDataset
import sys
import os

# ensure project root is on PYTHONPATH for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def plot_confusion(cm, classes, title, out_file):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                 ha='center', va='center',
                 color='white' if cm[i, j] > thresh else 'black')
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(out_file)
    plt.close()


def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model from {args.model}")
    model = torch.jit.load(args.model).to(device)
    model.eval()

    # Load dataset
    ds = ModulationJammingDataset(args.data, split=args.split)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    # Loads classes
    mod_classes = [l.strip() for l in open(args.mod_classes, 'r')]
    jam_type_classes = [l.strip() for l in open(args.jam_type_classes, 'r')]

    all_preds = {'mod': [], 'jam': [], 'jam_type': []}
    all_labels = {'mod': [], 'jam': [], 'jam_type': []}
    all_conf = []  # store max confidence per sample

    # Helper to unpack model outputs (TorchScript returns tuple)
    def _unpack(out):
        if isinstance(out, tuple) or isinstance(out, list):
            return out[0], out[1], out[2]
        return out.mod, out.jam, out.jam_type

    with torch.no_grad():
        for x, mod, jam, jam_type in loader:
            x = x.to(device)
            # test-time augmentation: reversal
            if args.tta:
                x_rev = torch.flip(x, dims=[2])
                o1 = model(x)
                o2 = model(x_rev)
                m1, j1, t1 = _unpack(o1)
                m2, j2, t2 = _unpack(o2)
                out_mod = (m1 + m2) / 2.0
                out_jam = (j1 + j2) / 2.0
                out_jam_type = (t1 + t2) / 2.0
            else:
                out = model(x)
                out_mod, out_jam, out_jam_type = _unpack(out)

            # predictions
            probs_mod = torch.softmax(out_mod, dim=1)
            probs_jam = torch.softmax(out_jam, dim=1)
            preds_mod = probs_mod.argmax(dim=1).cpu().numpy()
            preds_jam = probs_jam.argmax(dim=1).cpu().numpy()

            # record confidences
            all_conf.extend(probs_mod.max(dim=1)[0].cpu().numpy())

            # jam_type only on jammed
            mask = preds_jam == 1
            preds_jam_type = out_jam_type.argmax(dim=1).cpu().numpy()

            all_preds['mod'].extend(preds_mod.tolist())
            all_preds['jam'].extend(preds_jam.tolist())
            all_preds['jam_type'].extend(preds_jam_type[mask].tolist())

            all_labels['mod'].extend(mod.numpy().tolist())
            all_labels['jam'].extend(jam.numpy().tolist())
            all_labels['jam_type'].extend(jam_type.numpy()[mask].tolist())

    # Metrics
    acc_mod = accuracy_score(all_labels['mod'], all_preds['mod'])
    f1_mod = f1_score(all_labels['mod'], all_preds['mod'], average='weighted')
    acc_jam = accuracy_score(all_labels['jam'], all_preds['jam'])
    f1_jam = f1_score(all_labels['jam'], all_preds['jam'], average='weighted')
    print(f"Modulation Accuracy: {acc_mod:.4f}, F1-score: {f1_mod:.4f}")
    print(f"Jamming Accuracy: {acc_jam:.4f}, F1-score: {f1_jam:.4f}")

    # Confusion matrices
    cm_mod = confusion_matrix(all_labels['mod'], all_preds['mod'])
    cm_jam = confusion_matrix(all_labels['jam'], all_preds['jam'])
    plot_confusion(cm_mod, mod_classes, 'Modulation Confusion Matrix', args.out_dir + '/confusion_mod.png')
    plot_confusion(cm_jam, ['no-jam', 'jam'], 'Jamming Confusion Matrix', args.out_dir + '/confusion_jam.png')
    print(f"Confusion matrices saved to {args.out_dir}")

    # Confidence histogram
    import matplotlib.pyplot as plt
    plt.hist(all_conf, bins=20)
    plt.title('Prediction Confidence Histogram')
    plt.xlabel('Max class probability')
    plt.ylabel('Frequency')
    plt.savefig(args.out_dir + '/confidence_hist.png')
    plt.close()
    print(f"Confidence histogram saved to {args.out_dir}/confidence_hist.png")

    ds.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate RFML model with debug utilities")
    parser.add_argument('--data', type=str, default='gnuradio_jamming_dataset.h5', help='HDF5 dataset file')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val'], help='Dataset split')
    parser.add_argument('--model', type=str, required=True, help='TorchScript model file')
    parser.add_argument('--mod-classes', type=str, required=True, help='Modulation classes text file')
    parser.add_argument('--jam-type-classes', type=str, required=True, help='Jamming type classes text file')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--out-dir', type=str, default='evaluation')
    parser.add_argument('--tta', action='store_true', help='Enable test-time augmentation (signal reversal)')
    args = parser.parse_args()
    import os
    os.makedirs(args.out_dir, exist_ok=True)
    evaluate(args)
