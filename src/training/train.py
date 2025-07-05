import sys
import os
# ensure project root is on PYTHONPATH for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

sys.path.insert(0, os.path.abspath(os.getcwd()))
import argparse
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
import numpy as np
from pathlib import Path
from src.data.modulation_jamming_dataset import ModulationJammingDataset
from src.models.multi_task_cnn import MultiTaskCNN
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import logging

logging.basicConfig(level=logging.INFO)

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    train_ds = ModulationJammingDataset(args.data, split='train')
    val_ds = ModulationJammingDataset(args.data, split='val')
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = MultiTaskCNN(len(train_ds.mod_classes), len(train_ds.jam_type_classes), dropout=args.dropout).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        all_out = {'mod': [], 'jam': [], 'jam_type': []}
        all_labels = {'mod': [], 'jam': [], 'jam_type': []}

        for x, mod, jam, jam_type in train_loader:
            x, mod, jam, jam_type = x.to(device), mod.to(device), jam.to(device), jam_type.to(device)
            out = model(x)

            loss_mod = loss_fn(out.mod, mod)
            loss_jam = loss_fn(out.jam, jam)
            mask = (jam == 1)
            if mask.any():
                loss_jam_type = loss_fn(out.jam_type[mask], jam_type[mask])
            else:
                loss_jam_type = torch.tensor(0.0, device=device)
            loss = loss_mod + loss_jam + loss_jam_type

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            # store predictions and labels for metrics
            preds_mod = out.mod.argmax(dim=1).cpu().numpy()
            preds_jam = out.jam.argmax(dim=1).cpu().numpy()
            preds_jam_type = out.jam_type.argmax(dim=1).cpu().numpy()
            all_out['mod'].extend(preds_mod)
            all_out['jam'].extend(preds_jam)
            all_out['jam_type'].extend(preds_jam_type[mask.cpu().numpy()])
            all_labels['mod'].extend(mod.cpu().numpy())
            all_labels['jam'].extend(jam.cpu().numpy())
            all_labels['jam_type'].extend(jam_type.cpu().numpy()[mask.cpu().numpy()])

        train_loss = np.mean(losses)
        train_acc = accuracy_score(all_labels['mod'], all_out['mod'])
        logging.info(f"Epoch {epoch}/{args.epochs} - Train Loss: {train_loss:.4f} - Modulation Acc: {train_acc:.4f}")

        # Validation
        model.eval()
        val_losses = []
        val_out = {'mod': [], 'jam': [], 'jam_type': []}
        val_labels = {'mod': [], 'jam': [], 'jam_type': []}
        with torch.no_grad():
            for x, mod, jam, jam_type in val_loader:
                x, mod, jam, jam_type = x.to(device), mod.to(device), jam.to(device), jam_type.to(device)
                out = model(x)
                v_loss_mod = loss_fn(out.mod, mod)
                v_loss_jam = loss_fn(out.jam, jam)
                mask_val = (jam == 1)
                if mask_val.any():
                    v_loss_jam_type = loss_fn(out.jam_type[mask_val], jam_type[mask_val])
                else:
                    v_loss_jam_type = torch.tensor(0.0, device=device)
                val_losses.append((v_loss_mod+v_loss_jam+v_loss_jam_type).item())

                val_out['mod'].extend(out.mod.argmax(dim=1).cpu().numpy())
                val_out['jam'].extend(out.jam.argmax(dim=1).cpu().numpy())
                val_out['jam_type'].extend(out.jam_type.argmax(dim=1).cpu().numpy()[mask_val.cpu().numpy()])
                val_labels['mod'].extend(mod.cpu().numpy())
                val_labels['jam'].extend(jam.cpu().numpy())
                val_labels['jam_type'].extend(jam_type.cpu().numpy()[mask_val.cpu().numpy()])

        val_loss = np.mean(val_losses)
        val_mod_acc = accuracy_score(val_labels['mod'], val_out['mod'])
        val_f1 = f1_score(val_labels['mod'], val_out['mod'], average='weighted')
        cm = confusion_matrix(val_labels['mod'], val_out['mod'])
        logging.info(f"Epoch {epoch}/{args.epochs} - Val Loss: {val_loss:.4f} - Modulation Acc: {val_mod_acc:.4f} - Modulation F1: {val_f1:.4f}")
        logging.info(f"Confusion Matrix:\n{cm}")

    # Save TorchScript and class files
    trace_input = torch.randn(1, 2, args.input_length).to(device)
    traced = torch.jit.trace(model, trace_input, strict=False)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_file = out_dir / f"{args.model_name}.pt"
    traced.save(model_file)
    logging.info(f"Saved TorchScript model to {model_file}")

    # Save class mappings
    with open(out_dir / 'mod_classes.txt', 'w') as f:
        for c in train_ds.mod_classes:
            f.write(c + '\n')
    with open(out_dir / 'jam_type_classes.txt', 'w') as f:
        for c in train_ds.jam_type_classes:
            f.write(c + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Unified training for multi-task RFML model")
    parser.add_argument('--data', type=str, default='gnuradio_jamming_dataset.h5')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--input_length', type=int, default=1024)
    parser.add_argument('--output_dir', type=str, default='deployment')
    parser.add_argument('--model_name', type=str, default='multitask_cnn')
    parser.add_argument('--config', type=str, help='YAML/JSON config file path')
    args = parser.parse_args()
    if args.config:
        import yaml, json
        with open(args.config, 'r') as f:
            if args.config.lower().endswith(('.yml', '.yaml')):
                cfg = yaml.safe_load(f)
            else:
                cfg = json.load(f)
        for k, v in cfg.items():
            if hasattr(args, k):
                setattr(args, k, v)
    train(args)
