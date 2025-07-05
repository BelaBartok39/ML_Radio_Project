#!/usr/bin/env python3
"""
Production-Grade RF-ML Training Script
Optimized for large datasets and multi-GPU training
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import argparse
import yaml
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from src.data.modulation_jamming_dataset import ModulationJammingDataset
from src.models.multi_task_cnn import MultiTaskCNN

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class RFMLTrainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.use_amp = config.get('use_amp', False)
        # Initialize GradScaler with correct device
        # Initialize GradScaler for automatic mixed precision (device inferred from tensors)
        self.scaler = GradScaler() if self.use_amp else None
        
        # Setup output directories
        self.output_dir = Path(config.get('output_dir', 'deployments/production'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup TensorBoard logging
        tb_dir = config.get('tensorboard_log_dir', 'logs/production')
        self.writer = SummaryWriter(tb_dir)
        
        # Initialize datasets and model
        self._setup_data()
        self._setup_model()
        self._setup_optimizer()
        
        # Training state
        self.best_metric = 0.0
        self.epochs_without_improvement = 0
        self.global_step = 0
        
    def _setup_data(self):
        """Setup data loaders with production optimizations"""
        logger.info("Setting up datasets...")
        
        self.train_ds = ModulationJammingDataset(self.config['data'], split='train')
        self.val_ds = ModulationJammingDataset(self.config['data'], split='val')
        
        # Production data loader settings
        loader_kwargs = {
            'batch_size': self.config.get('batch_size', 64),
            'num_workers': self.config.get('num_workers', 4),
            'pin_memory': self.config.get('pin_memory', True),
            'prefetch_factor': self.config.get('prefetch_factor', 2),
            'persistent_workers': self.config.get('dataloader_persistent_workers', True)
        }
        
        self.train_loader = DataLoader(self.train_ds, shuffle=True, **loader_kwargs)
        self.val_loader = DataLoader(self.val_ds, shuffle=False, **loader_kwargs)
        
        logger.info(f"Training samples: {len(self.train_ds)}")
        logger.info(f"Validation samples: {len(self.val_ds)}")
        logger.info(f"Modulation classes: {self.train_ds.mod_classes}")
        logger.info(f"Jamming types: {self.train_ds.jam_type_classes}")
        
    def _setup_model(self):
        """Setup model with production optimizations"""
        logger.info("Setting up model...")
        
        self.model = MultiTaskCNN(
            num_mod_classes=len(self.train_ds.mod_classes),
            num_jam_types=len(self.train_ds.jam_type_classes),
            input_length=self.config.get('input_length', 1024),
            dropout=self.config.get('dropout', 0.3)
        ).to(self.device)
        
        # PyTorch 2.0 compilation for speed
        if self.config.get('torch_compile', False) and hasattr(torch, 'compile'):
            logger.info("Compiling model with PyTorch 2.0...")
            self.model = torch.compile(self.model)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
    def _setup_optimizer(self):
        """Setup optimizer and scheduler"""
        # Optimizer
        optimizer_name = self.config.get('optimizer', 'adam').lower()
        lr = self.config.get('lr', 1e-3)
        weight_decay = self.config.get('weight_decay', 0.0)
        
        if optimizer_name == 'adamw':
            self.optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            self.optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Learning rate scheduler
        scheduler_name = self.config.get('lr_scheduler', 'cosine')
        epochs = self.config.get('epochs', 50)
        
        if scheduler_name == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer, 
                T_max=epochs, 
                eta_min=self.config.get('min_lr', 1e-6)
            )
        elif scheduler_name == 'onecycle':
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=lr,
                epochs=epochs,
                steps_per_epoch=len(self.train_loader)
            )
        else:
            self.scheduler = None
            
        logger.info(f"Using {optimizer_name} optimizer with {scheduler_name} scheduler")
        
    def train_epoch(self, epoch: int):
        """Train one epoch"""
        self.model.train()
        losses = []
        predictions = {'mod': [], 'jam': [], 'jam_type': []}
        targets = {'mod': [], 'jam': [], 'jam_type': []}
        
        # Get loss weights
        mod_weight = self.config.get('mod_loss_weight', 1.0)
        jam_weight = self.config.get('jam_detection_weight', 0.5)
        jamtype_weight = self.config.get('jam_type_weight', 0.3)
        
        start_time = time.time()
        
        for batch_idx, (x, mod, jam, jam_type) in enumerate(self.train_loader):
            x, mod, jam, jam_type = x.to(self.device), mod.to(self.device), jam.to(self.device), jam_type.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass with optional mixed precision
            if self.use_amp:
                with autocast():
                    out = self.model(x)
                    loss = self._compute_loss(out, mod, jam, jam_type, mod_weight, jam_weight, jamtype_weight)
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.get('grad_clip_norm'):
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip_norm'])
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                out = self.model(x)
                loss = self._compute_loss(out, mod, jam, jam_type, mod_weight, jam_weight, jamtype_weight)
                loss.backward()
                
                # Gradient clipping
                if self.config.get('grad_clip_norm'):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip_norm'])
                
                self.optimizer.step()
            
            # Update scheduler (for OneCycle)
            if self.scheduler and self.config.get('lr_scheduler') == 'onecycle':
                self.scheduler.step()
            
            losses.append(loss.item())
            
            # Store predictions and targets
            predictions['mod'].extend(out.mod.argmax(dim=1).cpu().numpy())
            predictions['jam'].extend(out.jam.argmax(dim=1).cpu().numpy())
            
            jam_mask = (jam == 1).cpu().numpy()
            if jam_mask.any():
                predictions['jam_type'].extend(out.jam_type.argmax(dim=1).cpu().numpy()[jam_mask])
                targets['jam_type'].extend(jam_type.cpu().numpy()[jam_mask])
            
            targets['mod'].extend(mod.cpu().numpy())
            targets['jam'].extend(jam.cpu().numpy())
            
            # Logging
            if batch_idx % 100 == 0:
                logger.info(f"Epoch {epoch}, Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}")
                self.writer.add_scalar('Loss/Train_Batch', loss.item(), self.global_step)
            
            self.global_step += 1
        
        # Update scheduler (for others)
        if self.scheduler and self.config.get('lr_scheduler') != 'onecycle':
            self.scheduler.step()
        
        # Compute epoch metrics
        epoch_loss = np.mean(losses)
        epoch_time = time.time() - start_time
        
        mod_acc = accuracy_score(targets['mod'], predictions['mod'])
        jam_acc = accuracy_score(targets['jam'], predictions['jam'])
        
        logger.info(f"Epoch {epoch} Training - Loss: {epoch_loss:.4f}, Mod Acc: {mod_acc:.4f}, Jam Acc: {jam_acc:.4f}, Time: {epoch_time:.1f}s")
        
        # TensorBoard logging
        self.writer.add_scalar('Loss/Train_Epoch', epoch_loss, epoch)
        self.writer.add_scalar('Accuracy/Train_Modulation', mod_acc, epoch)
        self.writer.add_scalar('Accuracy/Train_Jamming', jam_acc, epoch)
        self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
        
        return epoch_loss, mod_acc
        
    def validate(self, epoch: int):
        """Validate model"""
        self.model.eval()
        losses = []
        predictions = {'mod': [], 'jam': [], 'jam_type': []}
        targets = {'mod': [], 'jam': [], 'jam_type': []}
        
        # Get loss weights
        mod_weight = self.config.get('mod_loss_weight', 1.0)
        jam_weight = self.config.get('jam_detection_weight', 0.5)
        jamtype_weight = self.config.get('jam_type_weight', 0.3)
        
        with torch.no_grad():
            for x, mod, jam, jam_type in self.val_loader:
                x, mod, jam, jam_type = x.to(self.device), mod.to(self.device), jam.to(self.device), jam_type.to(self.device)
                
                if self.use_amp:
                    with autocast():
                        out = self.model(x)
                        loss = self._compute_loss(out, mod, jam, jam_type, mod_weight, jam_weight, jamtype_weight)
                else:
                    out = self.model(x)
                    loss = self._compute_loss(out, mod, jam, jam_type, mod_weight, jam_weight, jamtype_weight)
                
                losses.append(loss.item())
                
                # Store predictions and targets
                predictions['mod'].extend(out.mod.argmax(dim=1).cpu().numpy())
                predictions['jam'].extend(out.jam.argmax(dim=1).cpu().numpy())
                
                jam_mask = (jam == 1).cpu().numpy()
                if jam_mask.any():
                    predictions['jam_type'].extend(out.jam_type.argmax(dim=1).cpu().numpy()[jam_mask])
                    targets['jam_type'].extend(jam_type.cpu().numpy()[jam_mask])
                
                targets['mod'].extend(mod.cpu().numpy())
                targets['jam'].extend(jam.cpu().numpy())
        
        # Compute metrics
        val_loss = np.mean(losses)
        mod_acc = accuracy_score(targets['mod'], predictions['mod'])
        mod_f1 = f1_score(targets['mod'], predictions['mod'], average='weighted')
        jam_acc = accuracy_score(targets['jam'], predictions['jam'])
        
        logger.info(f"Epoch {epoch} Validation - Loss: {val_loss:.4f}, Mod Acc: {mod_acc:.4f}, Mod F1: {mod_f1:.4f}, Jam Acc: {jam_acc:.4f}")
        
        # TensorBoard logging
        self.writer.add_scalar('Loss/Val_Epoch', val_loss, epoch)
        self.writer.add_scalar('Accuracy/Val_Modulation', mod_acc, epoch)
        self.writer.add_scalar('F1/Val_Modulation', mod_f1, epoch)
        self.writer.add_scalar('Accuracy/Val_Jamming', jam_acc, epoch)
        
        # Confusion matrix
        if epoch % 10 == 0:  # Log confusion matrix every 10 epochs
            cm = confusion_matrix(targets['mod'], predictions['mod'])
            logger.info(f"Modulation Confusion Matrix:\n{cm}")
        
        return val_loss, mod_acc, mod_f1
        
    def _compute_loss(self, out, mod, jam, jam_type, mod_weight, jam_weight, jamtype_weight):
        """Compute multi-task loss"""
        loss_fn = nn.CrossEntropyLoss()
        
        loss_mod = loss_fn(out.mod, mod)
        loss_jam = loss_fn(out.jam, jam)
        
        # Only compute jam type loss for actually jammed samples
        jam_mask = (jam == 1)
        if jam_mask.any():
            loss_jam_type = loss_fn(out.jam_type[jam_mask], jam_type[jam_mask])
        else:
            loss_jam_type = torch.tensor(0.0, device=self.device)
        
        total_loss = (mod_weight * loss_mod + 
                     jam_weight * loss_jam + 
                     jamtype_weight * loss_jam_type)
        
        return total_loss
        
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'config': self.config,
            'best_metric': self.best_metric
        }
        
        # Save regular checkpoint
        checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.output_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")
            
    def save_torchscript_model(self):
        """Save production TorchScript model"""
        self.model.eval()
        # Use original model for TorchScript tracing if compiled with torch.compile
        model_to_trace = getattr(self.model, '_orig_mod', self.model)
        trace_input = torch.randn(1, 2, self.config.get('input_length', 1024)).to(self.device)
        traced_model = torch.jit.trace(model_to_trace, trace_input, strict=False)
        
        model_path = self.output_dir / f"{self.config.get('model_name', 'rfml_model')}.pt"
        traced_model.save(model_path)
        logger.info(f"Saved TorchScript model to {model_path}")
        
        # Save class mappings
        with open(self.output_dir / 'mod_classes.txt', 'w') as f:
            for cls in self.train_ds.mod_classes:
                f.write(f"{cls}\n")
        
        with open(self.output_dir / 'jam_type_classes.txt', 'w') as f:
            for cls in self.train_ds.jam_type_classes:
                f.write(f"{cls}\n")
                
        logger.info("Saved class mapping files")
        
    def train(self):
        """Main training loop"""
        epochs = self.config.get('epochs', 50)
        save_every = self.config.get('save_every_n_epochs', 10)
        early_stopping_patience = self.config.get('early_stopping_patience', 15)
        best_metric_name = self.config.get('best_metric', 'val_mod_acc')
        
        logger.info(f"Starting training for {epochs} epochs...")
        logger.info(f"Device: {self.device}")
        logger.info(f"Batch size: {self.config.get('batch_size', 64)}")
        logger.info(f"Mixed precision: {self.use_amp}")
        
        for epoch in range(1, epochs + 1):
            # Training
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validation
            if epoch % self.config.get('val_frequency', 1) == 0:
                val_loss, val_mod_acc, val_mod_f1 = self.validate(epoch)
                
                # Check if this is the best model
                current_metric = val_mod_acc if best_metric_name == 'val_mod_acc' else val_mod_f1
                is_best = current_metric > self.best_metric
                
                if is_best:
                    self.best_metric = current_metric
                    self.epochs_without_improvement = 0
                    self.save_checkpoint(epoch, is_best=True)
                else:
                    self.epochs_without_improvement += 1
                
                # Early stopping
                if self.epochs_without_improvement >= early_stopping_patience:
                    logger.info(f"Early stopping triggered after {early_stopping_patience} epochs without improvement")
                    break
            
            # Save regular checkpoint
            if epoch % save_every == 0:
                self.save_checkpoint(epoch)
        
        # Save final TorchScript model
        self.save_torchscript_model()
        
        # Close TensorBoard writer
        self.writer.close()
        
        logger.info("Training completed!")

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML or JSON file"""
    with open(config_path, 'r') as f:
        if config_path.lower().endswith(('.yml', '.yaml')):
            return yaml.safe_load(f)
        else:
            return json.load(f)

def main():
    parser = argparse.ArgumentParser(description="Production RF-ML Training")
    parser.add_argument('--config', type=str, required=True, help='Config file (YAML/JSON)')
    parser.add_argument('--override', nargs='+', help='Override config values (key=value)')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config values from command line
    if args.override:
        for override in args.override:
            key, value = override.split('=', 1)
            # Simple type inference
            if value.lower() in ['true', 'false']:
                value = value.lower() == 'true'
            elif value.isdigit():
                value = int(value)
            elif value.replace('.', '').isdigit():
                value = float(value)
            config[key] = value
    
    # Initialize and run trainer
    trainer = RFMLTrainer(config)
    trainer.train()

if __name__ == '__main__':
    main()
