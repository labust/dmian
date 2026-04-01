#!/usr/bin/env python3
"""
Training script
"""

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import yaml
import argparse
import time
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from dataset_processor.dataset_processor import IMUDVLDataset
from .networks import EnsemblePredictor, combined_loss

class Trainer:
    def __init__(self, config_path):
        self.config = self.load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {self.device}")
        
        self.setup_model()
        self.setup_datasets()
        self.setup_training()
        
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def load_config(self, path):
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def setup_model(self):
        """Initialize ensemble model"""
        net_cfg = self.config['network']
        self.model = EnsemblePredictor(
            num_models=net_cfg['num_models'],
            hidden_size=net_cfg['hidden_size'],
            num_lstm_layers=net_cfg['num_lstm_layers'],
            dropout=net_cfg['dropout']
        ).to(self.device)
        
        params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {params:,}")
    
    def setup_datasets(self):
        """Setup train/val datasets"""
        ds_cfg = self.config['dataset']
        
        train_dirs = ds_cfg.get('train_dirs', [ds_cfg.get('train_dir')])
        val_dirs = ds_cfg.get('val_dirs', [ds_cfg.get('val_dir')])
        
        train_dirs = [d for d in train_dirs if d is not None]
        val_dirs = [d for d in val_dirs if d is not None]
        
        print(f"\nLoading datasets (using imu1, imu2, imu3)...")
        print(f"Training trajectories: {len(train_dirs)}")
        print(f"Validation trajectories: {len(val_dirs)}")
        
        train_dataset = IMUDVLDataset(
            train_dirs,
            ds_cfg,
            train=True
        )
        
        val_dataset = IMUDVLDataset(
            val_dirs,
            ds_cfg,
            train=False
        )
        
        batch_cfg = self.config['training']['batch_size']
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_cfg['train'],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_cfg['test'],
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
    
    def setup_training(self):
        """Setup optimizers and schedulers"""
        train_cfg = self.config['training']
        
        self.optimizers = []
        self.schedulers = []
        
        for model in self.model.models:
            optimizer = optim.Adam(
                model.parameters(),
                lr=train_cfg['learning_rate'],
                weight_decay=train_cfg.get('weight_decay', 1e-4)
            )
            
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=train_cfg['scheduler']['factor'],
                patience=train_cfg['scheduler']['patience'],
                verbose=True
            )
            
            self.optimizers.append(optimizer)
            self.schedulers.append(scheduler)
        
        self.loss_weights = train_cfg['loss_weights']
        self.gnll_start_epoch = train_cfg.get('gnll_start_epoch', 15)
        self.patience = train_cfg['early_stopping_patience']
    
    def train_epoch(self, epoch):
        """Train one epoch with phased loss"""
        self.model.train()
        losses = []
        mse_losses = []
        gnll_losses = []
        
        use_mse_only = epoch < self.gnll_start_epoch
        phase = "MSE" if use_mse_only else "GNLL"
        
        for batch_idx, batch in enumerate(self.train_loader):
            imu1 = batch['imu1_data'].to(self.device)
            imu2 = batch['imu2_data'].to(self.device)
            imu3 = batch['imu3_data'].to(self.device)
            dvl = batch['dvl_data'].to(self.device)
            gt_vel = batch['ground_truth'].to(self.device)
            
            batch_loss = 0
            batch_mse = 0
            batch_gnll = 0
            
            for model, optimizer in zip(self.model.models, self.optimizers):
                optimizer.zero_grad()
                
                vel_pred, log_std_pred = model(imu1, imu2, imu3, dvl)
                
                _, mse, gnll = combined_loss(
                    vel_pred, log_std_pred, gt_vel, 1.0, 1.0
                )
                
                loss = mse if use_mse_only else gnll
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                batch_loss += loss.item()
                batch_mse += mse.item()
                batch_gnll += gnll.item()
            
            batch_loss /= len(self.model.models)
            batch_mse /= len(self.model.models)
            batch_gnll /= len(self.model.models)
            
            losses.append(batch_loss)
            mse_losses.append(batch_mse)
            gnll_losses.append(batch_gnll)
            
            if batch_idx % 50 == 0:
                print(f'  Batch {batch_idx}/{len(self.train_loader)} ({phase}): '
                      f'Loss={batch_loss:.4f}, MSE={batch_mse:.4f}, GNLL={batch_gnll:.4f}')
        
        return np.mean(losses), np.mean(mse_losses), np.mean(gnll_losses), phase
    
    def validate(self, epoch):
        self.model.eval()
        losses = []
        mse_losses = []
        gnll_losses = []
        
        use_mse_only = epoch < self.gnll_start_epoch
        
        with torch.no_grad():
            for batch in self.val_loader:
                imu1 = batch['imu1_data'].to(self.device)
                imu2 = batch['imu2_data'].to(self.device)
                imu3 = batch['imu3_data'].to(self.device)
                dvl = batch['dvl_data'].to(self.device)
                gt_vel = batch['ground_truth'].to(self.device)
                
                vel_mean, aleatoric, epistemic = self.model(imu1, imu2, imu3, dvl)
                _, log_std = self.model.models[0](imu1, imu2, imu3, dvl)
                
                total, mse, gnll = combined_loss(
                    vel_mean, log_std, gt_vel,
                    self.loss_weights['mse_weight'],
                    self.loss_weights['gnll_weight']
                )
                
                losses.append(mse.item() if use_mse_only else total.item())
                mse_losses.append(mse.item())
                gnll_losses.append(gnll.item())
        
        return np.mean(losses), np.mean(mse_losses), np.mean(gnll_losses)
    
    def train(self):
        """Main training loop"""
        max_epochs = self.config['training']['max_epochs']
        save_dir = Path(self.config['training']['save_dir'])
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nStarting training for {max_epochs} epochs...")
        
        for epoch in range(max_epochs):
            start_time = time.time()
            
            train_loss, train_mse, train_gnll, phase = self.train_epoch(epoch)
            val_loss, val_mse, val_gnll = self.validate(epoch)
            
            for scheduler in self.schedulers:
                scheduler.step(val_loss)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            elapsed = time.time() - start_time
            
            print(f'\nEpoch {epoch+1}/{max_epochs} ({phase}):')
            print(f'  Train: {train_loss:.4f} (MSE: {train_mse:.4f}, GNLL: {train_gnll:.4f})')
            print(f'  Val: {val_loss:.4f} (MSE: {val_mse:.4f}, GNLL: {val_gnll:.4f})')
            print(f'  Time: {elapsed:.1f}s')
            
            if epoch + 1 == self.gnll_start_epoch:
                print(f'  >>> Switching to GNLL phase <<<')
                self.patience_counter = 0
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                print(f'  New best! Saving...')
                self.save_models(save_dir / 'best')
            else:
                self.patience_counter += 1
            
            if self.patience_counter >= self.patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
            
            if (epoch + 1) % 2 == 0:
                self.save_models(save_dir / f'checkpoint_epoch_{epoch+1}')
        
        print(f'\nTraining complete. Best val loss: {self.best_val_loss:.4f}')
        
        self.save_models(save_dir / 'final')
        self.plot_training(save_dir)
    
    def save_models(self, path):
        """Save all models"""
        path.mkdir(parents=True, exist_ok=True)
        
        for i, model in enumerate(self.model.models):
            torch.save(model.state_dict(), path / f'model_{i}.pt')
        
        with open(path / 'config.yaml', 'w') as f:
            yaml.dump(self.config, f)
    
    def plot_training(self, save_dir):
        """Plot training progress"""
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train')
        plt.plot(self.val_losses, label='Val')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training Progress')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.train_losses, label='Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Detail')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'training_progress.png', dpi=300, bbox_inches='tight')
        print(f"Training plot saved to {save_dir / 'training_progress.png'}")


def main():
    parser = argparse.ArgumentParser(description='Train DMIAN model')
    parser.add_argument('--config', type=str, required=True, 
                       help='Path to config file')
    args = parser.parse_args()
    
    trainer = Trainer(args.config)
    trainer.train()


if __name__ == '__main__':
    main()