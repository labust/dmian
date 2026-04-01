#!/usr/bin/env python3
"""
LSTM Network for 3 IMUs + DVL velocity estimation with covariance output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class IMUDVLFusionLSTM(nn.Module):
    def __init__(self, hidden_size=64, num_lstm_layers=2, dropout=0.1):
        super(IMUDVLFusionLSTM, self).__init__()
        
        self.imu_input_size = 6  
        self.dvl_input_size = 3  
        self.hidden_size = hidden_size
        self.num_layers = num_lstm_layers
        self.dropout = dropout
        
        self.imu1_lstm = nn.LSTM(
            self.imu_input_size,
            self.hidden_size,
            self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0
        )
        
        self.imu2_lstm = nn.LSTM(
            self.imu_input_size,
            self.hidden_size,
            self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0
        )
        
        self.imu3_lstm = nn.LSTM(
            self.imu_input_size,
            self.hidden_size,
            self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0
        )
        
        self.dvl_lstm = nn.LSTM(
            self.dvl_input_size,
            self.hidden_size,
            self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0
        )
        
        fusion_input = 4 * self.hidden_size
        self.fusion_fc1 = nn.Linear(fusion_input, 64)
        self.fusion_fc2 = nn.Linear(64, 32)
        
        self.velocity_head = nn.Linear(32, 3) 
        self.log_std_head = nn.Linear(32, 3) 
        
        self.activation = nn.PReLU()
        self.dropout_layer = nn.Dropout(self.dropout)
    
    def forward(self, imu1_seq, imu2_seq, imu3_seq, dvl_seq):
        """
        Args:
            imu1_seq: [batch, seq_len, 6]
            imu2_seq: [batch, seq_len, 6]
            imu3_seq: [batch, seq_len, 6]
            dvl_seq: [batch, seq_len, 3]
        
        Returns:
            velocity: [batch, seq_len, 3]
            log_std: [batch, seq_len, 3] (diagonal covariance elements)
        """
        imu1_features, _ = self.imu1_lstm(imu1_seq)
        imu2_features, _ = self.imu2_lstm(imu2_seq)
        imu3_features, _ = self.imu3_lstm(imu3_seq)
        
        dvl_features, _ = self.dvl_lstm(dvl_seq)
        
        fused = torch.cat([imu1_features, imu2_features, imu3_features, dvl_features], dim=-1)
        
        x = self.activation(self.fusion_fc1(fused))
        x = self.dropout_layer(x)
        x = self.activation(self.fusion_fc2(x))
        x = self.dropout_layer(x)
        
        velocity = self.velocity_head(x)
        log_std = self.log_std_head(x)
        
        return velocity, log_std


class EnsemblePredictor(nn.Module):
    """Ensemble of models for uncertainty quantification"""
    
    def __init__(self, num_models=3, hidden_size=64, num_lstm_layers=2, dropout=0.1):
        super(EnsemblePredictor, self).__init__()
        
        self.num_models = num_models
        self.models = nn.ModuleList([
            IMUDVLFusionLSTM(hidden_size, num_lstm_layers, dropout)
            for _ in range(num_models)
        ])
    
    def forward(self, imu1_seq, imu2_seq, imu3_seq, dvl_seq):
        """
        Args:
            imu1_seq: [batch, seq_len, 6]
            imu2_seq: [batch, seq_len, 6]
            imu3_seq: [batch, seq_len, 6]
            dvl_seq: [batch, seq_len, 3]
        
        Returns:
            velocity_mean: [batch, seq_len, 3]
            aleatoric_unc: [batch, seq_len, 3] 
            epistemic_unc: [batch, seq_len, 3]
        """
        predictions = []
        log_stds = []
        
        for model in self.models:
            vel, log_std = model(imu1_seq, imu2_seq, imu3_seq, dvl_seq)
            predictions.append(vel)
            log_stds.append(log_std)
        
        pred_stack = torch.stack(predictions, dim=0)
        log_std_stack = torch.stack(log_stds, dim=0)
        
        velocity_mean = pred_stack.mean(dim=0)
        
        variance_stack = torch.exp(2 * log_std_stack)
        aleatoric_unc = variance_stack.mean(dim=0)
        
        epistemic_unc = pred_stack.var(dim=0)
        
        return velocity_mean, aleatoric_unc, epistemic_unc


def combined_loss(velocity_pred, log_std_pred, velocity_true, mse_weight=1.0, gnll_weight=1.0):
    """
    Combined MSE + GNLL loss
    
    Args:
        velocity_pred: [batch, seq, 3]
        log_std_pred: [batch, seq, 3]
        velocity_true: [batch, seq, 3]
    """
    mse_loss = F.mse_loss(velocity_pred, velocity_true)
    
    variance = torch.exp(2 * log_std_pred)
    squared_error = (velocity_pred - velocity_true).pow(2)
    gnll = 0.5 * (squared_error / variance + 2 * log_std_pred)
    gnll_loss = gnll.mean()
    
    total_loss = mse_weight * mse_loss + gnll_weight * gnll_loss
    
    return total_loss, mse_loss, gnll_loss


def log_std_to_covariance(log_std):
    """
    Convert log std to 3x3 covariance matrix (diagonal)
    
    Args:
        log_std: [3] or [batch, 3] or [batch, seq, 3]
    
    Returns:
        covariance: [..., 3, 3]
    """
    variance = torch.exp(2 * log_std)
    
    cov_shape = list(variance.shape) + [3]
    covariance = torch.zeros(cov_shape, device=variance.device)
    
    if variance.ndim == 1:
        covariance[0, 0] = variance[0]
        covariance[1, 1] = variance[1]
        covariance[2, 2] = variance[2]
    elif variance.ndim == 2:
        covariance[:, 0, 0] = variance[:, 0]
        covariance[:, 1, 1] = variance[:, 1]
        covariance[:, 2, 2] = variance[:, 2]
    else:  # ndim == 3
        covariance[:, :, 0, 0] = variance[:, :, 0]
        covariance[:, :, 1, 1] = variance[:, :, 1]
        covariance[:, :, 2, 2] = variance[:, :, 2]
    
    return covariance


def numpy_log_std_to_covariance(log_std):
    """
    Convert numpy log std to 3x3 covariance matrix
    
    Args:
        log_std: [3] numpy array
    
    Returns:
        covariance: [3, 3] numpy array
    """
    variance = np.exp(2 * log_std)
    return np.diag(variance)
