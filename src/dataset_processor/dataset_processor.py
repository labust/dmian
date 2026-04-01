#!/usr/bin/env python3
"""
Dataset processor
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import yaml

class IMUDVLDataset(Dataset):
    def __init__(self, data_dirs, config, train=True):
        """
        Args:
            data_dirs: Path(s) to data directory/directories (can be string, Path, or list)
            config: Configuration dictionary
            train: Training mode flag
        """
        if isinstance(data_dirs, (str, Path)):
            data_dirs = [data_dirs]
        
        self.data_dirs = [Path(d) for d in data_dirs]
        self.config = config
        self.train = train
        
        self.seq_duration = config['seq_duration']
        self.seq_sample_density = config['seq_sample_density']
        
        self.target_fps = config['sensor_fps']['dvl']
        self.imu_fps = config['sensor_fps']['imu']
        self.dvl_fps = config['sensor_fps']['dvl']
        self.gt_fps = config['sensor_fps']['ground_truth']
        
        self.imu_downsample = max(1, self.imu_fps // self.target_fps)
        self.gt_downsample = max(1, self.gt_fps // self.target_fps)
        
        print(f"Dataset mode: {'TRAIN' if train else 'VAL'}")
        print(f"Using IMUs: imu1, imu2, imu3")
        print(f"Loading from {len(self.data_dirs)} trajectory/trajectories:")
        for d in self.data_dirs:
            print(f"  - {d.name}")
        print(f"Downsampling: IMU {self.imu_fps}Hz -> {self.target_fps}Hz (÷{self.imu_downsample})")
        print(f"Downsampling: GT {self.gt_fps}Hz -> {self.target_fps}Hz (÷{self.gt_downsample})")
        
        self.imu1_mean = np.array(config['sensor_stats']['imu_imu1']['mean'])
        self.imu1_std = np.array(config['sensor_stats']['imu_imu1']['std'])
        self.imu2_mean = np.array(config['sensor_stats']['imu_imu2']['mean'])
        self.imu2_std = np.array(config['sensor_stats']['imu_imu2']['std'])
        self.imu3_mean = np.array(config['sensor_stats']['imu_imu3']['mean'])
        self.imu3_std = np.array(config['sensor_stats']['imu_imu3']['std'])
        
        self.dvl_mean = np.array(config['sensor_stats']['dvl']['mean'])
        self.dvl_std = np.array(config['sensor_stats']['dvl']['std'])
        self.gt_mean = np.array(config['sensor_stats']['ground_truth']['mean'])
        self.gt_std = np.array(config['sensor_stats']['ground_truth']['std'])
        
        self.noise_config = config.get('noise', {})
        
        self.samples = self.load_and_process_data()
        
    def load_and_process_data(self):
        """Load and create sequence samples from all trajectories"""
        samples = []
        
        for traj_dir in self.data_dirs:
            print(f"\nProcessing: {traj_dir.name}")
            
            imu1_path = traj_dir / 'imu1_data.npy'
            imu2_path = traj_dir / 'imu2_data.npy'
            imu3_path = traj_dir / 'imu3_data.npy'
            dvl_path = traj_dir / 'dvl_data.npy'
            gt_path = traj_dir / 'ground_truth_data.npy'
            
            if not all([imu1_path.exists(), imu2_path.exists(), imu3_path.exists(),
                       dvl_path.exists(), gt_path.exists()]):
                print(f"  Skipping - missing data files")
                continue
            
            imu1_data = np.load(imu1_path)
            imu2_data = np.load(imu2_path)
            imu3_data = np.load(imu3_path)
            dvl_data = np.load(dvl_path)
            gt_data = np.load(gt_path)
            
            start_time = max(imu1_data[0, -1], imu2_data[0, -1], imu3_data[0, -1],
                           dvl_data[0, -1], gt_data[0, -1])
            end_time = min(imu1_data[-1, -1], imu2_data[-1, -1], imu3_data[-1, -1],
                          dvl_data[-1, -1], gt_data[-1, -1])
            duration = end_time - start_time
            
            print(f"  Duration: {duration:.1f}s")
            
            if duration < self.seq_duration:
                print(f"  Skipping - too short")
                continue
            
            num_samples = int(duration * self.seq_sample_density)
            traj_samples = self.create_sequences(
                imu1_data, imu2_data, imu3_data, dvl_data, gt_data, 
                start_time, end_time, num_samples
            )
            
            samples.extend(traj_samples)
            print(f"  Generated {len(traj_samples)} samples")
        
        print(f"\nTotal samples across all trajectories: {len(samples)}")
        return samples
    
    def create_sequences(self, imu1_data, imu2_data, imu3_data, dvl_data, gt_data, 
                        start_time, end_time, num_samples):
        samples = []
        
        valid_start = start_time + self.seq_duration
        valid_end = end_time - self.seq_duration
        
        if valid_end <= valid_start:
            return samples
        
        for _ in range(num_samples):
            sample_time = np.random.uniform(valid_start, valid_end)
            
            try:
                imu1_seq = self.extract_sequence(imu1_data, sample_time, self.imu_fps)
                imu2_seq = self.extract_sequence(imu2_data, sample_time, self.imu_fps)
                imu3_seq = self.extract_sequence(imu3_data, sample_time, self.imu_fps)
                dvl_seq = self.extract_sequence(dvl_data, sample_time, self.dvl_fps)
                gt_seq = self.extract_sequence(gt_data, sample_time, self.gt_fps)
                
                if any(x is None for x in [imu1_seq, imu2_seq, imu3_seq, dvl_seq, gt_seq]):
                    continue
                
                expected_imu = int(self.seq_duration * self.imu_fps)
                expected_dvl = int(self.seq_duration * self.dvl_fps)
                expected_gt = int(self.seq_duration * self.gt_fps)
                
                if (abs(len(imu1_seq) - expected_imu) > 1 or abs(len(imu2_seq) - expected_imu) > 1 or
                    abs(len(imu3_seq) - expected_imu) > 1 or abs(len(dvl_seq) - expected_dvl) > 1 or 
                    abs(len(gt_seq) - expected_gt) > 1):
                    continue
                
                imu1_seq = imu1_seq[:expected_imu]
                imu2_seq = imu2_seq[:expected_imu]
                imu3_seq = imu3_seq[:expected_imu]
                dvl_seq = dvl_seq[:expected_dvl]
                gt_seq = gt_seq[:expected_gt]
                
                imu1_seq_ds = self.downsample(imu1_seq, self.imu_downsample)
                imu2_seq_ds = self.downsample(imu2_seq, self.imu_downsample)
                imu3_seq_ds = self.downsample(imu3_seq, self.imu_downsample)
                dvl_seq_ds = dvl_seq
                gt_seq_ds = self.downsample(gt_seq, self.gt_downsample)
                
                expected_len = int(self.seq_duration * self.target_fps)
                
                if (abs(len(imu1_seq_ds) - expected_len) > 1 or abs(len(imu2_seq_ds) - expected_len) > 1 or
                    abs(len(imu3_seq_ds) - expected_len) > 1 or abs(len(dvl_seq_ds) - expected_len) > 1 or 
                    abs(len(gt_seq_ds) - expected_len) > 1):
                    continue
                
                imu1_seq_ds = imu1_seq_ds[:expected_len]
                imu2_seq_ds = imu2_seq_ds[:expected_len]
                imu3_seq_ds = imu3_seq_ds[:expected_len]
                dvl_seq_ds = dvl_seq_ds[:expected_len]
                gt_seq_ds = gt_seq_ds[:expected_len]
                
                if self.train:
                    imu1_seq_ds = self.augment_imu(imu1_seq_ds)
                    imu2_seq_ds = self.augment_imu(imu2_seq_ds)
                    imu3_seq_ds = self.augment_imu(imu3_seq_ds)
                    dvl_seq_ds = self.augment_dvl(dvl_seq_ds)
                
                imu1_norm = self.normalize(imu1_seq_ds[:, :-1], self.imu1_mean, self.imu1_std)
                imu2_norm = self.normalize(imu2_seq_ds[:, :-1], self.imu2_mean, self.imu2_std)
                imu3_norm = self.normalize(imu3_seq_ds[:, :-1], self.imu3_mean, self.imu3_std)
                dvl_vel = dvl_seq_ds[:, :3]
                dvl_norm = self.normalize(dvl_vel, self.dvl_mean, self.dvl_std)
                gt_vel = gt_seq_ds[:, :3]
                gt_vel_norm = self.normalize(gt_vel, self.gt_mean[:3], self.gt_std[:3])
                
                sample = {
                    'imu1_data': imu1_norm.astype(np.float32),
                    'imu2_data': imu2_norm.astype(np.float32),
                    'imu3_data': imu3_norm.astype(np.float32),
                    'dvl_data': dvl_norm.astype(np.float32),
                    'ground_truth': gt_vel_norm.astype(np.float32),
                    'timestamp': sample_time
                }
                samples.append(sample)
                
            except Exception:
                continue
        
        return samples
    
    def extract_sequence(self, data, center_time, fps):
        seq_len = int(self.seq_duration * fps)
        
        time_diffs = np.abs(data[:, -1] - center_time)
        center_idx = np.argmin(time_diffs)
        
        half_seq = seq_len // 2
        start_idx = center_idx - half_seq
        end_idx = start_idx + seq_len
        
        if start_idx < 0:
            start_idx = 0
            end_idx = seq_len
        elif end_idx > len(data):
            end_idx = len(data)
            start_idx = max(0, end_idx - seq_len)
        
        if end_idx - start_idx != seq_len:
            return None
        
        return data[start_idx:end_idx]
    
    def downsample(self, data, factor):
        if factor <= 1:
            return data
        return data[::factor]
    
    def normalize(self, data, mean, std):
        mean = np.array(mean)
        std = np.array(std)
        std_safe = np.where(std == 0, 1.0, std)
        return (data - mean) / std_safe
    
    def augment_imu(self, imu_seq):
        if not self.train:
            return imu_seq
        
        augmented = imu_seq.copy()
        
        if 'imu_noise_std' in self.noise_config:
            noise = np.random.normal(0, self.noise_config['imu_noise_std'], 
                                    (len(augmented), 6))
            augmented[:, :6] += noise
        
        if 'imu_bias_drift' in self.noise_config:
            bias = np.random.normal(0, self.noise_config['imu_bias_drift'], 6)
            augmented[:, :6] += bias
        
        return augmented
    
    def augment_dvl(self, dvl_seq):
        if not self.train:
            return dvl_seq
        
        augmented = dvl_seq.copy()
        
        if 'dvl_noise_std' in self.noise_config:
            noise = np.random.normal(0, self.noise_config['dvl_noise_std'], 
                                    (len(augmented), 3))
            augmented[:, :3] += noise
        
        return augmented
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]