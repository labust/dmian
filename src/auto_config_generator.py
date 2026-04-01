#!/usr/bin/env python3
"""
Auto config generator
"""

import numpy as np
import yaml
from pathlib import Path
import argparse


def compute_statistics_multi_trajectory(data_dirs):
    """
    Compute statistics across multiple trajectories for 3 IMUs
    
    Args:
        data_dirs: List of data directories
    
    Returns:
        Dictionary with statistics
    """
    print(f"\nComputing statistics across {len(data_dirs)} trajectories...")
    
    imu1_data_list = []
    imu2_data_list = []
    imu3_data_list = []
    dvl_data_list = []
    gt_data_list = []
    
    for data_dir in data_dirs:
        data_path = Path(data_dir)
        
        for imu_name, imu_list in [('imu1', imu1_data_list), 
                                    ('imu2', imu2_data_list), 
                                    ('imu3', imu3_data_list)]:
            imu_path = data_path / f'{imu_name}_data.npy'
            if imu_path.exists():
                data = np.load(imu_path)
                imu_list.append(data[:, :-1])
                print(f"  {data_path.name}: {imu_name.upper()} {data.shape}")
        
        dvl_path = data_path / 'dvl_data.npy'
        if dvl_path.exists():
            data = np.load(dvl_path)
            dvl_data_list.append(data[:, :3])
        
        gt_path = data_path / 'ground_truth_data.npy'
        if gt_path.exists():
            data = np.load(gt_path)
            gt_data_list.append(data[:, :3])
    
    stats = {}
    
    for imu_name, imu_list in [('imu1', imu1_data_list), 
                                ('imu2', imu2_data_list), 
                                ('imu3', imu3_data_list)]:
        if imu_list:
            imu_all = np.vstack(imu_list)
            stats[f'imu_{imu_name}'] = {
                'mean': imu_all.mean(axis=0).tolist(),
                'std': imu_all.std(axis=0).tolist()
            }
            print(f"\n{imu_name.upper()} combined statistics:")
            print(f"  Total samples: {len(imu_all)}")
    
    if dvl_data_list:
        dvl_all = np.vstack(dvl_data_list)
        stats['dvl'] = {
            'mean': dvl_all.mean(axis=0).tolist(),
            'std': dvl_all.std(axis=0).tolist()
        }
        print(f"\nDVL combined statistics:")
        print(f"  Total samples: {len(dvl_all)}")
    
    if gt_data_list:
        gt_all = np.vstack(gt_data_list)
        stats['ground_truth'] = {
            'mean': gt_all.mean(axis=0).tolist(),
            'std': gt_all.std(axis=0).tolist()
        }
    
    return stats


def analyze_and_create_config(train_dirs, val_dirs, test_dirs, output_config='config.yaml'):
    """
    Analyze data and create training configuration with multiple trajectories
    
    Args:
        train_dirs: List of training data directories
        val_dirs: List of validation data directories
        test_dirs: List of test data directories
        output_config: Output config file path
    """
    print("="*70)
    print("Multi-Trajectory Config Generator (3-IMU)")
    print("="*70)
    print(f"Using IMUs: imu1, imu2, imu3")
    print(f"\nTraining trajectories ({len(train_dirs)}):")
    for d in train_dirs:
        print(f"  - {Path(d).name}")
    print(f"\nValidation trajectories ({len(val_dirs)}):")
    for d in val_dirs:
        print(f"  - {Path(d).name}")
    print(f"\nTest trajectories ({len(test_dirs)}):")
    for d in test_dirs:
        print(f"  - {Path(d).name}")
    
    required_files = ['imu1_data.npy', 'imu2_data.npy', 'imu3_data.npy',
                     'dvl_data.npy', 'ground_truth_data.npy']
    
    all_dirs = train_dirs + val_dirs + test_dirs
    for data_dir in all_dirs:
        data_path = Path(data_dir)
        if not data_path.exists():
            print(f"Error: Directory doesn't exist: {data_path}")
            return False
        
        missing = [f for f in required_files if not (data_path / f).exists()]
        if missing:
            print(f"Warning: {data_path.name} missing files: {missing}")
    
    # Compute statistics from training data only
    stats = compute_statistics_multi_trajectory(train_dirs)
    
    first_train = Path(train_dirs[0])
    imu1_data = np.load(first_train / 'imu1_data.npy')
    dvl_data = np.load(first_train / 'dvl_data.npy')
    gt_data = np.load(first_train / 'ground_truth_data.npy')
    
    imu_duration = imu1_data[-1, -1] - imu1_data[0, -1]
    dvl_duration = dvl_data[-1, -1] - dvl_data[0, -1]
    gt_duration = gt_data[-1, -1] - gt_data[0, -1]
    
    imu_freq = len(imu1_data) / imu_duration
    dvl_freq = len(dvl_data) / dvl_duration
    gt_freq = len(gt_data) / gt_duration
    
    print(f"\nFrequencies (from {first_train.name}):")
    print(f"  IMU: {imu_freq:.1f} Hz")
    print(f"  DVL: {dvl_freq:.1f} Hz")
    print(f"  GT: {gt_freq:.1f} Hz")
    
    total_duration = 0
    for train_dir in train_dirs:
        train_path = Path(train_dir)
        imu_data = np.load(train_path / 'imu1_data.npy')
        duration = imu_data[-1, -1] - imu_data[0, -1]
        total_duration += duration
    
    print(f"\nTotal training duration: {total_duration:.1f}s ({total_duration/60:.1f} min)")
    
    config = {
        'network': {
            'num_models': 3,
            'hidden_size': 64,
            'num_lstm_layers': 2,
            'dropout': 0.2
        },
        'training': {
            'batch_size': {
                'train': 32,
                'test': 16
            },
            'learning_rate': 0.001,
            'max_epochs': 100,
            'early_stopping_patience': 20,
            'weight_decay': 0.0001,
            'gnll_start_epoch': 10,
            'scheduler': {
                'factor': 0.5,
                'patience': 10
            },
            'loss_weights': {
                'mse_weight': 1.0,
                'gnll_weight': 1.0
            },
            'save_dir': '../models'
        },
        'dataset': {
            'train_dirs': [str(Path(d)) for d in train_dirs],
            'val_dirs': [str(Path(d)) for d in val_dirs],
            'test_dirs': [str(Path(d)) for d in test_dirs],
            'seq_duration': 4.0,
            'seq_sample_density': 10.0,
            'sensor_fps': {
                'imu': int(round(imu_freq)),
                'dvl': int(round(dvl_freq)),
                'ground_truth': int(round(gt_freq))
            },
            'sensor_stats': stats,
            'noise': {
                'imu_noise_std': 0.01,
                'imu_bias_drift': 0.005,
                'dvl_noise_std': 0.02
            }
        }
    }
    
    output_path = Path(output_config)
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"\n{'='*70}")
    print(f"Configuration saved to: {output_path}")
    print(f"{'='*70}")
    print(f"\nNext steps:")
    print(f"  1. Train: python train_script.py --config {output_config}")
    print(f"  2. Test: python test_script.py --config {output_config} --data-dir <test_traj>")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Auto-generate 3-IMU training configuration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python auto_config_generator.py \\
    --data-root ../../dmian-dataset \\
    --train-dirs train/trajectory_06 train/trajectory_07 train/trajectory_08 train/trajectory_09 train/trajectory_10 train/trajectory_11 \\
    --val-dirs val/trajectory_12 val/trajectory_13 \\
    --test-dirs test/trajectory_01 test/trajectory_02 test/trajectory_03 test/trajectory_04 test/trajectory_05 \\
    --output ./config/config.yaml
        """
    )
    parser.add_argument('--data-root', type=str, 
                       default='../../dmian-dataset',
                       help='Root directory containing all trajectories')
    parser.add_argument('--train-dirs', type=str, nargs='+', required=True,
                       help='Training data directories')
    parser.add_argument('--val-dirs', type=str, nargs='+', required=True,
                       help='Validation data directories')
    parser.add_argument('--test-dirs', type=str, nargs='+', required=True,
                       help='Test data directories')
    parser.add_argument('--output', type=str, default='./config/config.yaml',
                       help='Output config file')
    
    args = parser.parse_args()
    
    data_root = Path(args.data_root)

    train_dirs = [str(data_root / d) for d in args.train_dirs]
    val_dirs = [str(data_root / d) for d in args.val_dirs]
    test_dirs = [str(data_root / d) for d in args.test_dirs]
    
    success = analyze_and_create_config(
        train_dirs, 
        val_dirs, 
        test_dirs, 
        args.output
    )
    
    if not success:
        print("\nConfig generation failed!")
        exit(1)


if __name__ == '__main__':
    main()