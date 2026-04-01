#!/usr/bin/env python3
"""
Test script
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
import argparse
import pandas as pd

from .networks import EnsemblePredictor

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def normalize(data, mean, std):
    mean = np.array(mean)
    std = np.array(std)
    std_safe = np.where(std == 0, 1.0, std)
    return (data - mean) / std_safe


def denormalize(data, mean, std):
    mean = np.array(mean)
    std = np.array(std)
    std_safe = np.where(std == 0, 1.0, std)
    return data * std_safe + mean


def extract_sequence(data, center_time, fps, duration):
    seq_len = int(duration * fps)
    
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


def downsample(data, factor):
    if factor <= 1:
        return data
    return data[::factor]


def test_model(config_path, model_dir, data_dir, results_dir=None):
    config = load_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    net_cfg = config['network']
    model = EnsemblePredictor(
        num_models=net_cfg['num_models'],
        hidden_size=net_cfg['hidden_size'],
        num_lstm_layers=net_cfg['num_lstm_layers'],
        dropout=net_cfg['dropout']
    ).to(device)
    
    model_path = Path(model_dir)
    models_loaded = 0
    for i in range(net_cfg['num_models']):
        model_file = model_path / f'model_{i}.pt'
        if model_file.exists():
            model.models[i].load_state_dict(torch.load(model_file, map_location=device))
            models_loaded += 1
    
    print(f"Loaded {models_loaded}/{net_cfg['num_models']} models")
    print(f"Using 3 IMUs: imu1, imu2, imu3")
    model.eval()
    
    data_path = Path(data_dir)
    
    imu1_data = np.load(data_path / 'imu1_data.npy')
    imu2_data = np.load(data_path / 'imu2_data.npy')
    imu3_data = np.load(data_path / 'imu3_data.npy')
    dvl_data = np.load(data_path / 'dvl_data.npy')
    gt_data = np.load(data_path / 'ground_truth_data.npy')
    
    print(f"\nTest data:")
    print(f"  IMU1: {imu1_data.shape}")
    print(f"  IMU2: {imu2_data.shape}")
    print(f"  IMU3: {imu3_data.shape}")
    print(f"  DVL: {dvl_data.shape}")
    print(f"  GT: {gt_data.shape}")
    
    ds_cfg = config['dataset']
    
    imu1_mean = np.array(ds_cfg['sensor_stats']['imu_imu1']['mean'])
    imu1_std = np.array(ds_cfg['sensor_stats']['imu_imu1']['std'])
    imu2_mean = np.array(ds_cfg['sensor_stats']['imu_imu2']['mean'])
    imu2_std = np.array(ds_cfg['sensor_stats']['imu_imu2']['std'])
    imu3_mean = np.array(ds_cfg['sensor_stats']['imu_imu3']['mean'])
    imu3_std = np.array(ds_cfg['sensor_stats']['imu_imu3']['std'])
    
    dvl_mean = np.array(ds_cfg['sensor_stats']['dvl']['mean'])
    dvl_std = np.array(ds_cfg['sensor_stats']['dvl']['std'])
    gt_mean = np.array(ds_cfg['sensor_stats']['ground_truth']['mean'])
    gt_std = np.array(ds_cfg['sensor_stats']['ground_truth']['std'])
    
    seq_duration = ds_cfg['seq_duration']
    imu_fps = ds_cfg['sensor_fps']['imu']
    dvl_fps = ds_cfg['sensor_fps']['dvl']
    gt_fps = ds_cfg['sensor_fps']['ground_truth']
    target_fps = dvl_fps
    
    imu_downsample = max(1, imu_fps // target_fps)
    gt_downsample = max(1, gt_fps // target_fps)
    seq_len = int(seq_duration * target_fps)
    
    print(f"\nSequence config:")
    print(f"  Duration: {seq_duration}s at {target_fps}Hz = {seq_len} steps")
    print(f"  IMU downsample: ÷{imu_downsample}, GT downsample: ÷{gt_downsample}")
    
    predictions = []
    ground_truths = []
    times = []
    uncertainties = []
    
    start_time = max(imu1_data[0, -1], imu2_data[0, -1], imu3_data[0, -1],
                    dvl_data[0, -1], gt_data[0, -1]) + seq_duration
    end_time = min(imu1_data[-1, -1], imu2_data[-1, -1], imu3_data[-1, -1],
                   dvl_data[-1, -1], gt_data[-1, -1]) - seq_duration
    
    test_times = np.arange(start_time, end_time, 1.0 / target_fps)
    print(f"\nTesting at {len(test_times)} time points ({target_fps} Hz)")
    
    with torch.no_grad():
        for test_time in test_times:
            try:
                imu1_seq = extract_sequence(imu1_data, test_time, imu_fps, seq_duration)
                imu2_seq = extract_sequence(imu2_data, test_time, imu_fps, seq_duration)
                imu3_seq = extract_sequence(imu3_data, test_time, imu_fps, seq_duration)
                dvl_seq = extract_sequence(dvl_data, test_time, dvl_fps, seq_duration)
                gt_seq = extract_sequence(gt_data, test_time, gt_fps, seq_duration)
                
                if any(x is None for x in [imu1_seq, imu2_seq, imu3_seq, dvl_seq, gt_seq]):
                    continue
                
                expected_imu = int(seq_duration * imu_fps)
                if (len(imu1_seq) != expected_imu or len(imu2_seq) != expected_imu or 
                    len(imu3_seq) != expected_imu):
                    continue
                if len(dvl_seq) != int(seq_duration * dvl_fps) or len(gt_seq) != int(seq_duration * gt_fps):
                    continue
                
                imu1_seq_ds = downsample(imu1_seq, imu_downsample)
                imu2_seq_ds = downsample(imu2_seq, imu_downsample)
                imu3_seq_ds = downsample(imu3_seq, imu_downsample)
                dvl_seq_ds = dvl_seq
                gt_seq_ds = downsample(gt_seq, gt_downsample)
                
                if (len(imu1_seq_ds) != seq_len or len(imu2_seq_ds) != seq_len or 
                    len(imu3_seq_ds) != seq_len):
                    continue
                if len(dvl_seq_ds) != seq_len or len(gt_seq_ds) != seq_len:
                    continue
                
                imu1_norm = normalize(imu1_seq_ds[:, :-1], imu1_mean, imu1_std)
                imu2_norm = normalize(imu2_seq_ds[:, :-1], imu2_mean, imu2_std)
                imu3_norm = normalize(imu3_seq_ds[:, :-1], imu3_mean, imu3_std)
                dvl_norm = normalize(dvl_seq_ds[:, :3], dvl_mean, dvl_std)
                
                gt_vel = gt_seq_ds[:, :3]
                gt_vel_norm = normalize(gt_vel, gt_mean, gt_std)
                
                imu1_tensor = torch.tensor(imu1_norm, dtype=torch.float32).unsqueeze(0).to(device)
                imu2_tensor = torch.tensor(imu2_norm, dtype=torch.float32).unsqueeze(0).to(device)
                imu3_tensor = torch.tensor(imu3_norm, dtype=torch.float32).unsqueeze(0).to(device)
                dvl_tensor = torch.tensor(dvl_norm, dtype=torch.float32).unsqueeze(0).to(device)
                
                vel_pred, aleatoric, epistemic = model(imu1_tensor, imu2_tensor, imu3_tensor, dvl_tensor)
                
                pred_norm = vel_pred[0, -1, :].cpu().numpy()
                gt_norm_final = gt_vel_norm[-1, :]
                
                pred_denorm = denormalize(pred_norm, gt_mean, gt_std)
                gt_denorm = denormalize(gt_norm_final, gt_mean, gt_std)
                
                predictions.append(pred_denorm)
                ground_truths.append(gt_denorm)
                times.append(test_time)
                
                total_unc = aleatoric[0, -1, :].cpu().numpy() + epistemic[0, -1, :].cpu().numpy()
                uncertainties.append(np.sqrt(total_unc))
                
            except Exception as e:
                print(f"Error at {test_time:.1f}s: {e}")
                continue
    
    if len(predictions) == 0:
        print("No successful predictions!")
        return
    
    predictions = np.array(predictions)
    ground_truths = np.array(ground_truths)
    times = np.array(times)
    uncertainties = np.array(uncertainties)
    
    print(f"\n{'='*60}")
    print(f"Successfully tested on {len(predictions)} samples")
    print(f"{'='*60}")
    
    errors = predictions - ground_truths
    rmse = np.sqrt(np.mean(errors**2, axis=0))

    overall_mae = np.mean(np.abs(errors))
    overall_mse = np.mean(errors**2)
    overall_rmse = np.sqrt(overall_mse)

    print(f"\nResults:")
    print(f"{'='*60}")
    print(f"  MAE:  {overall_mae:.4f} m/s")
    print(f"  MSE:  {overall_mse:.4f} m²/s²")
    print(f"  RMSE: {overall_rmse:.4f} m/s")
    print(f"{'='*60}")

    results_base = Path(results_dir) if results_dir else Path(config['training']['save_dir']).parent / 'results'
    results_dir = results_base / data_path.name
    results_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({
        'time': times,
        'gt_vx': ground_truths[:, 0],
        'gt_vy': ground_truths[:, 1],
        'pred_vx': predictions[:, 0],
        'pred_vy': predictions[:, 1],
        'unc_vx': uncertainties[:, 0],
        'unc_vy': uncertainties[:, 1],
    })
    csv_path = results_dir / 'test_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

    plot_results(times, predictions, ground_truths, uncertainties, rmse, results_dir)

    return predictions, ground_truths, times, uncertainties


def plot_results(times, predictions, ground_truths, uncertainties, rmse, results_dir):
    times_rel = times - times[0]
    errors = predictions - ground_truths

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle('Test Results', fontsize=13)

    colors = {'gt': '#2c7bb6', 'pred': '#d7191c', 'unc': '#fdae61'}

    for i, (ax, label) in enumerate(zip(axes[0, :2], ['X', 'Y'])):
        ax.plot(times_rel, ground_truths[:, i], color=colors['gt'], lw=1.5, label='Ground truth')
        ax.plot(times_rel, predictions[:, i], color=colors['pred'], lw=1.5, ls='--', label='Prediction')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'v{label} (m/s)')
        ax.set_title(f'{label}-velocity  RMSE={rmse[i]:.3f} m/s')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for i, (ax, label) in enumerate(zip(axes[1, :2], ['X', 'Y'])):
        ax.plot(times_rel, uncertainties[:, i], color=colors['unc'], lw=1.2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Std (m/s)')
        ax.set_title(f'{label}-velocity uncertainty (±1σ)')
        ax.grid(True, alpha=0.3)

    ax_scatter = axes[0, 2]
    lim = np.max(np.abs(np.concatenate([ground_truths[:, :2], predictions[:, :2]]))) * 1.05
    ax_scatter.scatter(ground_truths[:, 0], predictions[:, 0], s=8, alpha=0.5, label='X', color=colors['gt'])
    ax_scatter.scatter(ground_truths[:, 1], predictions[:, 1], s=8, alpha=0.5, label='Y', color=colors['pred'])
    ax_scatter.plot([-lim, lim], [-lim, lim], 'k--', lw=1)
    ax_scatter.set_xlim(-lim, lim)
    ax_scatter.set_ylim(-lim, lim)
    ax_scatter.set_xlabel('True velocity (m/s)')
    ax_scatter.set_ylabel('Predicted velocity (m/s)')
    ax_scatter.set_title('Correlation (XY)')
    ax_scatter.legend(fontsize=8)
    ax_scatter.set_aspect('equal')
    ax_scatter.grid(True, alpha=0.3)

    ax_err = axes[1, 2]
    ax_err.hist(errors[:, 0], bins=40, alpha=0.6, color=colors['gt'], label='X', edgecolor='none')
    ax_err.hist(errors[:, 1], bins=40, alpha=0.6, color=colors['pred'], label='Y', edgecolor='none')
    ax_err.set_xlabel('Error (m/s)')
    ax_err.set_ylabel('Count')
    ax_err.set_title('Error distribution (XY)')
    ax_err.legend(fontsize=8)
    ax_err.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = results_dir / 'test_results.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {plot_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Test DMIAN model')
    parser.add_argument('--config', type=str, required=True, help='Config file')
    parser.add_argument('--model-dir', type=str, default='./models/best', help='Model directory')
    parser.add_argument('--data-dir', type=str, required=True, help='Test data directory')
    parser.add_argument('--results-dir', type=str, default='../../results', help='Results data directory')

    args = parser.parse_args()

    test_model(args.config, args.model_dir, args.data_dir, args.results_dir)


if __name__ == '__main__':
    main()