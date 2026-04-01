#!/usr/bin/env python3

import argparse
import sys
import time
import yaml
from pathlib import Path

from networks.train_script import Trainer
from networks.test_script import test_model


def main():
    parser = argparse.ArgumentParser(description='Main script for training and testing DMIAN model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--model-dir', type=str, default='./models',help='Model directory')
    parser.add_argument('--test-dirs', type=str, nargs='+', required=True, help='Test data directories')
    parser.add_argument('--results-dir', type=str, default='./results', help='Results data directory')
    parser.add_argument('--skip-training', action='store_true', help='Skip training and only run testing')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    save_dir = args.model_dir or config['training']['save_dir']
    best_model_dir = str(Path(save_dir) / 'best')

    if not args.skip_training:
        print("=" * 60)
        print("TRAINING")
        print("=" * 60)
        t0 = time.time()

        if args.model_dir:
            config['training']['save_dir'] = save_dir
            import tempfile, os
            tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
            yaml.dump(config, tmp)
            tmp.close()
            config_path = tmp.name
        else:
            config_path = args.config

        trainer = Trainer(config_path)
        trainer.train()

        if args.model_dir:
            os.unlink(config_path)

        print(f"\nTraining finished in {(time.time() - t0) / 60:.1f} min")
    else:
        print("Skipping training.")

    print("\n" + "=" * 60)
    print("TESTING")
    print("=" * 60)

    for test_dir in args.test_dirs:
        print(f"\nTesting on: {Path(test_dir).name}")
        print("-" * 40)
        try:
            test_model(args.config, best_model_dir, test_dir, args.results_dir)
        except Exception as e:
            print(f"  Failed: {e}")


if __name__ == '__main__':
    main()