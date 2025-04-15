import argparse
import torch
from math import comb
import warnings
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from src.cortical_estimator import CORTICAL
from src.utils import set_seed
warnings.filterwarnings('ignore')

def get_box_param(dimension, num_thresholds):
    if dimension == 2:
        return 1.5
    elif dimension == 1:
        return max(1.5, 1.0 + int(num_thresholds / 2) * 0.5)

def main():
    parser = argparse.ArgumentParser(description="Train CORTICAL estimator")
    parser.add_argument('--dimension', type=int, default=1)
    parser.add_argument('--num_thresholds', type=int, default=1)
    parser.add_argument('--sample_size', type=int, default=10)
    parser.add_argument('--num_batches', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--subepochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_path', type=str, default='./models/cortical_models')
    parser.add_argument('--verbose', type=int, default=1, choices=[0,1,2])
    args = parser.parse_args()

    set_seed(args.seed)

    alphabet_size = int(sum(comb(args.num_thresholds, k) for k in range(args.dimension + 1)))
    box_param = get_box_param(args.dimension, args.num_thresholds)

    params = (args.sample_size, args.dimension, alphabet_size, args.num_thresholds, args.num_batches)
    cortical = CORTICAL(params=params, alphas=[1.0, 0.5, 0.1, 0.01],
                        lambda_entropy=0.3, cost_coef=10.0, box_param=box_param)

    # Train across SNR ranges
    ranges = [(-10.0, -2.0), (-2.0, 4.0), (4.0, 10.0), (10.0, 20.0)]
    for r in ranges:
        print(f"\nTraining on range {r}")
        cortical.train(r, epochs=args.epochs, subepochs=args.subepochs, verbose=args.verbose)

    cortical.save_models(args.save_path)
    print(f"\n✅ Models saved to {args.save_path}")

if __name__ == '__main__':
    main()
