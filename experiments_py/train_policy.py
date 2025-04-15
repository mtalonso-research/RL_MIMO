import argparse
import warnings
import os
import sys
from math import comb
import torch
warnings.filterwarnings('ignore')
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from src.rl_environment import defining_environments
from src.rl_policy import UnifiedPolicy
from src.cortical_estimator import CORTICAL
from src.ba_estimator import MI_ESTIMATOR
from src.utils import set_seed
from src.simulation_runner import train_cases

def init_thresholds(dimension, num_thresholds):
    if dimension == 1 and num_thresholds == 1:
        return torch.tensor([1.0, 0.0])
    elif dimension == 1 and num_thresholds == 2:
        return torch.tensor([1.0, -0.5, 1.0, 0.5])
    elif dimension == 1 and num_thresholds == 3:
        return torch.tensor([1.0, -0.5, 1.0, 0.0, 1.0, 0.5])
    elif dimension == 2 and num_thresholds == 2:
        return torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    elif dimension == 2 and num_thresholds == 3:
        return torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, -0.25, 1.0, 0.0, 0.25])
    elif dimension == 2 and num_thresholds == 4:
        return torch.tensor([1.0, 0.0, -0.25, 1.0, 0.0, 0.25, 0.0, 1.0, 0.25, 0.0, 1.0, -0.25])
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dimension', type=int, default=2)
    parser.add_argument('--num_thresholds', type=int, default=4)
    parser.add_argument('--mi_estimator', type=str, choices=['BA', 'CORTICAL'], default='BA')
    parser.add_argument('--box_param', type=float, default=1.5)
    parser.add_argument('--num_envs', type=int, default=10)
    parser.add_argument('--max_steps', type=int, default=2000)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--num_episodes', type=int, default=5)
    parser.add_argument('--run_id', type=str, default='run0')
    parser.add_argument('--kl_coeff', type=float, default=0.3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    alphabet_size = int(sum(comb(args.num_thresholds, k) for k in range(args.dimension + 1)))
    norm_patience = (2000, 2000)
    thrsh = init_thresholds(args.dimension, args.num_thresholds)

    # Initialize MI Estimator
    if args.mi_estimator == 'BA':
        mi_est = MI_ESTIMATOR(args.dimension, args.box_param, 'identity-csi', 10000)
    else:
        mi_est = CORTICAL((512, args.dimension, alphabet_size, args.num_thresholds, args.num_envs),
                          alphas=[1.0, 0.5, 0.1, 0.01],
                          lambda_entropy=0.3, cost_coef=10.0, box_param=args.box_param)
        mi_est.load_models('./models/cortical_models/')

    # Create Policy
    policy = UnifiedPolicy(args.dimension, alphabet_size, args.num_thresholds, args.box_param,
                           args.kl_coeff, mi_est, policy_scale=3)

    # Define environments
    envs, thrsh, qtpts = defining_environments(
        args.dimension, args.num_thresholds, alphabet_size, args.box_param,
        (-10.0, 40.0), args.num_envs, args.max_steps, args.patience, mi_est,
        norm_patience, True, False, thrsh
    )

    # Train policy
    policy = train_cases(args.dimension, args.num_thresholds, alphabet_size, args.box_param,
                         args.num_envs, args.max_steps, args.patience, args.num_episodes,
                         norm_patience, args.lr, mi_est, policy, args.run_id)

if __name__ == '__main__':
    main()
