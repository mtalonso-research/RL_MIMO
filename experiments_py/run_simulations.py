import argparse
import torch
from math import comb
import sys
import os
import warnings
warnings.filterwarnings('ignore')
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from src.rl_environment import defining_environments
from src.rl_policy import UnifiedPolicy
from src.cortical_estimator import CORTICAL
from src.ba_estimator import MI_ESTIMATOR
from src.simulation_runner import runing_sims

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
    parser = argparse.ArgumentParser(description="Run policy inference simulations")
    parser.add_argument('--dimension', type=int, default=2)
    parser.add_argument('--num_thresholds', type=int, default=4)
    parser.add_argument('--box_param', type=float, default=1.5)
    parser.add_argument('--mi_estimator', type=str, default='BA', choices=['BA', 'CORTICAL'])
    parser.add_argument('--channel_type', type=str, default='identity-csi')
    parser.add_argument('--max_steps', type=int, default=1000)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--ln_steps', type=int, default=50)
    parser.add_argument('--pt_steps', type=int, default=50)
    parser.add_argument('--num_sims', type=int, default=10)
    parser.add_argument('--run_id', type=str, default='run10')
    parser.add_argument('--model_path', type=str, default='./models/policy_models')
    parser.add_argument('--cortical_path', type=str, default='./models/cortical_models')
    args = parser.parse_args()

    box_param = float(args.box_param)
    alphabet_size = int(sum(comb(args.num_thresholds, k) for k in range(args.dimension + 1)))
    norm_patience = (args.patience, 1000)
    thrsh = init_thresholds(args.dimension, args.num_thresholds)

    if args.mi_estimator == 'BA':
        mi_est = MI_ESTIMATOR(args.dimension, box_param, args.channel_type, 10000)
    else:
        mi_est = CORTICAL((512, args.dimension, alphabet_size, args.num_thresholds, 10),
                          alphas=[1.0, 0.5, 0.1, 0.01], lambda_entropy=0.3,
                          cost_coef=10.0, box_param=box_param)
        mi_est.load_models(args.cortical_path)

    policy_path = f"{args.model_path}/unified_policy_{args.mi_estimator}-{args.dimension}D-{args.num_thresholds}-{args.run_id}.pth"
    print(f"Loading policy from {policy_path}")
    policy = torch.load(policy_path)

    for sim_count in range(args.num_sims):
        print(f"▶️ Running simulation {sim_count + 1}/{args.num_sims}")
        runing_sims(args.dimension, args.num_thresholds, alphabet_size, box_param,
                    args.max_steps, args.patience, norm_patience,
                    args.ln_steps, args.pt_steps, policy, mi_est,
                    args.run_id, sim_count, thrsh)

if __name__ == '__main__':
    main()
