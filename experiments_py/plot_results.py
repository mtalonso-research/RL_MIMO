import argparse
import sys
import sys
import os
import warnings
warnings.filterwarnings('ignore')
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from src.simulation_runner import plot_snr_vs_mi_with_shaded_std
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def main():

    def comma_str_list(arg):
        return [x.strip() for x in arg.split(',')]

    def comma_int_list(arg):
        return list(map(int, arg.split(',')))

    parser = argparse.ArgumentParser(description="Plot MI vs SNR")
    parser.add_argument('--sim_folder', type=str, default='/simulation_results')
    parser.add_argument('--mi_estimators', type=comma_str_list)
    parser.add_argument('--dimensions', type=comma_int_list)
    parser.add_argument('--num_thresholds_list', type=comma_int_list)
    parser.add_argument('--run_ids', type=comma_str_list)
    parser.add_argument('--channel_types', type=comma_str_list)
    parser.add_argument('--sim_counts', type=comma_int_list)
    parser.add_argument('--num_std', type=int, default=1)

    args = parser.parse_args()

    style_mapping = {
        'color': 'mi_estimator',
        'marker': 'num_thresholds',
        'linestyle': 'channel_type'
    }

    plot_snr_vs_mi_with_shaded_std(
        sim_folder=project_root + args.sim_folder,
        mi_estimators=args.mi_estimators,
        dimensions=args.dimensions,
        num_thresholds_list=args.num_thresholds_list,
        run_ids=args.run_ids,
        channel_types=args.channel_types,
        sim_counts=args.sim_counts,
        num_std=args.num_std,
        style_mapping=style_mapping
    )

if __name__ == '__main__':
    main()
