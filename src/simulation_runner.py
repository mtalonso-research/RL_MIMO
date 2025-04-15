import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import torch.distributions as dist
from torch.distributions import Normal
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import warnings
from math import comb
import torch.nn as nn
import copy
import random
import plotly.graph_objects as go
from shapely.geometry import Polygon, LineString, Point, box
from shapely.ops import polygonize, unary_union
import plotly.express as px
import itertools
from itertools import cycle
import time
import sys
import os
warnings.filterwarnings('ignore')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from src.rl_environment import RLEnvironment, defining_environments
from src.rl_policy import UnifiedPolicy, Policy
from src.cortical_estimator import CORTICAL
from src.ba_estimator import MI_ESTIMATOR

def simulation_main_runner(dimension,num_thresholds,alphabet_size,box_param,max_steps,patience,norm_patience,ln_steps,pt_steps,policy,mi_est,thrsh):
    mi_dict = {}
    time_dict = {}
    steps_dict = {}
    best_state_dict = {}  
    envs, _,_ = defining_environments(dimension,num_thresholds,alphabet_size,box_param,(-10.0,40.0),50,max_steps,patience,mi_est,norm_patience,True,False,thrsh)

    for env in tqdm(envs):
        snr_val = env.state[0]
        start_time = time.time()
        best_reward = float('-inf')  
        best_mi = None
        best_state = None  

        done_flag = False
        while env.done == False:
            state_tensor = torch.tensor(env.state, dtype=torch.float32).unsqueeze(0).to(device)
            
            for i in range(ln_steps):
                if done_flag: break
                action_points = torch.zeros(alphabet_size*dimension).unsqueeze(0).to(device)
                _, action_lines = policy.select_action(state_tensor, 'threshold')
                action = torch.cat([action_lines.view(-1), action_points.view(-1)]).view(-1)
                env.state, reward, done_flag = env.step_line(action, False)
                if reward > best_reward:
                    best_reward = reward
                    best_mi = env.mi_output[0]
                    best_state = env.state.clone()  

            for j in range(pt_steps):
                if done_flag: break
                action_lines = torch.zeros(num_thresholds * (dimension + 1)).unsqueeze(0).to(device)
                _, action_points = policy.select_action(state_tensor, 'point')
                action = torch.cat([action_lines.view(-1), action_points.view(-1)]).view(-1)
                env.state, reward, done_flag = env.step_point(action)
                if reward > best_reward:
                    best_reward = reward
                    best_mi = env.mi_output[0]
                    best_state = env.state.clone()  
        
        time_dict[snr_val] = time.time() - start_time
        steps_dict[snr_val] = env.step_num
        mi_dict[snr_val] = best_mi
        best_state_dict[snr_val] = best_state.detach().cpu().numpy()

    return mi_dict, time_dict, steps_dict, best_state_dict

def true_rate_calculation(best_state_dict,dimension,num_thresholds,box_param):
    mi_dict = {}
    mi_est = MI_ESTIMATOR(dimension, box_param, 'identity-csi', 100000)

    for snr,state in tqdm(best_state_dict.items()):
        _,mi_outputs = mi_est(torch.tensor(state),num_thresholds)
        mi_dict[snr] = mi_outputs[0]
    return mi_dict

def runing_sims(dimension,num_thresholds,alphabet_size,box_param,max_steps,patience,norm_patience,ln_steps,pt_steps,policy,mi_est,run_id,sim_count,thrsh=None):
    mi_dict, time_dict, steps_dict, state_dict = simulation_main_runner(dimension,num_thresholds,alphabet_size,box_param,max_steps,patience,norm_patience,ln_steps,pt_steps,policy,mi_est,thrsh)
    mi_dict = true_rate_calculation(state_dict,dimension,num_thresholds,box_param)
    
    if type(mi_est).__name__ == 'MI_ESTIMATOR': mi_estimator = 'BA'
    else: mi_estimator = 'CORTICAL'
    channel_type = mi_est.channel.channel_type
    
    if channel_type == 'identity-csi':
        np.savez(f"/simulation_results/{dimension}D/T{num_thresholds}/{mi_estimator}/{mi_estimator}-{dimension}D-{num_thresholds}-{run_id}-{channel_type}-{sim_count}.npz", 
                SNR=list(mi_dict.keys()), MI=list(mi_dict.values()), Time=list(time_dict.values()), Steps=list(steps_dict.values()), State=list(state_dict.values()))
    else:
        np.savez(f"/simulation_results/H/T{num_thresholds}/{channel_type}/{mi_estimator}-{dimension}D-{num_thresholds}-{run_id}-{channel_type}-{sim_count}.npz", 
                SNR=list(mi_dict.keys()), MI=list(mi_dict.values()), Time=list(time_dict.values()), Steps=list(steps_dict.values()), State=list(state_dict.values()))  
    
def time_sharing(SNR_dB, Rate):
    Rate = Rate.copy()
    for i in range(1, len(SNR_dB) - 1):
        P1 = 10 ** (SNR_dB[i - 1] / 10)
        P2 = 10 ** (SNR_dB[i] / 10)
        P3 = 10 ** (SNR_dB[i + 1] / 10)

        lam = (P3 - P2) / (P3 - P1)
        R_new = lam * Rate[i - 1] + (1 - lam) * Rate[i + 1]
        Rate[i] = max(Rate[i], R_new)
    Rate[-1] = max(Rate[-2],Rate[-1])

    return Rate

def time_sharing_iterative(SNR_dB, Rate):
    Rate = Rate.copy()
    num_points = len(SNR_dB)

    for i in range(num_points - 1): 
        P1 = 10 ** (SNR_dB[i] / 10)
        for j in range(i + 1, num_points):  
            P2 = 10 ** (SNR_dB[j] / 10)
            lam = (P2 - P1) / (10 ** (SNR_dB[j] / 10) - 10 ** (SNR_dB[i] / 10))  
            R_interp = lam * Rate[i] + (1 - lam) * Rate[j]  
            Rate[j] = max(Rate[j], R_interp)
    return Rate

def plot_snr_vs_mi_with_shaded_std(sim_folder, mi_estimators, dimensions, num_thresholds_list,
                                    run_ids, channel_types, sim_counts, num_std,
                                    style_mapping,
                                    suffix={'BA':'','CORTICAL':'','CORTICAL_BA':'','BruteForce':'','PSK':'','QAM':''}):

    plt.figure(figsize=(7, 6))
    missing_files = []

    all_combos = []
    for mi_estimator in mi_estimators:
        for dimension in dimensions:
            for num_thresholds in num_thresholds_list:
                for run_id in run_ids:
                    for channel_type in channel_types:
                        all_combos.append({
                            'mi_estimator': mi_estimator,
                            'dimension': dimension,
                            'num_thresholds': num_thresholds,
                            'run_id': run_id,
                            'channel_type': channel_type,
                        })

    def get_unique_values(mapping_key):
        if mapping_key is None:
            return []
        return sorted(set(
            combo[mapping_key] if isinstance(mapping_key, str)
            else tuple(combo[k] for k in mapping_key)
            for combo in all_combos
        ))

    color_keys = get_unique_values(style_mapping.get('color'))
    color_cycle = plt.cm.viridis(np.linspace(0, 1, len(color_keys))) if color_keys else []
    color_map = {k: c for k, c in zip(color_keys, color_cycle)}

    marker_cycle = cycle(['o', 'X', '*', 'p', 's', 'D', '^', 'v', '>', '<'])
    linestyle_cycle = cycle(['-', '--', '-.', ':'])
    marker_map = {}
    linestyle_map = {}

    def get_style_value(feature, key_dict):
        mapping = style_mapping.get(feature)
        if mapping is None:
            return None
        if isinstance(mapping, str):
            style_key = key_dict[mapping]
        elif isinstance(mapping, (tuple, list)):
            style_key = tuple(key_dict[k] for k in mapping)
        else:
            raise ValueError(f"Unsupported mapping for {feature}: {mapping}")

        if feature == 'color':
            return color_map.get(style_key, 'black')
        elif feature == 'marker':
            if style_key not in marker_map:
                marker_map[style_key] = next(marker_cycle)
            return marker_map[style_key]
        elif feature == 'linestyle':
            if style_key not in linestyle_map:
                linestyle_map[style_key] = next(linestyle_cycle)
            return linestyle_map[style_key]

    for combo in all_combos:
        mi_estimator = combo['mi_estimator']
        dimension = combo['dimension']
        num_thresholds = combo['num_thresholds']
        run_id = combo['run_id']
        channel_type = combo['channel_type']

        all_snr = []
        all_mi = []
        found_file = False
        run_id_found = False
        plotted_sim_counts = 0

        for sim_count in sim_counts:
            filename = f"{sim_folder}/{mi_estimator}-{dimension}D-{num_thresholds}-{run_id}-{channel_type}-{sim_count}{suffix[mi_estimator]}.npz"
            try:
                data = np.load(filename)
                snr = data['SNR']
                mi = data['MI']
                mi_refined = time_sharing_iterative(snr, mi)

                all_snr.append(snr)
                all_mi.append(np.squeeze(mi_refined))
                found_file = True
                run_id_found = True
                plotted_sim_counts += 1
            except:
                if channel_type == 'identity-csi': 
                    filename = f"/{sim_folder}/{dimension}D/T{num_thresholds}/{mi_estimator}/{mi_estimator}-{dimension}D-{num_thresholds}-{run_id}-{channel_type}-{sim_count}{suffix[mi_estimator]}.npz"
                else: 
                    filename = f"/{sim_folder}/H/T{num_thresholds}/{channel_type}/{mi_estimator}-{dimension}D-{num_thresholds}-{run_id}-{channel_type}-{sim_count}{suffix[mi_estimator]}.npz"
                try:
                    data = np.load(filename)
                    snr = data['SNR']
                    mi = data['MI']
                    mi_refined = time_sharing_iterative(snr, mi)

                    all_snr.append(snr)
                    all_mi.append(np.squeeze(mi_refined))
                    found_file = True
                    run_id_found = True
                    plotted_sim_counts += 1
                except FileNotFoundError:
                    continue
                except Exception as e:
                    print(f"Error reading {filename}: {e}")
                    missing_files.append(filename)

        if not found_file and not run_id_found and run_id != '-' and mi_estimator != 'BruteForce':
            missing_files.append(
                f"{mi_estimator}-{dimension}D-{num_thresholds}-{run_id}-{channel_type}-*"
            )

        if all_snr and all_mi:
            snr = all_snr[0]
            all_mi = np.array(all_mi)
            mean_mi = np.mean(all_mi, axis=0)
            std_mi = np.std(all_mi, axis=0)
            mean_mi = time_sharing_iterative(snr, mean_mi)

            color = get_style_value('color', combo)
            if color is None:
                color = 'black'

            marker = get_style_value('marker', combo)
            if marker is None:
                marker = 'o'

            linestyle = get_style_value('linestyle', combo)
            if linestyle is None:
                linestyle = '-'

            if channel_type == 'changing-csi-smooth-0.01': label = fr"{mi_estimator}, $n_t$ = {dimension}, $n_q$ = {num_thresholds}, time-variying-0.01"
            elif channel_type == 'changing-csi-smooth-0.05': label = fr"{mi_estimator}, $n_t$ = {dimension}, $n_q$ = {num_thresholds}, time-variying-0.05"
            elif channel_type != 'identity-csi': label = fr"{mi_estimator}, $n_t$ = {dimension}, $n_q$ = {num_thresholds}, {channel_type}"
            else: label = fr"{mi_estimator}, $n_t$ = {dimension}, $n_q$ = {num_thresholds}"

            plt.plot(snr, mean_mi, label=label,
                     color=color, linestyle=linestyle, marker=marker,
                     markersize=6, linewidth=1.5, markeredgecolor='black')

            if len(all_mi) > 1:
                try:
                    plt.fill_between(snr, mean_mi - num_std * std_mi, mean_mi + num_std * std_mi,
                                     color=color, alpha=0.4)
                except Exception as e:
                    print(f"Could not shade std for {label}: {e}")

    plt.xlabel("SNR (dB)")
    plt.ylabel("Mutual Information (MI)")
    plt.grid(True)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), fontsize='small', loc='best')
    plt.tight_layout()
    plt.show()

def train_cases(dimension,num_thresholds,alphabet_size,box_param,num_envs,max_steps,patience,num_episodes,norm_patience,lr,mi_est,policy,run_id):
    loss_history = {'lowSNR-thrsh':[],'midSNR-thrsh':[],'highSNR-thrsh':[],
                    'lowSNR-qtpts':[],'midSNR-qtpts':[],'highSNR-qtpts':[]}
    
    print('---------Training Low SNR---------')
    envs,_,_ = defining_environments(dimension,num_thresholds,alphabet_size,box_param,(-10.0,0.0),num_envs,max_steps,patience,mi_est,norm_patience,True,False)
    loss_history['lowSNR-thrsh'] = policy.train_policies(num_episodes,num_envs,envs,(-10.0,0.0),'threshold',lr)
    loss_history['lowSNR-qtpts'] = policy.train_policies(num_episodes,num_envs,envs,(-10.0,0.0),'point',lr)
    print('----------------------------------')

    print('---------Training Mid SNR---------')
    envs,_,_ = defining_environments(dimension,num_thresholds,alphabet_size,box_param,(0.0,10.0),num_envs,max_steps,patience,mi_est,norm_patience,True,False)
    loss_history['midSNR-thrsh'] = policy.train_policies(num_episodes,num_envs,envs,(0.0,10.0),'threshold',lr)
    loss_history['midSNR-qtpts'] = policy.train_policies(num_episodes,num_envs,envs,(0.0,10.0),'point',lr)
    print('----------------------------------')
    
    print('---------Training High SNR---------')
    envs,_,_ = defining_environments(dimension,num_thresholds,alphabet_size,box_param,(10.0,20.0),num_envs,max_steps,patience,mi_est,norm_patience,True,False)
    loss_history['highSNR-thrsh'] = policy.train_policies(num_episodes,num_envs,envs,(10.0,20.0),'threshold',lr)
    loss_history['highSNR-qtpts'] = policy.train_policies(num_episodes,num_envs,envs,(10.0,20.0),'point',lr)
    print('----------------------------------')

    if type(mi_est).__name__ == 'MI_ESTIMATOR': mi_estimator = 'BA'
    else: mi_estimator = 'CORTICAL'

    torch.save(policy, f"./models/policy_models/unified_policy_{mi_estimator}-{dimension}D-{num_thresholds}-{run_id}.pth")
    print('Saved Trained Policy Model')
    return policy