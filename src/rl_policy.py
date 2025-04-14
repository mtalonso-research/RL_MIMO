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
import torch.nn.functional as F
import copy
import random
import plotly.graph_objects as go
from shapely.geometry import Polygon, LineString, Point, box
from shapely.ops import polygonize, unary_union
import plotly.express as px
import itertools
from shapely.geometry import Polygon, LineString, Point, box
warnings.filterwarnings('ignore')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Policy(nn.Module):
    def __init__(self, s_size, o_size, h_scale=1):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(s_size, h_scale*64)
        self.fc2 = nn.Linear(h_scale*64, h_scale*64*2)
        self.fc3_mean = nn.Linear(h_scale*64*2, o_size)  
        self.fc3_log_std = nn.Linear(h_scale*64*2, o_size) 
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, state):
        out = self.relu(self.fc1(state))
        out = self.relu(self.fc2(out))
        log_std = self.fc3_log_std(out)
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        mean = self.tanh(self.fc3_mean(out).clamp(-10, 10))
        return mean, std
    
class UnifiedPolicy(nn.Module):
    def __init__(self,dimension,alphabet_size,num_thresholds,box_param,kl_coeff,mi_estimator,policy_scale):
        super(UnifiedPolicy, self).__init__()
        input_size = dimension * alphabet_size + (dimension + 1) * num_thresholds + 1
        self.alphabet_size = alphabet_size
        self.num_thresholds = num_thresholds
        self.dimension = dimension
        self.box_param = box_param
        self.mi_estimator = mi_estimator
        self.kl_coeff = kl_coeff
        
        self.threshold_policies = nn.ModuleDict({
            "lowSNR": Policy(input_size, num_thresholds * (dimension + 1), policy_scale*self.dimension).to(device),
            "midSNR": Policy(input_size, num_thresholds * (dimension + 1), policy_scale*self.dimension).to(device),
            "highSNR": Policy(input_size, num_thresholds * (dimension + 1), policy_scale*self.dimension).to(device),
        })

        self.point_policies = nn.ModuleDict({
            "lowSNR": Policy(input_size, alphabet_size * dimension, policy_scale*self.dimension).to(device),
            "midSNR": Policy(input_size, alphabet_size * dimension, policy_scale*self.dimension).to(device),
            "highSNR": Policy(input_size, alphabet_size * dimension, policy_scale*self.dimension).to(device),
        })

    def select_policy(self,snr,policy_type):
        if snr < 0.0:
            snr_level = "lowSNR"
        elif snr < 10.0:
            snr_level = "midSNR"
        else:
            snr_level = "highSNR"

        if policy_type == "threshold":
            return self.threshold_policies[snr_level], snr_level
        elif policy_type == "point":
            return self.point_policies[snr_level], snr_level
        else:
            raise ValueError("Invalid policy type. Choose 'threshold' or 'point'.")
        
    def compute_discounted_returns(self, rewards, gamma=0.8):
        discounted_returns = torch.zeros_like(rewards).to(device)
        running_return = 0
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + gamma * running_return
            discounted_returns[t] = running_return
        return discounted_returns

    def compute_loss(self, rewards, log_probs):
        rewards = torch.tensor(rewards, dtype=torch.float32)
        log_probs = torch.stack(log_probs)
        returns = self.compute_discounted_returns(rewards, gamma=0.99)
        loss = -torch.sum(log_probs * returns,dim=0).mean()
        return loss

    def compute_kl_divergence(self,mean_new, std_new, mean_old, std_old):
        kl_div = torch.log2(std_old / (std_new+1e-7) + 1e-7) + (std_new**2 + (mean_new - mean_old)**2) / (2 * std_old**2 +1e-7) - 0.5
        return kl_div.sum(dim=-1).mean()
    
    def train_policies(self, num_episodes, num_envs, envs, snr_range, policy_type, lr):
        num_envs = len(envs)
        policy, snr_level = self.select_policy(sum(snr_range)/2,policy_type)
        optimizer = optim.Adam(policy.parameters(), lr=lr)

        loss_list = []
        for episode in tqdm(range(num_episodes),desc=policy_type):
            optimizer.zero_grad()
            states = [env.reset() for env in envs]
            states_tensor = torch.stack(states).to(device).to(torch.float32)
            
            log_probs = []
            rewards_list = []
            
            done = [False] * num_envs
            while not all(done):
                action_mean, action_std = policy(states_tensor)

                if torch.isnan(action_mean).any() or torch.isnan(action_std).any():
                    print('NaN values have been detected')
                    print(torch.isfinite(states_tensor).all())
                    print(action_mean,action_std)
                    print(action_mean_old,action_std_old)
                    print(rewards)

                action_mean = action_mean + 1e-7
                action_std = action_std + 1e-7
                action_dist = Normal(action_mean, action_std)

                if policy_type == 'threshold':
                    action_lines = action_dist.sample()
                    action_points = torch.zeros(num_envs,self.alphabet_size*self.dimension).to(device)
                elif policy_type == "point":
                    action_points = action_dist.sample()
                    action_lines = torch.zeros(num_envs,self.num_thresholds*(self.dimension+1)).to(device) 
                actions = torch.cat([action_lines, action_points], dim=1)
                
                if policy_type == 'threshold':
                    next_states, rewards, done = zip(*[env.step_line(act,True) for env, act in zip(envs, actions)])
                    log_probs.append(action_dist.log_prob(action_lines).sum(dim=-1))
                elif policy_type == 'point':
                    next_states, rewards, done = zip(*[env.step_point(act) for env, act in zip(envs, actions)])
                    log_probs.append(action_dist.log_prob(action_points).sum(dim=-1))

                rewards_list.append([r.item() for r in rewards])
                
                states_tensor = torch.stack(next_states).to(device).to(torch.float32) 

            loss = self.compute_loss(rewards_list, log_probs)
            if episode == 0:
                action_mean_old = action_mean.detach()
                action_std_old = action_std.detach()
            else:
                kl_div = self.compute_kl_divergence(action_mean,action_std,action_mean_old,action_std_old)
                action_mean_old = action_mean.detach()
                action_std_old = action_std.detach()
                loss += self.kl_coeff*kl_div 

            loss_list.append(loss.item())
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()

        return loss_list

    def forward(self, state, policy_type):
        snr = state[0]
        policy, _ = self.select_policy(snr, policy_type)
        return policy(state)
    
    def select_action(self, state, policy_type):
        try:
            snr = state[0].item()
            policy, _ = self.select_policy(snr,policy_type)

            mean, std = policy(state)
            distribution = dist.Normal(mean, std)
            action = distribution.sample()
            log_prob = distribution.log_prob(action).sum(dim=-1)
        except:
            snr = state[0][0]
            policy, _ = self.select_policy(snr,policy_type)

            mean, std = policy(state)
            distribution = dist.Normal(mean, std)
            action = distribution.sample()
            log_prob = distribution.log_prob(action).sum(dim=-1) 
        
        return log_prob, action.to(device)