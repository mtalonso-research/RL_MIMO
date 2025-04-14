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

class Channel():
    def __init__(self,dimension,channel_type,box_param,num_samples):
        self.num_samples = num_samples
        self.dimension = dimension
        self.box_param = box_param
        self.N_base = torch.randn(num_samples, dimension)
        if channel_type in ['identity-csi','noisy-csi-0.01',
                            'noisy-csi-0.05','noisy-csi-0.1',
                            'noisy-csi-0.5','changing-csi-smooth-0.01',
                            'changing-csi-smooth-0.05']:
            self.channel_type = channel_type
            self.base_H = torch.eye(dimension)
            noise_var = {'identity-csi':0.00,
                         'noisy-csi-0.01':0.01,
                         'noisy-csi-0.05':0.05,
                         'noisy-csi-0.1':0.1,
                         'noisy-csi-0.5':0.5,
                         'changing-csi-smooth-0.01':0.00,
                         'changing-csi-smooth-0.05':0.00}[channel_type]  
            self.noise_var = torch.tensor(noise_var)

            self.H_t = self.base_H.clone()
            self.alpha = 0.98  
            self.drift_std = {'changing-csi-smooth-0.01': 0.01,
                              'changing-csi-smooth-0.05': 0.05}.get(channel_type, 0.00)
            
        else: print('Invalid Channel')

    def quantizer(self, X, quant_points,threshold_lines):
        X = X.view(-1, self.dimension)
        region_labels = torch.zeros(X.shape[0], device=device)
        for num,line in enumerate(threshold_lines):
            condition = -line[-1] < torch.sum(line[:-1] * X, dim=1)
            region_labels += condition.long() * 10**num
        _, region_num = torch.unique(region_labels, sorted=True, return_inverse=True)
        quant_dict = {reg_num:point for reg_num, point in enumerate(quant_points)}
        return region_labels, region_num, quant_dict
    
    def generate_noise_for_quantized_y(self,X,P_Y_given_X,quant_dict):
        N = torch.zeros_like(X, dtype=torch.long)
        for i, x in enumerate(X):
            if P_Y_given_X[i].sum() == 0.0:
                P_Y_given_X[i] = torch.ones_like(P_Y_given_X[i])/len(P_Y_given_X[i])
            y_hat_index = torch.multinomial(P_Y_given_X[i], num_samples=1).item()
            y_hat = quant_dict[y_hat_index]
            N[i] = y_hat - x
        return N
    
    def conditional_probability(self, X, Y, num_regions):
        num_x = num_regions 
        num_y = num_regions 
        joint = torch.zeros(num_x, num_y, dtype=torch.float32)

        joint.index_put_((X, Y), torch.ones_like(X, dtype=torch.float32), accumulate=True)
        px = joint.sum(dim=1, keepdim=True)
        p_y_given_x = joint / px.clamp(min=1e-12)  
        zero_rows = p_y_given_x.sum(dim=1) == 0
        p_y_given_x[zero_rows] = 1.0 / num_y
        return p_y_given_x
    
    def __call__(self,X,quant_points,threshold_lines,num_points,SNR,diff_requirement=True,verbose=False,stt=True):
        n_var = 10 ** (-torch.tensor([SNR])/10)
        N = self.N_base * torch.sqrt(n_var)
        if self.channel_type.startswith('changing-csi-smooth'):
            self.H_t = self.alpha * self.H_t + (1 - self.alpha) * self.base_H \
                       + torch.randn_like(self.H_t) * self.drift_std
            H = self.H_t
        else:
            noise = torch.randn_like(self.base_H) * self.noise_var.sqrt()
            H = self.base_H + noise

        Y = ((H @ X.T).T + N).clamp(min=-self.box_param,max=self.box_param)

        _, Y_q, _ = self.quantizer(Y, quant_points, threshold_lines.view(-1,self.dimension+1))
        _, X_q, quant_dict = self.quantizer(X, quant_points, threshold_lines.view(-1,self.dimension+1))
        P_YgX = self.conditional_probability(X_q,Y_q,num_points).to(torch.float)

        if diff_requirement:
            N_hat = self.generate_noise_for_quantized_y(X,P_YgX[X_q],quant_dict)
            Y = ((H @ X.T).T + N_hat).clamp(min=-self.box_param,max=self.box_param)
        else: Y = Y_q

        if verbose: 
            fig = px.imshow(H,text_auto=True,title='Channel State Matrix')
            if stt: 
                import streamlit as st
                st.plotly_chart(fig)
            else: fig.show()
            fig = px.imshow(P_YgX,text_auto=True,title='Transition Probability Matrix')
            fig.update_layout(yaxis_title="X",xaxis_title="Y",)
            if stt: st.plotly_chart(fig)
            else: fig.show()
        return Y, P_YgX