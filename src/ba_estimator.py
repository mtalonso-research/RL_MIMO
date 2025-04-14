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
import sys
import os
warnings.filterwarnings('ignore')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
project_root = os.path.abspath("..")  
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from src.utils import create_dynamic_bounding_box
from src.channel import Channel

def BlahutArimoto(dimension, P_Y_g_X, P_X, quant_points, outer_iter=100, inner_iter=100, compute_cost=True):
    device = P_Y_g_X.device
    P_Y_g_X = P_Y_g_X + 1e-7 

    batch_size, alph_X, alph_Y = P_Y_g_X.shape
    P_X = P_X.view(batch_size, alph_X)
    quant_points = quant_points.view(batch_size, alph_X, dimension)

    Q_X_g_Y = torch.zeros(batch_size, alph_X, alph_Y, device=device)
    C = -10.0 * torch.ones(batch_size, device=device)

    epsilon = torch.ones(batch_size, device=device)
    alpha = 0.1

    for i in range(outer_iter):
        C_prev = C.clone()

        P_X_reshaped = P_X.view(batch_size, 1, alph_X) 
        P_Y = torch.matmul(P_X_reshaped, P_Y_g_X).view(batch_size, alph_Y) 

        P_X_expanded = P_X.view(batch_size, alph_X, 1) 
        P_Y_expanded = P_Y.view(batch_size, 1, alph_Y) 
        Q_X_g_Y = (P_X_expanded * P_Y_g_X) / (P_Y_expanded + 1e-7) 

        if compute_cost:
            lambd = torch.ones(batch_size, 1, device=device) * 0.1
            upsilon = torch.ones(batch_size, device=device)

            for j in range(inner_iter):
                prev_P_X = P_X.clone()

                distortion = (quant_points ** 2).sum(dim=2) 
                log_term = torch.sum(P_Y_g_X * torch.log2(Q_X_g_Y + 1e-7), dim=2) 
                Temp = log_term - lambd.view(batch_size, 1) * distortion  

                P_X = (2 ** Temp) / (torch.sum(2 ** Temp, dim=1, keepdim=True) + 1e-7) 

                avg_power = (P_X * distortion).sum(dim=1)  
                lambd = (lambd + alpha * (avg_power.view(batch_size, 1) - 1.0)).clamp(min=0)

                upsilon = torch.sqrt(torch.sum((P_X - prev_P_X) ** 2, dim=1)) 
                if torch.all(upsilon < 1e-5):
                    break

        else:
            log_term = torch.sum(P_Y_g_X * torch.log2(Q_X_g_Y + 1e-7), dim=2) 
            Temp = log_term
            P_X = (2 ** Temp) / (torch.sum(2 ** Temp, dim=1, keepdim=True) + 1e-7)

        P_X = P_X / (P_X.sum(dim=1, keepdim=True) + 1e-7) 

        P_Y_broadcast = P_Y.view(batch_size, 1, alph_Y)  
        log_ratio = torch.log2(P_Y_g_X / (P_Y_broadcast + 1e-7) + 1e-7) 
        C = (P_X.view(batch_size, alph_X, 1) * (P_Y_g_X * log_ratio)).sum(dim=2).sum(dim=1) 

        epsilon = torch.sqrt(torch.sum((C - C_prev) ** 2))  
        if torch.all(epsilon < 1e-5):
            break

    flat_P_X = P_X.view(-1)  
    flat_quant = quant_points.view(-1, dimension) 
    power = (flat_P_X * (flat_quant ** 2).sum(dim=1)).sum() 

    return C, P_X, power

class MI_ESTIMATOR:
    def __init__(self,dimension,box_param,channel_type='identity-csi',num_samples=int(10e+5),cost_coef=10.0):
        self.box_param = box_param
        self.cost_coef = cost_coef
        self.num_samples = num_samples
        self.dimension = dimension
        self.channel = Channel(dimension,channel_type,box_param,num_samples)
    
    def count_regions(self, threshold_lines):
        threshold_lines = threshold_lines.view(-1,self.dimension+1)

        if self.dimension == 2:
            bounding_box = create_dynamic_bounding_box(self.dimension,self.box_param)
            lines = torch.cat([threshold_lines,bounding_box])
            segments = [LineString([(x, (-a * x - c) / (b+1e-7)) for x in [-self.box_param, self.box_param]]) if torch.abs(b)>torch.abs(a) 
                        else LineString([((-b * y - c) / (a+1e-7), y) for y in [-self.box_param, self.box_param]]) for (a, b, c) in lines]
            polygons = list(polygonize(unary_union(segments)))
            return len(polygons)
        
        elif self.dimension == 1:
            return len(threshold_lines) + 1

    def __call__(self,state,num_lines,verbose=False,stt=True):
        SNR = state[0]
        threshold_lines = state[1:num_lines*(self.dimension+1)+1].view(-1,(self.dimension+1))
        num_regions = self.count_regions(threshold_lines)
        quant_points = state[num_lines*(self.dimension+1)+1:].view(-1,(self.dimension))[:num_regions].to(torch.float)

        pxx = (torch.ones(num_regions)/num_regions).to(torch.float).to(device) 
        X = quant_points[torch.multinomial(pxx, self.num_samples, replacement=True)]
        _, P_YgX = self.channel(X,quant_points,threshold_lines,num_regions,SNR,diff_requirement=False,verbose=verbose,stt=stt)

        MI, P_X, power = BlahutArimoto(self.dimension,P_YgX.view(1,num_regions,-1),pxx, quant_points) 
        cost = (power - 1.001).clamp(min=0.0)
        reward = MI - self.cost_coef*cost

        used_regions = (P_X > 0.001).sum()
        if verbose: 
            print('Num Regions:',num_regions)
            print('Used Regions:',used_regions.item())
            print('MI:',MI.item())
            print('P(X):',P_X.detach().numpy())
            print('Used Power:',power.item())

        return reward.requires_grad_(), (MI,used_regions,power,P_X,num_regions)