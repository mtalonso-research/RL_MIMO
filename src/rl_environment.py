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
from ba_estimator import MI_ESTIMATOR
from utils import create_dynamic_bounding_box, all_regions_have_points, assigning_centroids, init_thresholds_2D

class RLEnvironment:
    def __init__(self,dimension,initial_state,num_lines,num_points,box_param,max_steps,patience,mi_estimator,norm_patience,display_on=False):
        self.box_param = box_param
        self.num_lines = num_lines
        self.num_points = num_points
        self.state = initial_state
        self.initial_state = initial_state
        self.dimension = dimension
        self.norm_patience = norm_patience
        self.p_x = [0.0 for qp in self.state[self.num_lines*(self.dimension+1)+1:]]

        self.max_steps = max_steps
        self.step_num = 0
        self.done = False
        self.mi_output = torch.zeros(5)
        self.best_reward = torch.tensor(float('-inf'))
        self.patience_counter = 0
        self.patience = patience
        self.display_on = display_on
        self.reward_function = mi_estimator
        self.final_mi_est = MI_ESTIMATOR(dimension, box_param, 'identity-csi', int(10e+5))
        self.training_mode = True

        if display_on:
            self.render()

    def assigning_centroids(self, threshold_lines):
        threshold_lines = threshold_lines.view(-1,self.dimension+1)
        bounding_box = create_dynamic_bounding_box(self.dimension,self.box_param)
        lines = torch.cat([threshold_lines,bounding_box])

        if self.dimension == 2:
            segments = [LineString([(x, (-a * x - c) / (b+1e-7)) for x in [-self.box_param, self.box_param]]) if torch.abs(b)>torch.abs(a) 
                        else LineString([((-b * y - c) / (a+1e-7), y) for y in [-self.box_param, self.box_param]]) for (a, b, c) in lines]
            
            polygons = list(polygonize(unary_union(segments)))
            centroids = [[poly.centroid.x,poly.centroid.y] for poly in polygons]

        elif self.dimension == 1:
            intercepts = []
            for (a, c) in lines:
                if a != 0:  
                    intercepts.append((-c / (a+1e-7)).item())
            intercepts = sorted(intercepts)  
            centroids = [(intercepts[i] + intercepts[i + 1]) / 2 for i in range(len(intercepts) - 1)]

        return torch.tensor(centroids).to(device)
    
    def oned_line_bound(self,thrsh):
        if self.dimension == 1:
            box_param = self.box_param - 0.2
            a = thrsh.view(-1,2)[:,0]
            b = thrsh.view(-1,2)[:,1]
            b_clamped = torch.clamp(b, min=a*(-box_param), max=a*box_param)
            thrsh = torch.stack((a,b_clamped),dim=1).view(-1)
            thrsh = thrsh.to(device)
        return thrsh
    
    def normalize_points_and_lines(self, points, lines):
        points_reshaped = points.view(-1, self.dimension)
        lines_reshaped = lines.view(-1, self.dimension + 1)
        if type(self.reward_function).__name__ == 'CORTICAL' and self.training_mode == True:
            avrg_loss = self.reward_function.train_on_the_go(self.state)
        reward, mi_outputs = self.reward_function(self.state,self.num_lines)
        power, pxx = mi_outputs[2],mi_outputs[3]

        if self.step_num < self.norm_patience[0]:
            return points, lines, reward, mi_outputs
        elif power <= 1.001 and self.step_num < self.norm_patience[1]:
            return points, lines, reward, mi_outputs   
        elif 0.3 <= power <= 1.001:
            return points, lines, reward, mi_outputs   
        else:
            alpha = (1.0 / power).sqrt()

            # Scale points
            norm_points = (points_reshaped * alpha).view(-1)
            points_reshaped = norm_points.view(-1, self.dimension)[:len(pxx.view(-1))]

            # Scale Lines
            coeff = lines_reshaped[:, :self.dimension] 
            bias = lines_reshaped[:, self.dimension:] * alpha  
            norm_lines = torch.cat([coeff, bias], dim=1).view(-1)

            snr = torch.tensor([self.state[0]]).to(device)
            if len(norm_points) < self.num_points*self.dimension:
                padding = torch.zeros(self.num_points*self.dimension - len(norm_points)).to(device) 
            else: padding = torch.tensor([]).to(device)
            norm_points = torch.cat([norm_points,padding])
            norm_lines = self.oned_line_bound(norm_lines)

            self.state = torch.cat([snr,norm_lines,norm_points])
            reward, mi_outputs = self.reward_function(self.state,self.num_lines)
            return norm_points, norm_lines, reward, mi_outputs
    
    def check_regions_have_pts(self,pts,snr,thrsh):
        if not all_regions_have_points(pts, thrsh, self.dimension):
            pts = self.assigning_centroids(thrsh).view(-1)
            if len(pts) < self.num_points*self.dimension:
                padding = torch.zeros(self.num_points*self.dimension - len(pts)).to(device) 
            else: padding = torch.tensor([]).to(device)
            pts = torch.cat([pts,padding])
            self.state = torch.cat([snr,thrsh,pts])
            pts, thrsh, self.reward, self.mi_output = self.normalize_points_and_lines(pts, thrsh)
    
    def step_line(self, action, training_line_policy=True):

        snr = torch.tensor([self.state[0]]).to(device)
        thrsh = self.state[1:self.num_lines*(self.dimension+1)+1] + action[:self.num_lines*(self.dimension+1)]*1e-2
        pts = self.state[self.num_lines*(self.dimension+1)+1:] 
        thrsh = self.oned_line_bound(thrsh)

        if training_line_policy: pts = self.assigning_centroids(thrsh).view(-1)
        if len(pts) < self.num_points*self.dimension:
            padding = torch.zeros(self.num_points*self.dimension - len(pts)).to(device) 
        else: padding = torch.tensor([]).to(device)
        self.state = torch.cat([snr,thrsh,torch.cat([pts,padding])])

        pts, thrsh, self.reward, self.mi_output = self.normalize_points_and_lines(pts, thrsh)
        num_regions = self.mi_output[4]
        self.step_num += 1

        self.done = self.check_done()
        self.check_regions_have_pts(pts,snr,thrsh)

        if self.reward <= self.best_reward and self.reward > 0.0:
            self.patience_counter += 1
        elif self.reward <= self.best_reward:
            self.patience_counter += 0.1
        else: 
            self.patience_counter = 0
            self.best_reward = self.reward

        return self.state, self.reward, self.done

    def step_point(self, action):
        snr = torch.tensor([self.state[0]]).to(device)
        thrsh = self.state[1:self.num_lines*(self.dimension+1)+1]
        pts = self.state[self.num_lines*(self.dimension+1)+1:] + action[self.num_lines*(self.dimension+1):]*1e-2

        if len(pts) < self.num_points*(self.dimension):
            padding = torch.zeros(self.num_points*(self.dimension) - len(pts)).to(device) 
        else: padding = torch.tensor([]).to(device)
        self.state = torch.cat([snr,thrsh,torch.cat([pts,padding])])

        pts, thrsh, self.reward, self.mi_output = self.normalize_points_and_lines(pts, thrsh)
        num_regions = self.mi_output[4]
        self.step_num += 1

        self.done = self.check_done()
        self.check_regions_have_pts(pts,snr,thrsh)

        if self.reward <= self.best_reward and self.reward > 0.0:
            self.patience_counter += 1
        elif self.reward <= self.best_reward:
            self.patience_counter += 0.1
        else: 
            self.patience_counter = 0
            self.best_reward = self.reward

        return self.state, self.reward, self.done
    
    def check_done(self):
        max_step_condition = bool(self.step_num == self.max_steps)
        max_reward_condition = bool(np.abs(self.best_reward.detach().numpy() - np.log(self.num_points)/np.log(2)) < 0.001)
        patience_condition = bool(self.patience_counter > self.patience and self.best_reward > 0.0)
        return bool(max_step_condition+max_reward_condition+patience_condition)

    def reset(self,to_initial=True):
        if to_initial: self.state = self.initial_state
        self.step_num = 0
        self.done = False
        self.fig = go.FigureWidget()
        self.patience_counter = 0
        self.best_reward = torch.tensor(float('-inf'))
        return self.state
    
    def render(self):
        snr = self.state[0]
        line_params = self.state[1:self.num_lines*(self.dimension+1)+1]
        point_params = self.state[self.num_lines*(self.dimension+1)+1:].view(-1, self.dimension)
        _,mi_outputs = self.final_mi_est(self.state,self.num_lines)

        x = point_params[:, 0].float().cpu().detach().numpy()
        if self.dimension==2: y = point_params[:, 1].float().cpu().detach().numpy()
        elif self.dimension==1: y = torch.zeros_like(point_params[:, 0]).float().cpu().detach().numpy()
        self.p_x = [f'Px: {p}' for p in mi_outputs[3][0]]
        figure_data = (go.Scatter(x=x,y=y,mode='markers',name='quant_points',text=self.p_x),)

        for i in range(0, len(line_params), self.dimension+1):
            if self.dimension == 2:
                a, b, c = line_params[i:i+self.dimension+1].cpu().detach().numpy()
                if np.abs(b) > np.abs(a):
                    x_line = np.linspace(-1000, 1000, 10000)
                    y_line = (-a * x_line - c) / (b+1e-7)
                else:
                    y_line = np.linspace(-1000, 1000, 10000)
                    x_line = (-b * y_line - c) / (a+1e-7)
                figure_data += (go.Scatter(x=x_line, y=y_line, mode='lines',name=f'{a:.2}X1 + {b:.2}X2 + {c:.2} = 0'),)

            elif self.dimension == 1:
                a, b = line_params[i:i+self.dimension+1].cpu().detach().numpy()
                y_line = np.linspace(-self.box_param, self.box_param, 10)
                x_line = np.ones_like(y_line) * (-b / (a+1e-7))
                figure_data += (go.Scatter(x=x_line, y=y_line, mode='lines',name=f'{a:.2}X1 + {b:.2} = 0'),)

        fig = go.Figure(data=figure_data)
        fig.update_layout(
            title=f'SNR: {snr}  Used Regions: {(mi_outputs[1])}  MI: {float(mi_outputs[0]):.3f}  Power: {float(mi_outputs[2]):.3f}, Steps: {float(self.step_num)}',
            xaxis=dict(range=[-self.box_param, self.box_param]),
            yaxis=dict(range=[-self.box_param, self.box_param]),
            shapes=[dict(type="rect",
                         x0=-self.box_param, y0=-self.box_param,
                         x1=self.box_param, y1=self.box_param,
                         line=dict(color="white", width=2),
                         fillcolor="rgba(0,0,0,0)")])
        return fig
    
def defining_environments(dimension,num_thresholds,alphabet_size,box_param,SNR_range,num_envs,max_steps,patience,mi_estimator,norm_patience,center_points,cortical_training_format,thrsh=None):
    envs = []
    SNR = torch.linspace(SNR_range[0],SNR_range[1],num_envs)

    if dimension == 2:
        for i in range(num_envs):
            if thrsh is not None: 
                thresholds = thrsh.view(-1).to(device)
            else:
                thresholds = init_thresholds_2D(num_thresholds,box_param)
                thresholds = thresholds.view(-1).to(device)

            if center_points: 
                quant_points = assigning_centroids(dimension, thresholds, box_param).view(-1)
                if len(quant_points) < alphabet_size*dimension:
                    padding = torch.zeros(alphabet_size*dimension - len(quant_points)).to(device) 
                else: padding = torch.tensor([]).to(device)
                quant_points = torch.cat([quant_points,padding]).view(-1)
            else: quant_points = torch.tensor([[0.0] * dimension]*alphabet_size).view(-1).to(device)

            if cortical_training_format: 
                envs.append(torch.cat([torch.tensor([SNR[i]]).to(device),thresholds,quant_points]))
            else: 
                state = torch.cat([torch.tensor([SNR[i]]).to(device),thresholds,quant_points])
                envs.append(RLEnvironment(dimension, state, num_thresholds, alphabet_size, box_param, max_steps, patience, mi_estimator,norm_patience))

        return envs, thresholds, quant_points
    
    elif dimension == 1:
        for i in range(num_envs):
            if thrsh is not None:
                thresholds = thrsh.view(-1).to(device)
            else:
                thresholds = []

                if num_thresholds % 2 == 0:
                    for _ in range(int(num_thresholds/2)):
                        a = create_dynamic_bounding_box(dimension, random.uniform(0.1, box_param-0.2))
                        thresholds.append(a)
                else:
                    for _ in range(int(num_thresholds/2)):
                        a = create_dynamic_bounding_box(dimension, random.uniform(0.1, box_param-0.2))
                        thresholds.append(a)
                    b = torch.tensor([[1.0,0.0]])
                    thresholds.append(b)
                thresholds = torch.cat(thresholds).sort(dim=0)[0].view(-1).to(device)

            if center_points:
                quant_points = assigning_centroids(dimension, thresholds, box_param).view(-1)
                if len(quant_points) < alphabet_size*dimension:
                    padding = torch.zeros(alphabet_size*dimension - len(quant_points)).to(device) 
                else: padding = torch.tensor([]).to(device)
                quant_points = torch.cat([quant_points,padding])
            else: quant_points = torch.tensor([[0.0] * dimension]*alphabet_size).view(-1).to(device)

            if cortical_training_format: 
                envs.append(torch.cat([torch.tensor([SNR[i]]).to(device),thresholds,quant_points]))
            else: 
                state = torch.cat([torch.tensor([SNR[i]]).to(device),thresholds,quant_points])
                envs.append(RLEnvironment(dimension, state, num_thresholds, alphabet_size, box_param, max_steps, patience, mi_estimator,norm_patience))

        return envs, thresholds, quant_points