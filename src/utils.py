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
import sys
import os
from shapely.geometry import Polygon, LineString, Point, box
warnings.filterwarnings('ignore')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
project_root = os.path.abspath("..")  
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 

def create_dynamic_bounding_box(d, box_param):
    bounding_box = []

    for i in range(d):
        pos_bound = torch.zeros(d, dtype=torch.float32)
        pos_bound[i] = 1.0
        pos_bound = torch.cat((pos_bound, torch.tensor([box_param], dtype=torch.float32)))
        
        neg_bound = torch.zeros(d, dtype=torch.float32)
        neg_bound[i] = 1.0
        neg_bound = torch.cat((neg_bound, torch.tensor([-box_param], dtype=torch.float32)))
        
        bounding_box.append(pos_bound)
        bounding_box.append(neg_bound)

    bounding_box = torch.stack(bounding_box).to(device)
    return bounding_box

def assigning_centroids(dimension, threshold_lines, box_param):
    threshold_lines = threshold_lines.view(-1,dimension+1)
    bounding_box = create_dynamic_bounding_box(dimension,box_param)
    lines = torch.cat([threshold_lines,bounding_box])
    if dimension == 2:
        bounding_box = create_dynamic_bounding_box(dimension,box_param)
        segments = [LineString([(x, (-a * x - c) / (b+1e-7)) for x in [-box_param, box_param]]) if torch.abs(b)>torch.abs(a) 
                    else LineString([((-b * y - c) / (a+1e-7), y) for y in [-box_param, box_param]]) for (a, b, c) in lines]
        
        polygons = list(polygonize(unary_union(segments)))
        centroids = [[poly.centroid.x,poly.centroid.y] for poly in polygons]
    elif dimension == 1:
        intercepts = []
        for (a, c) in lines:
            if a != 0:  
                intercepts.append((-c / (a+1e-7)).item())

        intercepts = sorted(intercepts)  

        centroids = [(intercepts[i] + intercepts[i + 1]) / 2 for i in range(len(intercepts) - 1)]
    return torch.tensor(centroids).to(device)

def init_thresholds_2D(num_thresholds,bound):

    v_thresh = int((num_thresholds + 1)/2)
    h_thresh = int(num_thresholds/2)

    thresholds = []
    if v_thresh >= 1 and v_thresh%2 != 0:
        thresholds.append([1, 0, 0])  

    if h_thresh >= 1 and h_thresh%2 != 0:
        thresholds.append([0, 1, 0]) 

    if v_thresh >= 2:
        rand_num = random.uniform(0.1, bound-0.2)
        thresholds.append([1, 0, -rand_num])  
        thresholds.append([1, 0, rand_num])  

    if h_thresh >= 2:
        rand_num = random.uniform(0.1, bound-0.2)
        thresholds.append([0, 1, -rand_num])  
        thresholds.append([0, 1, rand_num]) 

    idx = 7
    while len(thresholds) < num_thresholds:
        offset = random.uniform(0.1, bound-0.2)
        if idx % 2 == 1:
            thresholds.append([1, 0, -offset])  
            thresholds.append([1, 0, offset]) 
        else:
            thresholds.append([0, 1, -offset])  
            thresholds.append([0, 1, offset])  
        idx += 1
        offset /= 2  

    return torch.tensor(thresholds[:num_thresholds], dtype=torch.float)

def all_regions_have_points(points, thresholds, dim):
    points = points.view(-1, dim)

    if dim == 1:
        thresholds = thresholds.view(-1, 2)
        xs = []

        for a, c in thresholds:
            if a != 0:
                xs.append(-c / a)

        if not xs:
            return points.numel() > 0 
        
        xs = torch.tensor(xs)
        xs, _ = torch.sort(xs)

        regions = [float('-inf')] + xs.tolist() + [float('inf')]
        for i in range(len(regions) - 1):
            left, right = regions[i], regions[i + 1]
            if not ((points[:, 0] > left) & (points[:, 0] < right)).any():
                return False

        return True
    
    elif dim == 2:
        threshold_lines = thresholds.view(-1,dim+1)
        box_param=1.0
        bounding_box = create_dynamic_bounding_box(dim,box_param)
        lines = torch.cat([threshold_lines,bounding_box])
        segments = [LineString([(x, (-a * x - c) / (b+1e-7)) for x in [-box_param, box_param]]) if torch.abs(b)>torch.abs(a) 
                    else LineString([((-b * y - c) / (a+1e-7), y) for y in [-box_param, box_param]]) for (a, b, c) in lines]
        
        polygons = list(polygonize(unary_union(segments)))

        points = points.view(-1, 2)
        shapely_points = [Point(x.item(), y.item()) for x, y in points]

        for region in polygons:
            if not any(region.contains(pt) for pt in shapely_points):
                return False
        return True

    else:
        raise ValueError("Only 1D and 2D inputs are supported.")