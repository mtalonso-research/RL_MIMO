import torch
import sys
import os
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from src.rl_policy import UnifiedPolicy, Policy
from src.ba_estimator import MI_ESTIMATOR
from src.cortical_estimator import CORTICAL
from src.channel import Channel