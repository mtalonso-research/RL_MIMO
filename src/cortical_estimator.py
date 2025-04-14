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
from utils import assigning_centroids
from rl_environment import defining_environments
from channel import Channel

class Generator(nn.Module):
    def __init__(self, dimension, num_regions):
        super(Generator, self).__init__()
        self.dimension = dimension

        self.mlp = nn.Sequential(
            nn.Linear(1 + num_regions * dimension, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_regions),
        )

    def forward(self, z, quant_points):
        logits = self.mlp(z)
        logits = torch.clamp(logits, -10, 10)
        probs = F.softmax(logits, dim=-1)
        weights = F.gumbel_softmax(logits, tau=1, hard=True).unsqueeze(2)
        outputs = torch.sum(weights * quant_points, dim=1).reshape(-1, self.dimension)
        return outputs, probs

class Discriminator(nn.Module):
    def __init__(self, dimension, model='MINE'):
        super(Discriminator, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(2*dimension, 512), 
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 512), 
            nn.ReLU(),
            nn.Linear(512, 1) 
        )
        
        if model == 'i-DIME':
            self.final_activation = nn.Sigmoid()
        elif model == 'd-DIME':
            self.final_activation = nn.Softplus()  
        elif model == 'MINE':
            self.final_activation = nn.Identity() 
        else: print('Invalid model choice')

    def forward(self, x):
        x = self.mlp(x)
        if isinstance(self.final_activation, nn.Softplus):
            x = torch.clamp(x, max=20)
        x = self.final_activation(x)
        return x

class CORTICAL():
    def __init__(self, params, alphas, lambda_entropy, cost_coef=10.0, box_param=1.5):
        self.params = params
        self.box_param = box_param
        self.lambda_entropy = lambda_entropy
        self.cost_coef = cost_coef
        sample_size, dimension, alphabet_size, num_thresholds, num_batches = self.params

        self.G_low = Generator(dimension, alphabet_size)
        self.G_midL = Generator(dimension, alphabet_size)
        self.G_midH = Generator(dimension, alphabet_size)
        self.G_high = Generator(dimension, alphabet_size)

        self.D_low = Discriminator(dimension, model='d-DIME')
        self.D_midL = Discriminator(dimension, model='d-DIME')
        self.D_midH = Discriminator(dimension, model='d-DIME')
        self.D_high = Discriminator(dimension, model='d-DIME')

        self.channel = Channel(dimension,'identity-csi',box_param,sample_size)
        self.alphas = alphas
        self.snr_limits = [-2.0,4.0,10.0]

    def J_alpha_func(self, D1, D2, alpha):
        E1 = torch.mean(torch.log2(D1+1e-7))
        E2 = torch.mean(-D2)
        return alpha * E1 + E2

    def j_alpha_loss(self, D1, D2, alpha):
        J_alpha = self.J_alpha_func(D1,D2,alpha)
        C = J_alpha/alpha + 1 - torch.log2(alpha)
        return -J_alpha, C
    
    def avrg_pwr(self, outputs):
        pwr_c = (torch.mean(outputs**2) - 1.0).clamp(min=0) 
        return pwr_c
    
    def pi(self,y):
        d = y.shape[0]
        permuted_y = torch.empty_like(y)

        perm_indices = torch.randperm(d)
        permuted_y = y[perm_indices]

        return permuted_y
    
    def select_models(self, SNR, verbose=False):
        if SNR < self.snr_limits[0]:
            G = self.G_low
            D = self.D_low
            alpha = torch.tensor([self.alphas[0]])
            if verbose: print('Training Model for Low SNR')
            model_name = 'low_SNR'
        elif SNR < self.snr_limits[1]:
            G = self.G_midL
            D = self.D_midL
            alpha = torch.tensor([self.alphas[1]]) 
            if verbose: print('Training Model for Mid SNR')
            model_name = 'mid_SNRL'
        elif SNR < self.snr_limits[2]:
            G = self.G_midH
            D = self.D_midH
            alpha = torch.tensor([self.alphas[2]]) 
            if verbose: print('Training Model for Mid SNR')
            model_name = 'mid_SNRH'
        else:
            G = self.G_high
            D = self.D_high
            alpha = torch.tensor([self.alphas[3]]) 
            if verbose: print('Training Model for High SNR')
            model_name = 'high_SNR'
        return G, D, alpha, model_name
    
    def update_models(self,SNR,G,D,verbose=False):
        if SNR < self.snr_limits[0]:
            self.G_low = G
            self.D_low = D
            if verbose: print('Updated Model for Low SNR')
        elif SNR < self.snr_limits[1]:
            self.G_midL = G
            self.D_midL = D
            if verbose: print('Updated Model for MidL SNR')
        elif SNR < self.snr_limits[2]:
            self.G_midH = G
            self.D_midH = D
            if verbose: print('Updated Model for MidH SNR')
        else:
            self.G_high = G
            self.D_high = D
            if verbose: print('Updated Model for High SNR')
    
    def G_training_iteration(self, G, D, alpha, batch_config):
        sample_size, dimension, alphabet_size, num_thresholds, num_batches = self.params
        SNR, thresholds, quant_points = batch_config[0].view(-1, 1), batch_config[1:1+(dimension+1)*num_thresholds].view(-1, dimension+1), batch_config[1+(dimension+1)*num_thresholds:].view(-1, dimension)

        Z_in = torch.cat([batch_config[:1], batch_config[1+(dimension+1)*num_thresholds:]])
        Z_in = Z_in.unsqueeze(0).expand(sample_size, -1)

        X, probs = G(Z_in, quant_points.expand(sample_size, -1, dimension))
        X = X.view(-1, dimension)

        loss1 = self.avrg_pwr(X)

        Y,_ = self.channel(X, quant_points, thresholds, alphabet_size, SNR)
        D1 = D(torch.cat([X, Y], dim=1))
        D2 = D(torch.cat([X, self.pi(Y)], dim=1))

        loss2, _ = self.j_alpha_loss(D1, D2, alpha)
        entropy = -torch.sum(probs * torch.log2(probs + 1e-7), dim=1).mean()
        total_loss = loss1 + loss2 - self.lambda_entropy * entropy

        return X, total_loss

    
    def D_training_iteration(self, G, D, alpha, batch_config):
        sample_size, dimension, alphabet_size, num_thresholds, num_batches = self.params
        SNR, thresholds, quant_points = batch_config[0].view(-1, 1), batch_config[1:1+(dimension+1)*num_thresholds].view(-1, dimension+1), batch_config[1+(dimension+1)*num_thresholds:].view(-1, dimension)
        
        Z_in = torch.cat([batch_config[:1], batch_config[1+(dimension+1)*num_thresholds:]])
        Z_in = Z_in.unsqueeze(0).expand(sample_size, -1)

        X, probs = G(Z_in,quant_points.expand(sample_size,-1,dimension))
        X = X.view(-1,dimension)

        Y,_ = self.channel(X, quant_points, thresholds, alphabet_size, SNR)
        D1 = D(torch.cat([X, Y], dim=1))
        D2 = D(torch.cat([X, self.pi(Y)], dim=1))
        loss, c = self.j_alpha_loss(D1, D2, alpha)

        return c, loss
    
    def validation_iteration(self, G, D, alpha, batch_config):
        sample_size, dimension, alphabet_size, num_thresholds, num_batches = self.params
        SNR, thresholds, quant_points = batch_config[0].view(-1, 1), batch_config[1:1+(dimension+1)*num_thresholds].view(-1, dimension+1), batch_config[1+(dimension+1)*num_thresholds:].view(-1, dimension)
        
        Z_in = torch.cat([batch_config[:1], batch_config[1+(dimension+1)*num_thresholds:]])
        Z_in = Z_in.unsqueeze(0).expand(sample_size, -1)

        with torch.no_grad():
            X, probs = G(Z_in,quant_points.expand(sample_size,-1,dimension))
            X = X.view(-1,dimension)

            Y,_ = self.channel(X, quant_points, thresholds, alphabet_size, SNR)
            D1 = D(torch.cat([X, Y], dim=1))
            D2 = D(torch.cat([X, self.pi(Y)], dim=1))
            loss, c = self.j_alpha_loss(D1, D2, alpha)

        return X, Y, c, loss
    
    def train(self, SNR_values, epochs=50, subepochs= 10, patience=10, verbose=0, test_params=None):
        sample_size, dimension, alphabet_size, num_thresholds, num_batches = self.params
        SNR_mean = np.mean(SNR_values)
        G, D, alpha, _ = self.select_models(SNR_mean,verbose=True if verbose==2 else False)
        gen_optimizer = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5,0.999)) 
        disc_optimizer = optim.Adam(D.parameters(), lr=2e-3, betas=(0.5,0.999)) 

        best_loss = np.inf
        train_loss_list = []
        val_loss_list = []
        test_mi_list = []

        iterator = range(epochs) if verbose>0 or test_params!=None else tqdm(range(epochs))
        for i in iterator:
            configurations, _, _ = defining_environments(dimension,num_thresholds,alphabet_size,self.box_param,SNR_values,num_batches,None,None,None,None,True,True)
            val_configurations, _, _ = defining_environments(dimension,num_thresholds,alphabet_size,self.box_param,SNR_values,100,None,None,None,None,True,True)
            avrg_loss = 0

            # Training D
            for j in range(subepochs):
                for _, batch_config in enumerate(configurations):
                    disc_optimizer.zero_grad()
                    mi, disc_loss = self.D_training_iteration(G, D, alpha, batch_config)
                    disc_loss.backward()
                    torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=1.0)
                    disc_optimizer.step()
                    avrg_loss += disc_loss

            # Training G
            for _, batch_config in enumerate(configurations):
                gen_optimizer.zero_grad()
                x, gen_loss = self.G_training_iteration(G, D, alpha, batch_config)
                gen_loss.backward()
                torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
                gen_optimizer.step()
                avrg_loss += gen_loss

            # Average loss calculation
            avrg_loss /= ((1+subepochs) * len(configurations))

            val_losses = 0
            for _, batch_config in enumerate(val_configurations):
                _, _, _, val_loss = self.validation_iteration(G, D, alpha, batch_config)
                val_losses += val_loss
            val_losses /= len(val_configurations)

            if verbose>0: print(f"Epoch: {i+1}/{epochs}, Train Loss: {float(avrg_loss.detach()):.5}, Val Loss: {float(val_losses):.5}")
            if test_params != None:
                reward, call_output = self.__call__(test_params,None)
                mi,used_regions,power,prob,num_regions,X,Y,mdn = call_output
                print(f'Test Config MI: {float(mi):.5}, Test Config Px: {prob[0].detach().numpy()}, Model: {mdn}')
            train_loss_list.append(float(avrg_loss.detach()))
            val_loss_list.append(float(val_losses))
            test_mi_list.append(float(mi))

            # Saving models if loss improves
            if val_losses < best_loss:
                best_loss = val_losses
                self.update_models(SNR_mean,G,D,verbose=True if verbose==2 else False)
                patience_counter = 0

            if patience_counter > patience:
                G, D, _, _ = self.select_models(SNR_mean,verbose=True if verbose==2 else False)
                patience_counter = 0
                if verbose==2: print("LOADING BEST MODELS TO CONTINUE TRAINING")

        return train_loss_list, val_loss_list, test_mi_list

    def save_models(self,output_dir):
        torch.save(self.G_high, output_dir + f'/G_high_Dim:{self.params[1]}_T:{self.params[3]}.pth')
        torch.save(self.D_high, output_dir + f'/D_high_Dim:{self.params[1]}_T:{self.params[3]}.pth')

        torch.save(self.G_midH, output_dir + f'/G_midH_Dim:{self.params[1]}_T:{self.params[3]}.pth')
        torch.save(self.D_midH, output_dir + f'/D_midH_Dim:{self.params[1]}_T:{self.params[3]}.pth')

        torch.save(self.G_midL, output_dir + f'/G_midL_Dim:{self.params[1]}_T:{self.params[3]}.pth')
        torch.save(self.D_midL, output_dir + f'/D_midL_Dim:{self.params[1]}_T:{self.params[3]}.pth')

        torch.save(self.G_low, output_dir + f'/G_low_Dim:{self.params[1]}_T:{self.params[3]}.pth')
        torch.save(self.D_low, output_dir + f'/D_low_Dim:{self.params[1]}_T:{self.params[3]}.pth')
        print('Models have been saved')

    def load_models(self,directory):
        self.G_high = torch.load(directory + f'/G_high_Dim:{self.params[1]}_T:{self.params[3]}.pth')
        self.D_high = torch.load(directory + f'/D_high_Dim:{self.params[1]}_T:{self.params[3]}.pth')

        self.G_midH = torch.load(directory + f'/G_midH_Dim:{self.params[1]}_T:{self.params[3]}.pth')
        self.D_midH = torch.load(directory + f'/D_midH_Dim:{self.params[1]}_T:{self.params[3]}.pth')

        self.G_midL = torch.load(directory + f'/G_midL_Dim:{self.params[1]}_T:{self.params[3]}.pth')
        self.D_midL = torch.load(directory + f'/D_midL_Dim:{self.params[1]}_T:{self.params[3]}.pth')

        self.G_low = torch.load(directory + f'/G_low_Dim:{self.params[1]}_T:{self.params[3]}.pth')
        self.D_low = torch.load(directory + f'/D_low_Dim:{self.params[1]}_T:{self.params[3]}.pth')

    def train_on_the_go(self, batch_config, epochs=1,subepochs=2):
        sample_size, dimension, alphabet_size, num_thresholds, num_batches = self.params
        SNR, thresholds, quant_points = batch_config[0].view(-1, 1), batch_config[1:1+(dimension+1)*num_thresholds].view(-1, dimension+1), batch_config[1+(dimension+1)*num_thresholds:].view(-1, dimension)
        G, D, alpha, mdn = self.select_models(SNR)
        gen_optimizer = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5,0.999)) 
        disc_optimizer = optim.Adam(D.parameters(), lr=2e-3, betas=(0.5,0.999)) 
        avrg_loss = 0
        for i in range(epochs):
            for j in range(subepochs):
                disc_optimizer.zero_grad()
                mi, disc_loss = self.D_training_iteration(G, D, alpha, batch_config)
                disc_loss.backward()
                torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=1.0)
                disc_optimizer.step()
                avrg_loss += disc_loss

            gen_optimizer.zero_grad()
            x, gen_loss = self.G_training_iteration(G, D, alpha, batch_config)
            gen_loss.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
            gen_optimizer.step()
            avrg_loss += gen_loss

        return avrg_loss
    
    def __call__(self, batch_config, num_lines):
        sample_size, dimension, alphabet_size, num_thresholds, num_batches = self.params
        SNR, thresholds, quant_points = batch_config[0].view(-1, 1), batch_config[1:1+(dimension+1)*num_thresholds].view(-1, dimension+1), batch_config[1+(dimension+1)*num_thresholds:].view(-1, dimension)
        
        G, D, alpha, mdn = self.select_models(SNR)
        Z_in = torch.cat([batch_config[:1], batch_config[1+(dimension+1)*num_thresholds:]])
        Z_in = Z_in.unsqueeze(0).expand(sample_size, -1)

        X, probs = G(Z_in,quant_points.expand(sample_size,-1,dimension))
        X = X.view(-1,dimension)

        Y,_ = self.channel(X, quant_points, thresholds, alphabet_size, SNR)
        D1 = D(torch.cat([X, Y], dim=1))
        D2 = D(torch.cat([X, self.pi(Y)], dim=1))
        _, c = self.j_alpha_loss(D1, D2, alpha)
        power = (probs[0] * (quant_points**2).sum(1)).sum()
        cost = (power - 1.001).clamp(min=0.0)

        used_regions = (probs[0] > 0.001).sum()
        reward = c - self.cost_coef*cost
        num_regions = len(assigning_centroids(dimension,thresholds,1.0))
        return reward, (c,used_regions,power,probs,num_regions,X,Y,mdn)
