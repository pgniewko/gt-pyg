import pandas as pd


from tdc.single_pred import ADME
from rdkit import RDLogger
from rdkit import rdBase
from rdkit import Chem
import rdkit
import torch
import math
import torch.nn.functional as F
import numpy as np
from torch import nn
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_scatter.composite import scatter_softmax
from torch_scatter.scatter import scatter_add
from torch_geometric.utils import softmax
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.nn.aggr import MultiAggregation
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
import os.path as osp
from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree
from torch.optim.lr_scheduler import ReduceLROnPlateau


from .layers import MLP, GTConv

class GraphTransformerNet(nn.Module):
    def __init__(self, node_dim_in,
                 edge_dim_in=None,
                 pe_in_dim=None,
                 hidden_dim=128, 
                 norm='bn',
                 num_gt_layers=4, 
                 num_heads=8,
                 aggregators=['sum'],
                 act='relu', dropout=0.0):
        super(GraphTransformerNet, self).__init__()
        
        self.node_emb = nn.Linear(node_dim_in, hidden_dim)
        if edge_dim_in:
            self.edge_emb = nn.Linear(edge_dim_in, hidden_dim)
        else:
            self.edge_emb = self.register_parameter('edge_emb', None)
            
        if pe_in_dim:
            self.pe_emb = nn.Linear(pe_in_dim, hidden_dim)
        else:
            self.pe_emb = self.register_parameter('pe_emb', None)
        
        self.gt_layers = nn.ModuleList()
        for _ in range(num_gt_layers):
            self.gt_layers.append(GTConv(node_in_dim=hidden_dim,
                                         hidden_dim=hidden_dim, 
                                         edge_in_dim=hidden_dim,
                                         num_heads=num_heads,
                                         act=act,
                                         dropout=dropout,
                                         norm='bn'))
        
        self.global_pool = MultiAggregation(aggregators, mode='cat')
        
        num_aggrs = len(aggregators)
        self.mu_mlp = MLP(input_dim=num_aggrs * hidden_dim, output_dim=1,
                          hidden_dims=hidden_dim,
                          num_hidden_layers=1, dropout=0.0, act=act)
        self.std_mlp = MLP(input_dim=num_aggrs * hidden_dim, output_dim=1,
                           hidden_dims=hidden_dim,
                           num_hidden_layers=1, dropout=0.0, act=act)
        
        
        self.reset_parameters()
        
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.node_emb.weight)
        nn.init.xavier_uniform_(self.edge_emb.weight)
            
            
    def forward(self, x, edge_index, edge_attr, pe, batch, return_std=False):
        x = self.node_emb(x.squeeze())
        x = x + self.pe_emb(pe) # squezee?
        edge_attr = self.edge_emb(edge_attr)
        
        for gt_layer in self.gt_layers:
            (x, edge_attr) = gt_layer(x, edge_index, edge_attr=edge_attr)

        x = self.global_pool(x, batch)
        mu = self.mu_mlp(x)
        log_var = self.std_mlp(x)
        std = torch.exp(0.5 * log_var)
        
        if self.training:
            eps = torch.randn_like(std)
            return mu + std * eps, std
        else:
            return mu, std
        
    def num_parameters(self):   
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())
        count = sum([p.numel() for p in trainable_params])
        return count


