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


class MLP(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_dims,
                 num_hidden_layers=1,
                 dropout=0.0,
                 act='relu',
                 act_kwargs=None):
        super(MLP, self).__init__()
    
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims] * num_hidden_layers
        
        assert len(hidden_dims) == num_hidden_layers

        hidden_dims = [input_dim] + hidden_dims
        layers = []
        
        for (i_dim, o_dim) in zip(hidden_dims[:-1], hidden_dims[1:]):
            layers.append(nn.Linear(i_dim, o_dim, bias=True))
            layers.append(activation_resolver(act, **(act_kwargs or {})))
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
                
        layers.append(nn.Linear(hidden_dims[-1], output_dim, bias=True))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.mlp(x)

    
class GTConv(MessagePassing):
    def __init__(self, node_in_dim, hidden_dim, edge_in_dim=None, num_heads=1, dropout=0.0, norm='bn', act='relu'):
        super(GTConv, self).__init__(node_dim=0, aggr='add')
        
        assert hidden_dim % num_heads == 0
        assert (edge_in_dim is None) or (edge_in_dim > 0)
        
        self.WQ = nn.Linear(node_in_dim, hidden_dim, bias=True)
        self.WK = nn.Linear(node_in_dim, hidden_dim, bias=True)
        self.WV = nn.Linear(node_in_dim, hidden_dim, bias=True)
        self.WO = nn.Linear(hidden_dim, node_in_dim, bias=True)
        
        if edge_in_dim is not None:
            assert node_in_dim == edge_in_dim
            self.WE = nn.Linear(edge_in_dim, hidden_dim, bias=True)
            self.WOe = nn.Linear(hidden_dim, edge_in_dim, bias=True)
            self.ffn_e = MLP(input_dim=edge_in_dim,
                             output_dim=edge_in_dim,
                             hidden_dims=hidden_dim,
                             num_hidden_layers=1,
                             dropout=dropout, act=act)
            if norm.lower() in ['bn', 'batchnorm', 'batch_norm']:
                self.norm1e = nn.BatchNorm1d(edge_in_dim)
                self.norm2e = nn.BatchNorm1d(edge_in_dim)
            elif norm.lower() in ['ln', 'layernorm', 'layer_norm']:
                self.norm1e = nn.LayerNorm(edge_in_dim)
                self.norm2e = nn.LayerNorm(edge_in_dim)
            else:
                raise ValueError
        else:
            self.WE = self.register_parameter('WE', None)
            self.WOe = self.register_parameter('WOe', None)
            self.ffn_e = self.register_parameter('ffn_e', None)
            self.norm1e = self.register_parameter('norm1e', None)
            self.norm2e = self.register_parameter('norm2e', None)
        
        if norm.lower() in ['bn', 'batchnorm', 'batch_norm']:
            self.norm1 = nn.BatchNorm1d(node_in_dim)
            self.norm2 = nn.BatchNorm1d(node_in_dim)
        elif norm.lower() in ['ln', 'layernorm', 'layer_norm']:
            self.norm1 = nn.LayerNorm(node_in_dim)
            self.norm2 = nn.LayerNorm(node_in_dim)
            
        self.dropout_layer = nn.Dropout(p=dropout)
            
        self.ffn = MLP(input_dim=node_in_dim,
                       output_dim=node_in_dim,
                       hidden_dims=hidden_dim,
                       num_hidden_layers=1,
                       dropout=dropout, act=act)
        
        self.num_heads = num_heads
        self.node_in_dim = node_in_dim
        self.edge_in_dim = edge_in_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.norm = norm.lower()
        
        self.reset_parameters()
           
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.WQ.weight)
        nn.init.xavier_uniform_(self.WK.weight)
        nn.init.xavier_uniform_(self.WV.weight)
        nn.init.xavier_uniform_(self.WO.weight)
        if self.edge_in_dim is not None:
            nn.init.xavier_uniform_(self.WE.weight)
            nn.init.xavier_uniform_(self.WOe.weight)
        
    
    def forward(self, x, edge_index, edge_attr=None):
        x_ = x
        Q = self.WQ(x).view(-1, self.num_heads, self.hidden_dim // self.num_heads)
        K = self.WK(x).view(-1, self.num_heads, self.hidden_dim // self.num_heads)
        V = self.WV(x).view(-1, self.num_heads, self.hidden_dim // self.num_heads)
        
        out = self.propagate(edge_index, Q=Q, K=K, V=V,
                             edge_attr=edge_attr, size=None)
        out = out.view(-1, self.hidden_dim)
        
        ## NODES
        out = self.dropout_layer(out)
        out = self.WO(out) + x_ # Residual connection
        out = self.norm1(out)
        # FFN-NODES
        ffn_in = out
        out = self.ffn(out)
        out = self.norm2(ffn_in + out)
        
        if self.edge_in_dim is None:
            out_eij = None
        else:
            out_eij = self._eij
            self._eij = None
            out_eij = out_eij.view(-1, self.hidden_dim)

            ## EDGES
            out_eij_ = out_eij
            out_eij = self.dropout_layer(out_eij)
            out_eij = self.WOe(out_eij) + out_eij_ # Residual connection
            out_eij = self.norm1e(out_eij)
            # FFN-EDGES
            ffn_eij_in = out_eij
            out_eij = self.ffn_e(out_eij)
            out_eij = self.norm2e(ffn_eij_in + out_eij)

        return (out, out_eij)
        
        
    def message(self, Q_i, K_j, V_j, index, edge_attr=None):
        if self.WE is not None:
            assert edge_attr is not None
            E = self.WE(edge_attr).view(-1, self.num_heads, self.hidden_dim // self.num_heads)
            K_j = E * K_j
        
        d_k = Q_i.size(-1)
        qijk = (Q_i * K_j).sum(dim=-1) / math.sqrt(d_k)
        self._eij = (Q_i * K_j) / math.sqrt(d_k)
        alpha = softmax(qijk, index) # Log-Sum-Exp trick used. No need for clipping (-5,5)
        
        return alpha.view(-1, self.num_heads, 1) * V_j
    
    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.node_in_dim}, '
                f'{self.hidden_dim}, heads={self.num_heads})')
