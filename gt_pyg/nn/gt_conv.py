# Standard
import math
from typing import List, Optional

# Third party
import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch_geometric.nn.aggr import MultiAggregation

# GT-PyG
from .mlp import MLP


class GTConv(MessagePassing):
    def __init__(
        self,
        node_in_dim: int,
        hidden_dim: int,
        edge_in_dim: Optional[int] = None,
        num_heads: int = 8,
        gate=False,
        qkv_bias=False,
        dropout: float = 0.0,
        norm: str = "bn",
        act: str = "relu",
        aggregators: List[str] = ["sum"],
    ):
        """
        Graph Transformer Convolution (GTConv) module.

        Args:
            node_in_dim (int): Dimensionality of the input node features.
            hidden_dim (int): Dimensionality of the hidden representations.
            edge_in_dim (int, optional): Dimensionality of the input edge features.
                                         Default is None.
            num_heads (int, optional): Number of attention heads. Default is 8.
            dropout (float, optional): Dropout probability. Default is 0.0.
            gate (bool, optional): Use a gate attantion mechanism.
                                   Default is False
            qkv_bias (bool, optional): Bias in the attention mechanism.
                                       Default is False
            norm (str, optional): Normalization type. Options: "bn" (BatchNorm), "ln" (LayerNorm).
                                  Default is "bn".
            act (str, optional): Activation function name. Default is "relu".
            aggregators (List[str], optional): Aggregation methods for the messages aggregation.
                                               Default is ["sum"].
        """
        super().__init__(node_dim=0, aggr=MultiAggregation(aggregators, mode="cat"))

        assert (
            "sum" in aggregators
        )  # makes sure that the original sum_j is always part of the message passing
        assert hidden_dim % num_heads == 0
        assert (edge_in_dim is None) or (edge_in_dim > 0)

        self.aggregators = aggregators
        self.num_aggrs = len(aggregators)

        self.WQ = nn.Linear(node_in_dim, hidden_dim, bias=qkv_bias)
        self.WK = nn.Linear(node_in_dim, hidden_dim, bias=qkv_bias)
        self.WV = nn.Linear(node_in_dim, hidden_dim, bias=qkv_bias)
        self.WO = nn.Linear(hidden_dim * self.num_aggrs, node_in_dim, bias=True)

        if edge_in_dim is not None:
            self.WE = nn.Linear(edge_in_dim, hidden_dim, bias=True)
            self.WOe = nn.Linear(hidden_dim, edge_in_dim, bias=True)
            self.ffn_e = MLP(
                input_dim=edge_in_dim,
                output_dim=edge_in_dim,
                hidden_dims=hidden_dim,
                num_hidden_layers=1,
                dropout=dropout,
                act=act,
            )
            if norm.lower() in ["bn", "batchnorm", "batch_norm"]:
                self.norm1e = nn.BatchNorm1d(edge_in_dim)
                self.norm2e = nn.BatchNorm1d(edge_in_dim)
            elif norm.lower() in ["ln", "layernorm", "layer_norm"]:
                self.norm1e = nn.LayerNorm(edge_in_dim)
                self.norm2e = nn.LayerNorm(edge_in_dim)
            else:
                raise ValueError
        else:
            assert gate is False
            self.WE = self.register_parameter("WE", None)
            self.WOe = self.register_parameter("WOe", None)
            self.ffn_e = self.register_parameter("ffn_e", None)
            self.norm1e = self.register_parameter("norm1e", None)
            self.norm2e = self.register_parameter("norm2e", None)

        if norm.lower() in ["bn", "batchnorm", "batch_norm"]:
            self.norm1 = nn.BatchNorm1d(node_in_dim)
            self.norm2 = nn.BatchNorm1d(node_in_dim)
        elif norm.lower() in ["ln", "layernorm", "layer_norm"]:
            self.norm1 = nn.LayerNorm(node_in_dim)
            self.norm2 = nn.LayerNorm(node_in_dim)

        if gate:
            self.n_gate = nn.Linear(node_in_dim, hidden_dim, bias=True)
            self.e_gate = nn.Linear(edge_in_dim, hidden_dim, bias=True)
        else:
            self.n_gate = self.register_parameter("n_gate", None)
            self.e_gate = self.register_parameter("e_gate", None)

        self.dropout_layer = nn.Dropout(p=dropout)

        self.ffn = MLP(
            input_dim=node_in_dim,
            output_dim=node_in_dim,
            hidden_dims=hidden_dim,
            num_hidden_layers=1,
            dropout=dropout,
            act=act,
        )

        self.num_heads = num_heads
        self.node_in_dim = node_in_dim
        self.edge_in_dim = edge_in_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.norm = norm.lower()
        self.gate = gate
        self.qkv_bias = qkv_bias

        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize all learnable parameters.

        - Linear layers use Xavier uniform on weights and zeros on biases.
        - Norm layers are reset to weight=1, bias=0 (and BN running stats reset).
        - Submodules with `reset_parameters` are recursively reset.
        """
        # Q, K, V, O
        nn.init.xavier_uniform_(self.WQ.weight)
        if self.WQ.bias is not None:
            nn.init.zeros_(self.WQ.bias)

        nn.init.xavier_uniform_(self.WK.weight)
        if self.WK.bias is not None:
            nn.init.zeros_(self.WK.bias)

        nn.init.xavier_uniform_(self.WV.weight)
        if self.WV.bias is not None:
            nn.init.zeros_(self.WV.bias)

        nn.init.xavier_uniform_(self.WO.weight)
        if self.WO.bias is not None:
            nn.init.zeros_(self.WO.bias)

        # Edge-related linears (if present)
        if self.edge_in_dim is not None:
            nn.init.xavier_uniform_(self.WE.weight)
            if self.WE.bias is not None:
                nn.init.zeros_(self.WE.bias)

            nn.init.xavier_uniform_(self.WOe.weight)
            if self.WOe.bias is not None:
                nn.init.zeros_(self.WOe.bias)

        # Gate linears (if present)
        if self.gate and self.n_gate is not None:
            nn.init.xavier_uniform_(self.n_gate.weight)
            if self.n_gate.bias is not None:
                nn.init.zeros_(self.n_gate.bias)
        if self.gate and self.e_gate is not None:
            nn.init.xavier_uniform_(self.e_gate.weight)
            if self.e_gate.bias is not None:
                nn.init.zeros_(self.e_gate.bias)

        # Norm layers
        if isinstance(self.norm1, nn.BatchNorm1d):
            self.norm1.reset_running_stats()
            nn.init.ones_(self.norm1.weight)
            nn.init.zeros_(self.norm1.bias)
        elif isinstance(self.norm1, nn.LayerNorm):
            nn.init.ones_(self.norm1.weight)
            nn.init.zeros_(self.norm1.bias)

        if isinstance(self.norm2, nn.BatchNorm1d):
            self.norm2.reset_running_stats()
            nn.init.ones_(self.norm2.weight)
            nn.init.zeros_(self.norm2.bias)
        elif isinstance(self.norm2, nn.LayerNorm):
            nn.init.ones_(self.norm2.weight)
            nn.init.zeros_(self.norm2.bias)

        if self.edge_in_dim is not None:
            if isinstance(self.norm1e, nn.BatchNorm1d):
                self.norm1e.reset_running_stats()
                nn.init.ones_(self.norm1e.weight)
                nn.init.zeros_(self.norm1e.bias)
            elif isinstance(self.norm1e, nn.LayerNorm):
                nn.init.ones_(self.norm1e.weight)
                nn.init.zeros_(self.norm1e.bias)

            if isinstance(self.norm2e, nn.BatchNorm1d):
                self.norm2e.reset_running_stats()
                nn.init.ones_(self.norm2e.weight)
                nn.init.zeros_(self.norm2e.bias)
            elif isinstance(self.norm2e, nn.LayerNorm):
                nn.init.ones_(self.norm2e.weight)
                nn.init.zeros_(self.norm2e.bias)

        # Submodules (FFNs)
        if hasattr(self.ffn, "reset_parameters"):
            self.ffn.reset_parameters()
        if self.edge_in_dim is not None and hasattr(self.ffn_e, "reset_parameters"):
            self.ffn_e.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None):
        x_ = x
        edge_attr_ = edge_attr

        Q = self.WQ(x).view(-1, self.num_heads, self.hidden_dim // self.num_heads)
        K = self.WK(x).view(-1, self.num_heads, self.hidden_dim // self.num_heads)
        V = self.WV(x).view(-1, self.num_heads, self.hidden_dim // self.num_heads)
        if self.gate:
            G = self.n_gate(x).view(
                -1, self.num_heads, self.hidden_dim // self.num_heads
            )
        else:
            G = torch.ones_like(V)  # G*V = V

        out = self.propagate(
            edge_index, Q=Q, K=K, V=V, G=G, edge_attr=edge_attr, size=None
        )
        out = out.view(-1, self.hidden_dim * self.num_aggrs)  # concatenation

        # NODES
        out = self.dropout_layer(out)
        out = self.WO(out) + x_
        out = self.norm1(out)
        # FFN--nodes
        ffn_in = out
        out = self.ffn(out)
        out = self.norm2(ffn_in + out)

        if self.edge_in_dim is None:
            out_eij = None
        else:
            out_eij = self._eij
            self._eij = None
            out_eij = out_eij.view(-1, self.hidden_dim)

            # EDGES
            out_eij = self.dropout_layer(out_eij)
            out_eij = self.WOe(out_eij) + edge_attr_  # Residual connection
            out_eij = self.norm1e(out_eij)
            # FFN--edges
            ffn_eij_in = out_eij
            out_eij = self.ffn_e(out_eij)
            out_eij = self.norm2e(ffn_eij_in + out_eij)

        return (out, out_eij)

    def message(self, Q_i, K_j, V_j, G_j, index, edge_attr=None):
        Dh = self.hidden_dim // self.num_heads

        assert Dh == Q_i.size(-1)

        logits_vec = (Q_i * K_j) / math.sqrt(Dh)  # [E, H, Dh]

        if self.edge_in_dim is not None:
            assert edge_attr is not None
            E_vec = self.WE(edge_attr).view(-1, self.num_heads, Dh)  # [E, H, Dh]
            logits_vec = E_vec * logits_vec
            self._eij = logits_vec
        else:
            self._eij = None

        if self.gate:
            assert edge_attr is not None
            e_gate = self.e_gate(edge_attr).view(-1, self.num_heads, Dh)
            logits_vec = logits_vec * torch.sigmoid(e_gate)

        logits = logits_vec.sum(dim=-1)      # [E, H]
        alpha = softmax(logits, index) # [E, H]

        if self.gate:
            V_j = V_j * torch.sigmoid(G_j)

        return alpha.view(-1, self.num_heads, 1) * V_j

    def __repr__(self) -> str:
        aggrs = ",".join(self.aggregators)
        return (
            f"{self.__class__.__name__}({self.node_in_dim}, "
            f"{self.hidden_dim}, heads={self.num_heads}, "
            f"aggrs: {aggrs}, "
            f"qkv_bias: {self.qkv_bias}, "
            f"gate: {self.gate})"
        )

