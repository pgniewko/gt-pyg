# Standard
import math
from typing import List, Optional

# Third party
import torch
from torch import nn
from torch.nn import functional as F
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
        gate: bool = False,
        qkv_bias: bool = False,
        dropout: float = 0.0,
        norm: str = "ln",        # changed default to LN for transformer-like behavior
        act: str = "gelu",       # changed default to GELU
        aggregators: Optional[List[str]] = None,
    ):
        """
        Graph Transformer Convolution (GTConv) module. 

        - Pre-norm residual blocks for attention and FFN.
        - Wider, deeper FFNs by default.
        - Edge features contribute both as additive attention bias and to values.
        - Optional gating on values and logits.

        Args:
            node_in_dim (int): Dimensionality of the input node features.
            hidden_dim (int): Dimensionality of the hidden representations (per layer).
            edge_in_dim (int, optional): Dimensionality of the input edge features.
            num_heads (int, optional): Number of attention heads. Default is 8.
            gate (bool, optional): Use a gate attention mechanism. Default is False.
            qkv_bias (bool, optional): Bias in the attention projections. Default is False.
            dropout (float, optional): Dropout probability. Default is 0.0.
            norm (str, optional): "bn" or "ln" (BatchNorm/LayerNorm). Default is "ln".
            act (str, optional): Activation function name for FFNs. Default is "gelu".
            aggregators (List[str], optional): MultiAggregation methods. Default ["sum"].
        """
        if aggregators is None:
            aggregators = ["sum"]

        # Choose aggregation
        if len(aggregators) == 1 and aggregators[0] in ("sum", "add"):
            aggr = "add"
        else:
            aggr = MultiAggregation(aggregators, mode="cat")

        super().__init__(node_dim=0, aggr=aggr)

        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
            )
        if edge_in_dim is not None and edge_in_dim <= 0:
            raise ValueError(f"edge_in_dim must be positive or None, got {edge_in_dim}")

        self.aggregators = aggregators
        self.num_aggrs = len(aggregators)
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads

        self.node_in_dim = node_in_dim
        self.edge_in_dim = edge_in_dim
        self.dropout_p = dropout
        self.norm_type = norm.lower()
        self.gate = gate
        self.qkv_bias = qkv_bias

        # Node projections
        self.WQ = nn.Linear(node_in_dim, hidden_dim, bias=qkv_bias)
        self.WK = nn.Linear(node_in_dim, hidden_dim, bias=qkv_bias)
        self.WV = nn.Linear(node_in_dim, hidden_dim, bias=qkv_bias)

        # Node output projection (after aggregation over heads & aggregators)
        self.WO = nn.Linear(hidden_dim * self.num_aggrs, node_in_dim, bias=True)

        # Edge-related modules
        if edge_in_dim is not None:
            # Edge -> attention bias (per head)
            self.WE_logits = nn.Linear(edge_in_dim, num_heads, bias=True)
            # Edge -> value contribution (same hidden_dim as V)
            self.WE_value = nn.Linear(edge_in_dim, hidden_dim, bias=True)

            # Edge output projection (edge "attention block" -> edge space)
            self.WOe = nn.Linear(hidden_dim, edge_in_dim, bias=True)

            # Stronger edge FFN
            edge_ffn_hidden = max(hidden_dim, 2 * edge_in_dim)
            self.ffn_e = MLP(
                input_dim=edge_in_dim,
                output_dim=edge_in_dim,
                hidden_dims=edge_ffn_hidden,
                num_hidden_layers=2,
                dropout=dropout,
                act=act,
            )

            # Edge norms
            # NOTE: norm2e is registered but no longer called in forward().
            # It is kept so that checkpoints saved before the pre-norm fix
            # can still be loaded without missing-key errors.
            if self.norm_type in ["bn", "batchnorm", "batch_norm"]:
                self.norm1e = nn.BatchNorm1d(edge_in_dim)
                self.norm2e = nn.BatchNorm1d(edge_in_dim)
            elif self.norm_type in ["ln", "layernorm", "layer_norm"]:
                self.norm1e = nn.LayerNorm(edge_in_dim)
                self.norm2e = nn.LayerNorm(edge_in_dim)
            else:
                raise ValueError(f"Unknown norm type: {norm}")
        else:
            # No edge features
            self.WE_logits = self.register_parameter("WE_logits", None)
            self.WE_value = self.register_parameter("WE_value", None)
            self.WOe = self.register_parameter("WOe", None)
            self.ffn_e = self.register_parameter("ffn_e", None)
            self.norm1e = self.register_parameter("norm1e", None)
            self.norm2e = self.register_parameter("norm2e", None)

        # Node norms (pre-attention and pre-FFN)
        if self.norm_type in ["bn", "batchnorm", "batch_norm"]:
            self.norm1 = nn.BatchNorm1d(node_in_dim)  # pre-attn
            self.norm2 = nn.BatchNorm1d(node_in_dim)  # pre-FFN
        elif self.norm_type in ["ln", "layernorm", "layer_norm"]:
            self.norm1 = nn.LayerNorm(node_in_dim)    # pre-attn
            self.norm2 = nn.LayerNorm(node_in_dim)    # pre-FFN
        else:
            raise ValueError(f"Unknown norm type: {norm}")

        # Gating (optional)
        if gate:
            # Node gate: gate values per head & channel
            self.n_gate = nn.Linear(node_in_dim, hidden_dim, bias=True)
            # Edge gate: gate attention logits per head (scalar per head)
            if edge_in_dim is not None:
                self.e_gate = nn.Linear(edge_in_dim, num_heads, bias=True)
            else:
                self.e_gate = self.register_parameter("e_gate", None)
        else:
            self.n_gate = self.register_parameter("n_gate", None)
            self.e_gate = self.register_parameter("e_gate", None)

        # Dropouts
        self.dropout_layer = nn.Dropout(p=dropout)
        self.attn_dropout = nn.Dropout(p=dropout)

        # Stronger node FFN: 2–4x expansion
        node_ffn_hidden = max(hidden_dim, 4 * node_in_dim)
        self.ffn = MLP(
            input_dim=node_in_dim,
            output_dim=node_in_dim,
            hidden_dims=node_ffn_hidden,
            num_hidden_layers=2,
            dropout=dropout,
            act=act,
        )

        # Buffer to hold edge representations from message() for edge update
        self._eij = None

        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize all learnable parameters.
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

        # Edge-related
        if self.edge_in_dim is not None:
            nn.init.xavier_uniform_(self.WE_logits.weight)
            if self.WE_logits.bias is not None:
                nn.init.zeros_(self.WE_logits.bias)

            nn.init.xavier_uniform_(self.WE_value.weight)
            if self.WE_value.bias is not None:
                nn.init.zeros_(self.WE_value.bias)

            nn.init.xavier_uniform_(self.WOe.weight)
            if self.WOe.bias is not None:
                nn.init.zeros_(self.WOe.bias)

        # Gates
        if self.gate and self.n_gate is not None:
            nn.init.xavier_uniform_(self.n_gate.weight)
            if self.n_gate.bias is not None:
                nn.init.zeros_(self.n_gate.bias)

        if self.gate and self.e_gate is not None and not isinstance(self.e_gate, torch.nn.Parameter):
            nn.init.xavier_uniform_(self.e_gate.weight)
            if self.e_gate.bias is not None:
                nn.init.zeros_(self.e_gate.bias)

        # Node norms
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

        # Edge norms
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

        # FFNs
        if hasattr(self.ffn, "reset_parameters"):
            self.ffn.reset_parameters()
        if self.edge_in_dim is not None and hasattr(self.ffn_e, "reset_parameters"):
            self.ffn_e.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None):
        """
        Args:
            x: [N, node_in_dim]
            edge_index: [2, E]
            edge_attr: [E, edge_in_dim] or None

        Returns:
            updated_x: [N, node_in_dim]
            updated_edge_attr (or None): [E, edge_in_dim]
        """
        x_res = x
        edge_res = edge_attr

        # ---- Pre-norm for attention (nodes) ----
        x_norm = self.norm1(x_res)

        Q = self.WQ(x_norm).view(-1, self.num_heads, self.head_dim)
        K = self.WK(x_norm).view(-1, self.num_heads, self.head_dim)
        V = self.WV(x_norm).view(-1, self.num_heads, self.head_dim)

        if self.gate and self.n_gate is not None:
            G = self.n_gate(x_norm).view(-1, self.num_heads, self.head_dim)
        else:
            G = None

        # Message passing / attention aggregation
        out = self.propagate(
            edge_index, Q=Q, K=K, V=V, G=G, edge_attr=edge_attr, size=None
        )
        out = out.view(-1, self.hidden_dim * self.num_aggrs)  # [N, hidden_dim * num_aggrs]

        # ---- Node attention block output + residual ----
        attn_out = self.WO(out)
        attn_out = self.dropout_layer(attn_out)
        x1 = x_res + attn_out

        # ---- Pre-FFN norm (nodes) ----
        x1_norm = self.norm2(x1)
        ffn_out = self.ffn(x1_norm)
        ffn_out = self.dropout_layer(ffn_out)
        x_out = x1 + ffn_out  # final node output

        # ---- Edge updates (if present) ----
        if self.edge_in_dim is None or edge_attr is None or self._eij is None:
            edge_out = edge_attr
        else:
            # _eij has shape [E, H, Dh]; we use it as edge context
            e_context = self._eij.view(-1, self.hidden_dim)  # [E, hidden_dim]

            e_attn = self.WOe(e_context)  # [E, edge_in_dim]
            e_attn = self.dropout_layer(e_attn)

            e1 = edge_res + e_attn  # residual
            e1_norm = self.norm1e(e1)  # pre-norm before FFN
            e_ffn = self.ffn_e(e1_norm)
            e_ffn = self.dropout_layer(e_ffn)
            edge_out = e1 + e_ffn  # residual, no trailing norm (matches node path)

        # Clear buffer
        self._eij = None

        return x_out, edge_out

    def message(self, Q_i, K_j, V_j, G_j, index, edge_attr=None):
        """
        Compute messages on edges.

        Q_i: [E, H, Dh]
        K_j: [E, H, Dh]
        V_j: [E, H, Dh]
        G_j: [E, H, Dh] or None
        edge_attr: [E, edge_in_dim] or None

        Returns:
            Tensor: Attention-weighted values [E, H, Dh].
        """
        Dh = self.head_dim

        # Base logits from QK
        logits_vec = (Q_i * K_j) / math.sqrt(Dh)  # [E, H, Dh]

        # Edge contributions
        if self.edge_in_dim is not None and edge_attr is not None:
            # Additive bias to attention logits (per head)
            E_bias = self.WE_logits(edge_attr)  # [E, H]
            # Edge contribution to values
            E_val = self.WE_value(edge_attr).view(-1, self.num_heads, Dh)
            V_j = V_j + E_val

            # Optionally store an edge representation based on Q,K,E etc.
            self._eij = logits_vec
        else:
            E_bias = 0.0
            self._eij = logits_vec

        # Gating on values (nodes)
        if self.gate and G_j is not None:
            V_j = V_j * torch.sigmoid(G_j)

        # Combine logits and biases
        logits = logits_vec.sum(dim=-1)  # [E, H]
        if isinstance(E_bias, torch.Tensor):
            logits = logits + E_bias  # [E, H]

        # Optional edge-dependent gating on logits
        if self.gate and self.e_gate is not None and edge_attr is not None:
            # Gate per-head logits
            e_gate = self.e_gate(edge_attr)  # [E, H]
            logits = logits * torch.sigmoid(e_gate)

        # Attention weights
        alpha = softmax(logits, index)  # [E, H]
        alpha = self.attn_dropout(alpha)

        return alpha.view(-1, self.num_heads, 1) * V_j

    def __repr__(self) -> str:
        aggrs = ",".join(self.aggregators)
        return (
            f"{self.__class__.__name__}({self.node_in_dim}, "
            f"{self.hidden_dim}, heads={self.num_heads}, "
            f"aggrs: {aggrs}, "
            f"qkv_bias: {self.qkv_bias}, "
            f"gate: {self.gate}, "
            f"norm: {self.norm_type})"
        )

