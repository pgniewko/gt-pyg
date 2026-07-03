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
from .utils import make_norm, validate_aggregators, validate_dropout


class GTConv(MessagePassing):
    def __init__(
        self,
        hidden_dim: int,
        edge_in_dim: Optional[int] = None,
        num_heads: int = 8,
        gate: bool = False,
        qkv_bias: bool = False,
        dropout: float = 0.1,
        norm: str = "ln",
        act: str = "gelu",
        aggregators: Optional[List[str]] = None,
    ):
        """
        Graph Transformer Convolution (GTConv) module.

        - Pre-norm residual blocks for attention and FFN.
        - Wider, deeper FFNs by default.
        - Edge features contribute both as additive attention bias and to values.
        - Optional sigmoid gating on attention values.

        Args:
            hidden_dim (int): Dimensionality of the node features and hidden
                representations (input and output of the layer).
            edge_in_dim (int, optional): Dimensionality of the input edge features.
            num_heads (int, optional): Number of attention heads. Defaults to ``8``.
            gate (bool, optional): Sigmoid-gate the attention values per head
                and channel. Defaults to ``False``.
            qkv_bias (bool, optional): Bias in the attention projections. Defaults to ``False``.
            dropout (float, optional): Dropout probability. Defaults to ``0.1``.
            norm (str, optional): ``"bn"`` or ``"ln"`` (BatchNorm/LayerNorm). Defaults to ``"ln"``.
            act (str, optional): Activation function name for FFNs. Defaults to ``"gelu"``.
            aggregators (List[str], optional): MultiAggregation methods. Defaults to ``["sum"]``.
        """
        if aggregators is None:
            aggregators = ["sum"]

        validate_dropout("dropout", dropout)
        validate_aggregators("aggregators", aggregators)

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

        self.edge_in_dim = edge_in_dim
        self.dropout_p = dropout
        self.norm_type = norm.lower()
        self.gate = gate
        self.qkv_bias = qkv_bias

        # Node projections
        self.WQ = nn.Linear(hidden_dim, hidden_dim, bias=qkv_bias)
        self.WK = nn.Linear(hidden_dim, hidden_dim, bias=qkv_bias)
        self.WV = nn.Linear(hidden_dim, hidden_dim, bias=qkv_bias)

        # Node output projection (after aggregation over heads & aggregators)
        self.WO = nn.Linear(hidden_dim * self.num_aggrs, hidden_dim, bias=True)

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

            self.norm0e = make_norm(norm, edge_in_dim)  # pre-attn (edges)
            self.norm1e = make_norm(norm, edge_in_dim)  # pre-FFN (edges)
        else:
            # No edge features
            self.WE_logits = self.WE_value = self.WOe = None
            self.ffn_e = self.norm0e = self.norm1e = None

        # Node norms (pre-attention and pre-FFN)
        self.norm1 = make_norm(norm, hidden_dim)  # pre-attn
        self.norm2 = make_norm(norm, hidden_dim)  # pre-FFN

        # Gating (optional): gate values per head & channel
        self.n_gate = nn.Linear(hidden_dim, hidden_dim, bias=True) if gate else None

        # Dropouts
        self.dropout_layer = nn.Dropout(p=dropout)
        self.attn_dropout = nn.Dropout(p=dropout)

        # Stronger node FFN: 4x expansion
        self.ffn = MLP(
            input_dim=hidden_dim,
            output_dim=hidden_dim,
            hidden_dims=4 * hidden_dim,
            num_hidden_layers=2,
            dropout=dropout,
            act=act,
        )

        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize all learnable parameters.

        Linear projections use Xavier uniform with zero bias; norms and FFNs
        are reset via their own ``reset_parameters``.
        """
        linears = [
            self.WQ, self.WK, self.WV, self.WO,
            self.WE_logits, self.WE_value, self.WOe,
            self.n_gate,
        ]
        for lin in linears:
            if lin is not None:
                nn.init.xavier_uniform_(lin.weight)
                if lin.bias is not None:
                    nn.init.zeros_(lin.bias)

        # Norms reset to weight=1, bias=0 (+ running stats for BatchNorm)
        for norm in [self.norm1, self.norm2, self.norm0e, self.norm1e]:
            if norm is not None:
                norm.reset_parameters()

        # FFNs
        self.ffn.reset_parameters()
        if self.ffn_e is not None:
            self.ffn_e.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None):
        """
        Args:
            x: Node features ``[N, hidden_dim]``.
            edge_index: COO edges ``[2, E]``.
            edge_attr: Edge features ``[E, edge_in_dim]``, or ``None``.

        Returns:
            Tuple of updated node features ``[N, hidden_dim]`` and updated edge
            features ``[E, edge_in_dim]`` (``None`` if the layer has no edges).
        """
        if self.edge_in_dim is not None and edge_attr is None:
            raise ValueError(
                "edge_in_dim was set in __init__, but 'edge_attr' is None in forward(). "
                "Pass edge features or set edge_in_dim=None."
            )

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

        # Pre-compute edge projections for use in message() and the edge
        # update. All edge contributions consume the pre-normed features,
        # consistent with the pre-norm design of the node path.
        if self.edge_in_dim is not None and edge_attr is not None:
            edge_attr_norm = self.norm0e(edge_attr)
            E_val = self.WE_value(edge_attr_norm).view(-1, self.num_heads, self.head_dim)
        else:
            edge_attr_norm = None
            E_val = None

        # Scaled Q.K products per edge, computed once and shared by the
        # attention logits (in message()) and the edge update.
        # In source_to_target flow: Q_i=target=edge_index[1], K_j=source=edge_index[0]
        src, dst = edge_index[0], edge_index[1]
        qk = (Q[dst] * K[src]) / math.sqrt(self.head_dim)  # [E, H, Dh]

        # Message passing / attention aggregation
        out = self.propagate(
            edge_index, qk=qk, V=V, G=G, edge_attr=edge_attr_norm,
            E_val=E_val, size=None,
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
        if self.edge_in_dim is None or edge_attr is None:
            edge_out = edge_attr
        else:
            # Edge representation from the shared Q.K products: qk * E_val
            eij = qk * E_val                           # [E, H, Dh]
            e_context = eij.view(-1, self.hidden_dim)  # [E, hidden_dim]
            e_attn = self.WOe(e_context)               # [E, edge_in_dim]
            e_attn = self.dropout_layer(e_attn)

            e1 = edge_res + e_attn  # residual
            e1_norm = self.norm1e(e1)  # pre-norm before FFN
            e_ffn = self.ffn_e(e1_norm)
            e_ffn = self.dropout_layer(e_ffn)
            edge_out = e1 + e_ffn

        return x_out, edge_out

    def message(self, V_j, G_j, qk, index, edge_attr=None, E_val=None):
        """
        Compute messages on edges.

        Args:
            V_j: Per-edge value vectors ``[E, H, Dh]``.
            G_j: Per-edge gate vectors ``[E, H, Dh]``, or ``None``.
            qk: Scaled Q.K products ``[E, H, Dh]`` (pre-computed in ``forward``).
            index: Target-node index used for the attention softmax.
            edge_attr: Pre-normed edge features ``[E, edge_in_dim]``, or ``None``.
            E_val: Edge value contribution ``[E, H, Dh]``, or ``None``.

        Returns:
            Tensor: Attention-weighted values ``[E, H, Dh]``.
        """
        # Edge contribution to values (pre-computed in forward())
        if E_val is not None:
            V_j = V_j + E_val

        # Gating on values (nodes)
        if self.gate and G_j is not None:
            V_j = V_j * torch.sigmoid(G_j)

        # Attention logits: QK product plus additive edge bias (per head)
        logits = qk.sum(dim=-1)  # [E, H]
        if edge_attr is not None:
            logits = logits + self.WE_logits(edge_attr)  # [E, H]

        # Attention weights
        alpha = softmax(logits, index)  # [E, H]
        alpha = self.attn_dropout(alpha)

        return alpha.view(-1, self.num_heads, 1) * V_j

    def __repr__(self) -> str:
        aggrs = ",".join(self.aggregators)
        return (
            f"{self.__class__.__name__}("
            f"{self.hidden_dim}, heads={self.num_heads}, "
            f"aggrs: {aggrs}, "
            f"qkv_bias: {self.qkv_bias}, "
            f"gate: {self.gate}, "
            f"norm: {self.norm_type})"
        )
