from typing import Optional, List, Tuple, Union

import torch
from torch import nn, Tensor
from torch_geometric.data import Batch
from torch_geometric.nn.aggr import MultiAggregation

from .gt_conv import GTConv
from .mlp import MLP


class GraphTransformerNet(nn.Module):
    """
    Graph Transformer Network.

    Reference:
      1. A Generalization of Transformer Networks to Graphs
         https://arxiv.org/abs/2012.09699
    """

    def __init__(
        self,
        node_dim_in: int,
        edge_dim_in: Optional[int] = None,
        hidden_dim: int = 128,
        norm: str = "ln",
        gate: bool = False,
        qkv_bias: bool = False,
        num_gt_layers: int = 4,
        num_heads: int = 8,
        gt_aggregators: List[str] = ["sum"],
        aggregators: List[str] = ["sum"],
        act: str = "gelu",
        dropout: float = 0.1,
        num_tasks: int = 1,
    ) -> None:
        super().__init__()

        if num_tasks <= 0:
            raise ValueError("num_tasks must be >= 1")
        self.num_tasks = int(num_tasks)

        self.hidden_dim = hidden_dim
        self.norm_type = norm.lower()
        self.act = act
        self.dropout_p = dropout

        # ---- Embeddings ----
        # Node embedding
        self.node_emb = nn.Linear(node_dim_in, hidden_dim, bias=False)

        # Edge embedding (optional)
        if edge_dim_in is not None:
            self.edge_emb: Optional[nn.Module] = nn.Linear(
                edge_dim_in, hidden_dim, bias=False
            )
            edge_in_dim_hidden: Optional[int] = hidden_dim
        else:
            self.edge_emb = None
            edge_in_dim_hidden = None  # GTConv will be instantiated without edge features

        # Input norm & dropout
        if self.norm_type in ["bn", "batchnorm", "batch_norm"]:
            self.input_norm = nn.BatchNorm1d(hidden_dim)
        elif self.norm_type in ["ln", "layernorm", "layer_norm"]:
            self.input_norm = nn.LayerNorm(hidden_dim)
        else:
            raise ValueError(f"Unknown norm type: {norm}")

        self.input_dropout = nn.Dropout(p=dropout)

        # ---- Graph Transformer layers ----
        self.gt_layers = nn.ModuleList(
            [
                GTConv(
                    node_in_dim=hidden_dim,
                    hidden_dim=hidden_dim,
                    edge_in_dim=edge_in_dim_hidden,
                    num_heads=num_heads,
                    act=act,
                    dropout=dropout,
                    norm=norm,
                    gate=gate,
                    qkv_bias=qkv_bias,
                    aggregators=gt_aggregators,
                )
                for _ in range(num_gt_layers)
            ]
        )

        # ---- Global pooling and readout ----
        self.global_pool = MultiAggregation(aggregators, mode="cat")
        self.num_aggrs = len(aggregators)
        head_in_dim = self.num_aggrs * hidden_dim

        # Readout norm & dropout before heads
        if self.norm_type in ["bn", "batchnorm", "batch_norm"]:
            self.readout_norm = nn.BatchNorm1d(head_in_dim)
        elif self.norm_type in ["ln", "layernorm", "layer_norm"]:
            self.readout_norm = nn.LayerNorm(head_in_dim)
        else:
            raise ValueError(f"Unknown norm type: {norm}")

        self.readout_dropout = nn.Dropout(p=dropout)

        # Slightly stronger heads (still modest by default)
        head_hidden_dim = hidden_dim

        self.mu_mlp = MLP(
            input_dim=head_in_dim,
            output_dim=self.num_tasks,
            hidden_dims=head_hidden_dim,
            num_hidden_layers=1,
            dropout=dropout,
            act=act,
        )
        self.log_var_mlp = MLP(
            input_dim=head_in_dim,
            output_dim=self.num_tasks,
            hidden_dims=head_hidden_dim,
            num_hidden_layers=1,
            dropout=dropout,
            act=act,
        )

        # Initialize everything
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Re-initialize parameters.

        Embedding layers use Xavier uniform (they're added directly, no nonlinearity).
        GTConv layers, norms, and MLP heads are reset via their own `reset_parameters`.
        """
        # Embeddings
        nn.init.xavier_uniform_(self.node_emb.weight)
        if self.edge_emb is not None:
            nn.init.xavier_uniform_(self.edge_emb.weight)

        # Norms
        if isinstance(self.input_norm, nn.BatchNorm1d):
            self.input_norm.reset_running_stats()
            nn.init.ones_(self.input_norm.weight)
            nn.init.zeros_(self.input_norm.bias)
        elif isinstance(self.input_norm, nn.LayerNorm):
            nn.init.ones_(self.input_norm.weight)
            nn.init.zeros_(self.input_norm.bias)

        if isinstance(self.readout_norm, nn.BatchNorm1d):
            self.readout_norm.reset_running_stats()
            nn.init.ones_(self.readout_norm.weight)
            nn.init.zeros_(self.readout_norm.bias)
        elif isinstance(self.readout_norm, nn.LayerNorm):
            nn.init.ones_(self.readout_norm.weight)
            nn.init.zeros_(self.readout_norm.bias)

        # Propagate to children with their own reset
        for m in self.gt_layers:
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()

        if hasattr(self.mu_mlp, "reset_parameters"):
            self.mu_mlp.reset_parameters()
        if hasattr(self.log_var_mlp, "reset_parameters"):
            self.log_var_mlp.reset_parameters()

    @torch.no_grad()
    def num_parameters(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _get_batch_index(self, batch: Union[Batch, Tensor]) -> Tensor:
        """
        Support both passing a `Batch` object or a batch index tensor.
        """
        if isinstance(batch, Batch):
            return batch.batch
        return batch

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor],
        batch: Union[Batch, Tensor],
        zero_var: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass of the Graph Transformer Network.

        Args:
            x: Node features [num_nodes, node_dim_in].
            edge_index: Edge indices [2, num_edges].
            edge_attr: Edge features [num_edges, edge_dim_in] if provided
                       (required if edge_dim_in was set).
            batch: Either a `Batch` object or a batch index tensor of shape [num_nodes].
            zero_var (bool): If True, do NOT sample; return deterministic mu.
                             (Variance is still predicted and returned via log_var.)

        Returns:
            Tuple[Tensor, Tensor]: (prediction, log_var)
                - prediction: [batch_size, num_tasks]
                - log_var:    [batch_size, num_tasks]
        """
        # Node embedding
        h = self.node_emb(x)  # [N, H]

        # Input norm + dropout
        h = self.input_norm(h)
        h = self.input_dropout(h)

        # Edge embedding (if present)
        if self.edge_emb is not None:
            if edge_attr is None:
                raise ValueError(
                    "edge_dim_in was set in __init__, but 'edge_attr' is None in forward()."
                )
            e = self.edge_emb(edge_attr)
        else:
            e = None

        # Graph Transformer layers
        for gt_layer in self.gt_layers:
            h, e = gt_layer(x=h, edge_index=edge_index, edge_attr=e)

        # Global pooling
        batch_index = self._get_batch_index(batch)
        g = self.global_pool(h, batch_index)  # [B, num_aggrs * H]

        # Readout norm + dropout
        g = self.readout_norm(g)
        g = self.readout_dropout(g)

        # Heads
        mu = self.mu_mlp(g)            # [B, T]
        log_var = self.log_var_mlp(g)  # [B, T]

        # Numerical stability: clamp log_var range a bit
        log_var = torch.clamp(log_var, min=-10.0, max=10.0)
        std = torch.exp(0.5 * log_var)  # [B, T]

        if self.training and not zero_var:
            eps = torch.randn_like(std)
            pred = mu + std * eps       # reparameterized sample
        else:
            pred = mu                   # deterministic mean

        return pred, log_var

