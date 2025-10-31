from typing import Optional, List, Tuple

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
        pe_in_dim: Optional[int] = None,
        hidden_dim: int = 128,
        norm: str = "bn",
        gate: bool = False,
        qkv_bias: bool = False,
        num_gt_layers: int = 4,
        num_heads: int = 8,
        gt_aggregators: List[str] = ["sum"],
        aggregators: List[str] = ["sum"],
        act: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        # Embeddings
        self.node_emb = nn.Linear(node_dim_in, hidden_dim, bias=False)

        self.edge_emb: Optional[nn.Linear]
        if edge_dim_in is not None:
            self.edge_emb = nn.Linear(edge_dim_in, hidden_dim, bias=False)
        else:
            self.edge_emb = None

        self.pe_emb: Optional[nn.Linear]
        if pe_in_dim is not None:
            self.pe_emb = nn.Linear(pe_in_dim, hidden_dim, bias=False)
        else:
            self.pe_emb = None

        # Graph Transformer layers
        self.gt_layers = nn.ModuleList(
            [
                GTConv(
                    node_in_dim=hidden_dim,
                    hidden_dim=hidden_dim,
                    edge_in_dim=hidden_dim,
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

        # Global pooling and heads
        self.global_pool = MultiAggregation(aggregators, mode="cat")
        num_aggrs = len(aggregators)
        head_in_dim = num_aggrs * hidden_dim

        self.mu_mlp = MLP(
            input_dim=head_in_dim,
            output_dim=1,
            hidden_dims=hidden_dim,
            num_hidden_layers=1,
            dropout=0.0,
            act=act,
        )
        self.log_var_mlp = MLP(
            input_dim=head_in_dim,
            output_dim=1,
            hidden_dims=hidden_dim,
            num_hidden_layers=1,
            dropout=0.0,
            act=act,
        )

        # Initialize everything
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Re-initialize parameters.

        Embedding layers use Xavier uniform (they're added directly, no nonlinearity).
        GTConv layers and MLP heads are reset via their own `reset_parameters` if present.
        """
        nn.init.xavier_uniform_(self.node_emb.weight)
        if self.edge_emb is not None:
            nn.init.xavier_uniform_(self.edge_emb.weight)
        if self.pe_emb is not None:
            nn.init.xavier_uniform_(self.pe_emb.weight)

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

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        pe: Tensor,
        batch: Batch,
        zero_var: bool = False,
    ):
        """
        Forward pass of the Graph Transformer Network.

        Args:
            x: Node features [num_nodes, node_dim_in].
            edge_index: Edge indices [2, num_edges].
            edge_attr: Edge features [num_edges, edge_dim_in] if provided.
            pe: Positional encodings [num_nodes, pe_in_dim] if provided.
            batch: Batch vector.
            zero_var (bool): If True, do NOT sample; return deterministic mu.
                             (Variance is still predicted and returned via log_var.)
        Returns:
            Tuple[Tensor, Tensor]: (prediction, log_var)
                - prediction: mu if zero_var=True or not training; otherwise a reparameterized sample
                - log_var: predicted log(variance)
        """
        x = self.node_emb(x)
        if self.pe_emb is not None:
            x = x + self.pe_emb(pe)
        if self.edge_emb is not None:
            edge_attr = self.edge_emb(edge_attr)

        for gt_layer in self.gt_layers:
            (x, edge_attr) = gt_layer(x=x, edge_index=edge_index, edge_attr=edge_attr)

        x = self.global_pool(x, batch)
        mu = self.mu_mlp(x)
        log_var = self.log_var_mlp(x)             # predict log(sigma^2)
        std = torch.exp(0.5 * log_var)            # sigma

        if self.training and not zero_var:
            eps = torch.randn_like(std)
            pred = mu + std * eps                 # sample
        else:
            pred = mu                             # deterministic mean

        return pred, log_var


