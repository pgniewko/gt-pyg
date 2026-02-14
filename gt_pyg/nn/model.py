import logging
from typing import Optional, List, Tuple, Union, Dict, Any
from pathlib import Path

import torch

logger = logging.getLogger(__name__)
from torch import nn, Tensor
from torch_geometric.data import Batch
from torch_geometric.nn.aggr import MultiAggregation

from .gt_conv import GTConv
from .mlp import MLP


class GraphTransformerNet(nn.Module):
    """Graph Transformer Network with variational (Gaussian) readout.

    Embeds node and (optional) edge features, processes them through a stack
    of GTConv layers, pools to a graph-level representation, and produces
    per-task predictions via two MLP heads:

    * ``mu_mlp``      -- predicts the mean of a Gaussian.
    * ``log_var_mlp`` -- predicts the log-variance (clamped to [-10, 10]).

    During **training** the forward pass samples from the predicted Gaussian
    using the reparameterization trick::

        prediction = mu + std * epsilon,   epsilon ~ N(0, 1)

    This enables gradient-based optimization of probabilistic objectives
    (e.g. Gaussian NLL loss).  Set ``zero_var=True`` to disable sampling
    and return the deterministic mean instead.

    During **evaluation** the forward pass always returns the deterministic
    mean.  ``log_var`` is always returned for loss computation or uncertainty
    estimation.

    Reference:
        A Generalization of Transformer Networks to Graphs
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
        gt_aggregators: Optional[List[str]] = None,
        aggregators: Optional[List[str]] = None,
        act: str = "gelu",
        dropout: float = 0.1,
        num_tasks: int = 1,
    ) -> None:
        """Initialize the Graph Transformer network."""
        super().__init__()

        if gt_aggregators is None:
            gt_aggregators = ["sum"]
        if aggregators is None:
            aggregators = ["sum"]

        # Store config for checkpointing
        self._config = {
            "node_dim_in": node_dim_in,
            "edge_dim_in": edge_dim_in,
            "hidden_dim": hidden_dim,
            "norm": norm,
            "gate": gate,
            "qkv_bias": qkv_bias,
            "num_gt_layers": num_gt_layers,
            "num_heads": num_heads,
            "gt_aggregators": list(gt_aggregators),
            "aggregators": list(aggregators),
            "act": act,
            "dropout": dropout,
            "num_tasks": num_tasks,
        }

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

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"hidden_dim={self.hidden_dim}, "
            f"num_gt_layers={len(self.gt_layers)}, "
            f"num_tasks={self.num_tasks}, "
            f"norm={self.norm_type}, "
            f"params={self.num_parameters():,})"
        )

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
        """Forward pass with variational (reparameterization) sampling.

        In training mode (``zero_var=False``), predictions are stochastic::

            pred = mu + exp(0.5 * log_var) * epsilon,  epsilon ~ N(0, 1)

        In eval mode (or ``zero_var=True``), predictions are the
        deterministic mean ``mu``.  ``log_var`` is always returned.

        Args:
            x: Node features ``[num_nodes, node_dim_in]``.
            edge_index: Edge indices ``[2, num_edges]``.
            edge_attr: Edge features ``[num_edges, edge_dim_in]``.
                Required if ``edge_dim_in`` was set.
            batch: ``Batch`` object or batch-index tensor ``[num_nodes]``.
            zero_var: If True, skip sampling even during training.

        Returns:
            (prediction, log_var) — both ``[batch_size, num_tasks]``.
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


    def _get_component_modules(self, name: str) -> List[nn.Module]:
        """Map component name to list of modules."""
        embeddings = [self.node_emb] + ([self.edge_emb] if self.edge_emb else [])
        encoder = [self.input_norm, self.input_dropout] + list(self.gt_layers)
        heads = [self.readout_norm, self.readout_dropout, self.mu_mlp, self.log_var_mlp]
        pooling = [self.global_pool]

        components = {
            "embeddings": embeddings,
            "encoder": encoder,
            "gt_layers": list(self.gt_layers),
            "heads": heads,
            "pooling": pooling,
            "all": embeddings + encoder + heads + pooling,
        }
        # Handle gt_layer_{i}
        if name.startswith("gt_layer_"):
            idx = int(name.split("_")[-1])
            if idx < 0 or idx >= len(self.gt_layers):
                raise ValueError(f"Invalid layer index: {idx}. Model has {len(self.gt_layers)} layers.")
            return [self.gt_layers[idx]]

        if name not in components:
            raise ValueError(f"Unknown component: '{name}'. Valid: {sorted(components.keys())}")
        return components[name]

    def _set_requires_grad(self, modules: List[nn.Module], requires_grad: bool) -> None:
        """Set requires_grad for all params in modules. Handles BatchNorm eval mode."""
        for module in modules:
            for param in module.parameters():
                param.requires_grad = requires_grad
            # Set BatchNorm to eval mode when freezing
            for m in module.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    if requires_grad:
                        m.train()
                    else:
                        m.eval()

    def freeze(
        self,
        components: Optional[Union[str, List[str]]] = None,
        exclude: Optional[Union[str, List[str]]] = None,
    ) -> "GraphTransformerNet":
        """
        Freeze model components (set requires_grad=False).

        Args:
            components: Component(s) to freeze. None or "all" freezes everything.
                Options: "embeddings", "encoder", "gt_layers", "gt_layer_0", ..., "heads", "pooling", "all"
            exclude: Component(s) to exclude from freezing.

        Returns:
            self for method chaining.
        """
        if components is None:
            components = ["all"]
        elif isinstance(components, str):
            components = [components]

        if exclude is None:
            exclude = []
        elif isinstance(exclude, str):
            exclude = [exclude]

        # Get modules to freeze
        to_freeze = set()
        for comp in components:
            for m in self._get_component_modules(comp):
                to_freeze.add(m)

        # Remove excluded modules
        for comp in exclude:
            for m in self._get_component_modules(comp):
                to_freeze.discard(m)

        self._set_requires_grad(list(to_freeze), requires_grad=False)
        return self

    def unfreeze(
        self,
        components: Optional[Union[str, List[str]]] = None,
    ) -> "GraphTransformerNet":
        """
        Unfreeze model components (set requires_grad=True).

        Args:
            components: Component(s) to unfreeze. None or "all" unfreezes everything.

        Returns:
            self for method chaining.
        """
        if components is None:
            components = ["all"]
        elif isinstance(components, str):
            components = [components]

        to_unfreeze = []
        for comp in components:
            to_unfreeze.extend(self._get_component_modules(comp))

        self._set_requires_grad(to_unfreeze, requires_grad=True)
        return self

    def get_frozen_status(self) -> Dict[str, Optional[bool]]:
        """
        Get freeze status for each component group.

        Returns:
            Dict mapping component name to True if all params are frozen,
            False if any param is trainable, or None if the component has
            no learnable parameters.
        """
        status: Dict[str, Optional[bool]] = {}
        for name in ["embeddings", "encoder", "gt_layers", "heads", "pooling"]:
            modules = self._get_component_modules(name)
            params = [p for m in modules for p in m.parameters()]
            if not params:
                status[name] = None
            else:
                status[name] = all(not p.requires_grad for p in params)
        return status


    def get_config(self) -> Dict[str, Any]:
        """Return model config for reconstruction."""
        return dict(self._config)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "GraphTransformerNet":
        """Create model from config dict."""
        return cls(**config)

    def save_checkpoint(
        self,
        path: Union[str, Path],
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch: Optional[int] = None,
        global_step: Optional[int] = None,
        best_metric: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Save checkpoint to disk.

        Args:
            path: File path (.pt extension added if missing).
            optimizer: Optimizer to save state.
            scheduler: LR scheduler to save state.
            epoch: Current epoch number.
            global_step: Global training step.
            best_metric: Best validation metric.
            extra: Additional user data.
        """
        from .checkpoint import save_checkpoint

        merged_extra = {"frozen_status": self.get_frozen_status()}
        if extra:
            merged_extra.update(extra)

        save_checkpoint(
            model=self,
            path=path,
            config=self.get_config(),
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            global_step=global_step,
            best_metric=best_metric,
            extra=merged_extra,
        )

    @classmethod
    def load_checkpoint(
        cls,
        path: Union[str, Path],
        map_location: Optional[Union[str, torch.device]] = None,
        strict: bool = True,
    ) -> Tuple["GraphTransformerNet", Dict[str, Any]]:
        """
        Load model from checkpoint.

        Args:
            path: Checkpoint file path.
            map_location: Device mapping (e.g., "cpu", "cuda:0").
            strict: Enforce state_dict key matching.

        Returns:
            Tuple of (model, checkpoint_dict).
        """
        from .checkpoint import load_checkpoint

        checkpoint = load_checkpoint(path, map_location=map_location)
        model = cls.from_config(checkpoint["model_config"])
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        return model, checkpoint

    def load_weights(
        self,
        path: Union[str, Path],
        map_location: Optional[Union[str, torch.device]] = None,
        strict: bool = True,
    ) -> None:
        """
        Load weights from checkpoint into this model instance.

        Args:
            path: Checkpoint file path.
            map_location: Device mapping.
            strict: Enforce state_dict key matching (set False for transfer learning).
        """
        from .checkpoint import load_checkpoint

        checkpoint = load_checkpoint(path, map_location=map_location)

        if "model_config" in checkpoint:
            saved_config = checkpoint["model_config"]
            current_config = self.get_config()
            if saved_config != current_config:
                logger.warning(
                    f"Architecture mismatch between checkpoint and model. "
                    f"Saved: {saved_config}, Current: {current_config}"
                )

        self.load_state_dict(checkpoint["model_state_dict"], strict=strict)

