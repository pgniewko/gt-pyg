# Third party
from torch import nn
from torch import Tensor
from torch_geometric.nn.resolver import activation_resolver
from typing import List, Union, Optional, Dict, Any


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Union[int, List[int]],
        num_hidden_layers: int = 1,
        dropout: float = 0.0,
        act: str = "gelu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: bool = False,
        residual: bool = False,
    ):
        """
        Multi-Layer Perceptron (MLP) module.

        Args:
            input_dim (int): Dimensionality of the input features.
            output_dim (int): Dimensionality of the output features.
            hidden_dims (Union[int, List[int]]): Hidden layer dimensions.
                If int, the same hidden dimension is used for all hidden layers.
            num_hidden_layers (int, optional): Number of hidden layers. Default is 1.
                If 0, the MLP degenerates to a single Linear(input_dim, output_dim).
            dropout (float, optional): Dropout probability. Default is 0.0.
            act (str, optional): Activation function name. Default is "gelu".
            act_kwargs (Dict[str, Any], optional): Additional arguments for the
                activation function. Default is None.
            norm (bool, optional): If True, add LayerNorm after each hidden Linear
                (before activation). Not applied to the output layer. Default is False.
            residual (bool, optional): If True, add residual connections around
                hidden blocks where input and output dimensions match. Not applied
                to the output layer. Default is False.
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.act = act
        self.act_kwargs = act_kwargs or {}
        self.num_hidden_layers = num_hidden_layers
        self.dropout_p = dropout
        self.norm = norm
        self.residual = residual

        if num_hidden_layers < 0:
            raise ValueError(f"num_hidden_layers must be >= 0, got {num_hidden_layers}")

        # Normalize hidden_dims to a list
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims] * max(num_hidden_layers, 0)

        if num_hidden_layers > 0 and len(hidden_dims) != num_hidden_layers:
            raise ValueError(
                f"hidden_dims length ({len(hidden_dims)}) "
                f"must equal num_hidden_layers ({num_hidden_layers})"
            )

        self.blocks = nn.ModuleList()
        self._can_residual: List[bool] = []

        # Special case: no hidden layers -> just a single linear map
        if num_hidden_layers == 0:
            self.output_layer = nn.Linear(input_dim, output_dim, bias=True)
            self.reset_parameters()
            return

        # Hidden layers: [input_dim] + hidden_dims
        dims = [input_dim] + hidden_dims

        # Resolve activation module factory once
        # If act is None or "", use identity
        if self.act is None or str(self.act).lower() in {"", "none", "identity"}:
            def _make_activation():
                return nn.Identity()
        else:
            def _make_activation():
                return activation_resolver(self.act, **self.act_kwargs)

        for i_dim, o_dim in zip(dims[:-1], dims[1:]):
            block_layers: List[nn.Module] = []
            block_layers.append(nn.Linear(i_dim, o_dim, bias=True))
            if norm:
                block_layers.append(nn.LayerNorm(o_dim))
            block_layers.append(_make_activation())
            if dropout > 0.0:
                block_layers.append(nn.Dropout(p=dropout))
            self.blocks.append(nn.Sequential(*block_layers))
            self._can_residual.append(i_dim == o_dim)

        # Output layer (no activation, no dropout, no norm, no residual)
        self.output_layer = nn.Linear(dims[-1], output_dim, bias=True)

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Re-initialize all Linear layer parameters.

        - Hidden Linear layers:
            * If activation is ReLU-like, use Kaiming uniform (fan_in).
            * Else, use Xavier uniform.
        - Output Linear layer uses Xavier uniform.
        - LayerNorm layers: weight=1, bias=0.
        - All Linear biases are set to zero.
        """
        # Collect all Linear layers across blocks and output
        hidden_linears = []
        for block in self.blocks:
            for m in block:
                if isinstance(m, nn.Linear):
                    hidden_linears.append(m)

        act_lower = (self.act or "").lower()

        # Decide if we should use Kaiming for hidden layers
        relu_like = {"relu", "leaky_relu", "prelu", "rrelu"}
        use_kaiming = act_lower in relu_like

        # Negative slope for leaky_relu-like activations
        negative_slope = 0.0
        if act_lower == "leaky_relu":
            # Standard PyTorch default is 0.01 if not specified
            negative_slope = float(self.act_kwargs.get("negative_slope", 0.01))

        nonlinearity = "relu" if act_lower != "leaky_relu" else "leaky_relu"

        # Initialize hidden linear layers
        for lin in hidden_linears:
            if use_kaiming:
                nn.init.kaiming_uniform_(
                    lin.weight,
                    a=negative_slope,
                    nonlinearity=nonlinearity,
                )
            else:
                nn.init.xavier_uniform_(lin.weight)
            if lin.bias is not None:
                nn.init.zeros_(lin.bias)

        # Initialize output layer with Xavier
        nn.init.xavier_uniform_(self.output_layer.weight)
        if self.output_layer.bias is not None:
            nn.init.zeros_(self.output_layer.bias)

        # Initialize LayerNorm layers
        for block in self.blocks:
            for m in block:
                if isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the MLP module.

        Args:
            x (Tensor): Input tensor of shape [..., input_dim].

        Returns:
            Tensor: Output tensor of shape [..., output_dim].
        """
        for i, block in enumerate(self.blocks):
            if self.residual and self._can_residual[i]:
                x = x + block(x)
            else:
                x = block(x)
        return self.output_layer(x)
