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
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.act = act
        self.act_kwargs = act_kwargs or {}
        self.num_hidden_layers = num_hidden_layers
        self.dropout_p = dropout

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

        layers: List[nn.Module] = []

        # Special case: no hidden layers -> just a single linear map
        if num_hidden_layers == 0:
            layers.append(nn.Linear(input_dim, output_dim, bias=True))
            self.mlp = nn.Sequential(*layers)
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
            layers.append(nn.Linear(i_dim, o_dim, bias=True))
            layers.append(_make_activation())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))

        # Output layer (no activation, no dropout)
        layers.append(nn.Linear(dims[-1], output_dim, bias=True))

        self.mlp = nn.Sequential(*layers)

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Re-initialize all Linear layer parameters.

        - Hidden Linear layers:
            * If activation is ReLU-like, use Kaiming uniform (fan_in).
            * Else, use Xavier uniform.
        - Output Linear layer uses Xavier uniform.
        - All biases are set to zero.
        """
        linear_layers = [m for m in self.mlp if isinstance(m, nn.Linear)]
        if not linear_layers:
            return

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

        # Initialize all but the last linear as "hidden"
        for lin in linear_layers[:-1]:
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
        out_lin = linear_layers[-1]
        nn.init.xavier_uniform_(out_lin.weight)
        if out_lin.bias is not None:
            nn.init.zeros_(out_lin.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the MLP module.

        Args:
            x (Tensor): Input tensor of shape [..., input_dim].

        Returns:
            Tensor: Output tensor of shape [..., output_dim].
        """
        return self.mlp(x)

