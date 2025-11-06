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
        act: str = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Multi-Layer Perceptron (MLP) module.

        Args:
            input_dim (int): Dimensionality of the input features.
            output_dim (int): Dimensionality of the output features.
            hidden_dims (Union[int, List[int]]): Hidden layer dimensions.
                If int, same hidden dimension is used for all layers.
            num_hidden_layers (int, optional): Number of hidden layers. Default is 1.
            dropout (float, optional): Dropout probability. Default is 0.0.
            act (str, optional): Activation function name. Default is "relu".
            act_kwargs (Dict[str, Any], optional): Additional arguments for the activation function.
                                                   Default is None.
        """
        super().__init__()
        self.act = act
        self.act_kwargs = act_kwargs or {}

        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims] * num_hidden_layers
        assert len(hidden_dims) == num_hidden_layers

        dims = [input_dim] + hidden_dims
        layers: List[nn.Module] = []

        for i_dim, o_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(i_dim, o_dim, bias=True))
            layers.append(activation_resolver(self.act, **self.act_kwargs))
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))

        # Output layer
        layers.append(nn.Linear(dims[-1], output_dim, bias=True))

        self.mlp = nn.Sequential(*layers)

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Re-initialize all Linear layer parameters.

        - Hidden Linear layers use Kaiming uniform when activation is ReLU/LeakyReLU.
        - Output Linear layer uses Xavier uniform.
        - All biases are set to zero.
        """
        # Gather linear layers in order
        linear_layers = [m for m in self.mlp if isinstance(m, nn.Linear)]
        if not linear_layers:
            return

        # Determine if we should use Kaiming (fan_in, nonlinearity='relu') for hidden layers
        act_lower = (self.act or "").lower()
        use_kaiming = act_lower in {"relu", "leaky_relu", "prelu", "rrelu"}

        # Negative slope for leaky_relu if provided
        negative_slope = 0.0
        if act_lower == "leaky_relu":
            negative_slope = float(self.act_kwargs.get("negative_slope", self.act_kwargs.get("inplace", 0.0))) \
                             if "negative_slope" in self.act_kwargs or "inplace" in self.act_kwargs else 0.01

        # Initialize hidden linear layers
        for lin in linear_layers[:-1]:
            if use_kaiming:
                nn.init.kaiming_uniform_(lin.weight, a=negative_slope, nonlinearity="leaky_relu" if negative_slope > 0 else "relu")
            else:
                nn.init.xavier_uniform_(lin.weight)
            if lin.bias is not None:
                nn.init.zeros_(lin.bias)

        # Initialize output layer
        out_lin = linear_layers[-1]
        nn.init.xavier_uniform_(out_lin.weight)
        if out_lin.bias is not None:
            nn.init.zeros_(out_lin.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the MLP module.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        return self.mlp(x)

