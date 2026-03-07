import torch
import torch.nn as nn


class MLP(nn.Module):
    """Fully-connected network with configurable depth and width.

    Args:
        input_dim:   Number of input features.
        hidden_dims: List of hidden layer sizes, e.g. [128, 64].
        output_dim:  Number of output units (1 for binary classification).
        dropout:     Dropout probability applied after each hidden layer.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        dims = [input_dim] + hidden_dims
        layers: list[nn.Module] = []
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(in_d, out_d), nn.ReLU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(dims[-1], output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
