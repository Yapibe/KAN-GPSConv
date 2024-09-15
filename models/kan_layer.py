import torch
import torch.nn as nn
import torch.nn.functional as F

class KANLayer(nn.Module):
    """
    Kolmogorov-Arnold Network Layer.

    Args:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        grid_features (int): Size of the grid features (default: 100).
        use_bias (bool): If set to False, the layer will not learn an additive bias (default: True).
    """
    def __init__(self, in_features, out_features, grid_features=100, activation='sin', use_bias=True):
        super(KANLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_features = grid_features
        self.activation = activation

        self.linear_in = nn.Linear(in_features, grid_features, bias=use_bias)
        self.linear_out = nn.Linear(grid_features, out_features, bias=use_bias)

    def forward(self, input):
        x = self.linear_in(input)
        # Apply the selected activation function
        if self.activation == 'sin':
            x = torch.sin(x)
        elif self.activation == 'relu':
            x = F.relu(x)
        elif self.activation == 'tanh':
            x = torch.tanh(x)
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")
        output = self.linear_out(x)
        return output
