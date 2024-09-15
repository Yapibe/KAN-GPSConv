import torch
import torch.nn as nn
import torch.nn.functional as F
from efficient_kan import KANLinear  # Corrected import

class KANLayer(KANLinear):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        activation='silu',
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        if activation == 'silu':
            base_activation = torch.nn.SiLU
        elif activation == 'relu':
            base_activation = torch.nn.ReLU
        elif activation == 'tanh':
            base_activation = torch.nn.Tanh
        elif activation == 'leaky_relu':
            base_activation = torch.nn.LeakyReLU
        elif activation == 'elu':
            base_activation = torch.nn.ELU
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        super(KANLayer, self).__init__(
            in_features,
            out_features,
            grid_size,
            spline_order,
            scale_noise,
            scale_base,
            scale_spline,
            enable_standalone_scale_spline,
            base_activation,
            grid_eps,
            grid_range,
        )

class MultiKANLayer(nn.Module):
    """
    Multi-function Kolmogorov-Arnold Network Layer.

    Args:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        num_functions (int): Number of univariate functions to use (default: 4).
        grid_features (int): Size of the grid features for each function (default: 25).
        activation (str): Activation function to use ('sin', 'relu', 'tanh', 'leaky_relu', 'elu') (default: 'sin').
        use_bias (bool): If set to False, the layer will not learn an additive bias (default: True).
    """
    def __init__(self, in_features, out_features, num_functions=4, grid_features=25, activation='sin', use_bias=True):
        super(MultiKANLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_functions = num_functions
        self.grid_features = grid_features
        self.activation = activation

        self.linear_in = nn.ModuleList([
            nn.Linear(in_features, grid_features, bias=use_bias) for _ in range(num_functions)
        ])
        self.linear_out = nn.Linear(num_functions * grid_features, out_features, bias=use_bias)

    def forward(self, input):
        """
        Forward pass of the Multi-KAN layer.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the Multi-KAN layer.
        """
        outputs = []
        for i in range(self.num_functions):
            x = self.linear_in[i](input)
            # Apply the selected activation function
            if self.activation == 'sin':
                x = torch.sin(x)
            elif self.activation == 'relu':
                x = F.relu(x)
            elif self.activation == 'tanh':
                x = torch.tanh(x)
            elif self.activation == 'leaky_relu':
                x = F.leaky_relu(x)
            elif self.activation == 'elu':
                x = F.elu(x)
            else:
                raise ValueError(f"Unsupported activation function: {self.activation}")
            outputs.append(x)
        output = torch.cat(outputs, dim=-1)
        output = self.linear_out(output)
        return output
