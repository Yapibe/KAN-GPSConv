"""
This module contains the implementation of KAN-GPS models.

The models combine Kolmogorov-Arnold Network (KAN) layers with
Graph Positional Signatures (GPS) to enhance graph neural networks.
"""

import torch
import torch.nn as nn
from .kan_layer import KANLayer, MultiKANLayer
from .gps_layer import GPSLayer


class KANGPSBase(nn.Module):
    """
    Base class for KAN-GPS models.

    This class provides the foundation for combining KAN and GPS layers
    in a graph neural network architecture.

    Args:
        in_channels (int): Number of input features.
        hidden_channels (int): Number of hidden features.
        out_channels (int): Number of output features.
        num_layers (int): Number of KAN-GPS layers in the model.
        pos_dim (int, optional): Dimension of positional encoding. Defaults to 64.
    """

    def __init__(
        self, in_channels, hidden_channels, out_channels, num_layers, pos_dim=64
    ):
        super(KANGPSBase, self).__init__()
        self.num_layers = num_layers

        # Initial linear projection
        self.initial_linear = nn.Linear(in_channels, hidden_channels)

        # KAN layers
        self.kan_layers = nn.ModuleList(
            [KANLayer(hidden_channels, hidden_channels) for _ in range(num_layers)]
        )

        # GPS layers
        self.gps_layers = nn.ModuleList(
            [GPSLayer(hidden_channels, hidden_channels) for _ in range(num_layers)]
        )

        # Final linear projection
        self.final_linear = nn.Linear(hidden_channels, out_channels)

        self.pos_dim = pos_dim

    def forward(self, x, edge_index, pos_encoding=None):
        """
        Forward pass of the KAN-GPS base model.

        Args:
            x (torch.Tensor): Input node features.
            edge_index (torch.Tensor): Edge indices.
            pos_encoding (torch.Tensor, optional): Positional encoding of nodes.

        Returns:
            torch.Tensor: Output node features after applying KAN and GPS layers.
        """
        # Initialize positional encoding if not provided
        if pos_encoding is None:
            pos_encoding = torch.zeros(x.size(0), self.pos_dim, device=x.device)

        # Initial linear projection
        x = self.initial_linear(x)

        # Apply KAN and GPS layers sequentially
        for kan_layer, gps_layer in zip(self.kan_layers, self.gps_layers):
            x = kan_layer(x)
            x = gps_layer(x, edge_index, pos_encoding)
            x = torch.relu(x)  # Non-linearity between layers

        # Final linear projection
        x = self.final_linear(x)
        return x


class HybridKANGPS(KANGPSBase):
    """
    Hybrid KAN-GPS model.

    This model uses standard KAN layers in combination with GPS layers.

    Args:
        in_channels (int): Number of input features.
        hidden_channels (int): Number of hidden features.
        out_channels (int): Number of output features.
        num_layers (int): Number of KAN-GPS layers in the model.
        pos_dim (int, optional): Dimension of positional encoding. Defaults to 64.
    """

    def __init__(
        self, in_channels, hidden_channels, out_channels, num_layers, pos_dim=64
    ):
        super(HybridKANGPS, self).__init__(
            in_channels, hidden_channels, out_channels, num_layers, pos_dim
        )

        # Override KAN layers with standard KAN implementation
        self.kan_layers = nn.ModuleList(
            [KANLayer(hidden_channels, hidden_channels) for _ in range(num_layers)]
        )


class KANGPS(KANGPSBase):
    """
    Full KAN-GPS model.

    This model uses MultiKAN layers for both node feature transformation
    and message passing in combination with GPS layers.

    Args:
        in_channels (int): Number of input features.
        hidden_channels (int): Number of hidden features.
        out_channels (int): Number of output features.
        num_layers (int): Number of KAN-GPS layers in the model.
        pos_dim (int, optional): Dimension of positional encoding. Defaults to 64.
        num_functions (int, optional): Number of univariate functions in MultiKANLayer. Defaults to 4.
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers,
        pos_dim=64,
        num_functions=4,
    ):
        super(KANGPS, self).__init__(
            in_channels, hidden_channels, out_channels, num_layers, pos_dim
        )

        # MultiKAN layers for node feature transformation (MPNN-like)
        self.kan_mpnn_layers = nn.ModuleList(
            [
                MultiKANLayer(
                    hidden_channels, hidden_channels, num_functions=num_functions
                )
                for _ in range(num_layers)
            ]
        )

        # MultiKAN layers for attention mechanism
        self.kan_attn_layers = nn.ModuleList(
            [
                MultiKANLayer(
                    hidden_channels, hidden_channels, num_functions=num_functions
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, edge_index, pos_encoding=None):
        """
        Forward pass of the KAN-GPS model.

        Args:
            x (torch.Tensor): Input node features.
            edge_index (torch.Tensor): Edge indices.
            pos_encoding (torch.Tensor, optional): Positional encoding of nodes.

        Returns:
            torch.Tensor: Output node features after applying KAN-MPNN, KAN-Attention, and GPS layers.
        """
        # Initialize positional encoding if not provided
        if pos_encoding is None:
            pos_encoding = torch.zeros(x.size(0), self.pos_dim, device=x.device)

        # Initial linear projection
        x = self.initial_linear(x)

        # Apply KAN-MPNN, KAN-Attention, and GPS layers sequentially
        for i in range(self.num_layers):
            x_mpnn = self.kan_mpnn_layers[i](x)
            x_attn = self.kan_attn_layers[i](x)
            x_mpnn = self.gps_layers[i](x_mpnn, edge_index, pos_encoding)
            x = x_mpnn + x_attn  # Combine MPNN and attention outputs
            x = torch.relu(x)  # Non-linearity between layers

        # Final linear projection
        x = self.final_linear(x)
        return x
