import torch
import torch.nn as nn
from .kan_layer import KANLayer, MultiKANLayer
from .gps_layer import GPSLayer

class KANGPSBase(nn.Module):
    """
    Base class for KAN-GPS models.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of hidden layer channels.
        out_channels (int): Size of each output sample.
        num_layers (int): Number of layers in the model.
        pos_dim (int): Dimension of positional encoding (default: 64).
        activation (str): Activation function to use in KAN layers ('silu', 'relu', 'tanh', 'leaky_relu', 'elu') (default: 'silu').
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, pos_dim=64, activation='silu'):
        super(KANGPSBase, self).__init__()
        self.num_layers = num_layers
        
        self.initial_linear = nn.Linear(in_channels, hidden_channels)
        
        self.kan_layers = nn.ModuleList([
            KANLayer(hidden_channels, hidden_channels, activation=activation) for _ in range(num_layers)
        ])
        
        self.gps_layers = nn.ModuleList([
            GPSLayer(hidden_channels, hidden_channels) for _ in range(num_layers)
        ])
        
        self.final_linear = nn.Linear(hidden_channels, out_channels)

        self.pos_dim = pos_dim

    def forward(self, x, edge_index, pos_encoding=None):
        """
        Forward pass of the KAN-GPS base model.

        Args:
            x (torch.Tensor): Input node features.
            edge_index (torch.Tensor): Edge indices.
            pos_encoding (torch.Tensor): Positional encoding of nodes (default: None).

        Returns:
            torch.Tensor: Output node features after applying KAN and GPS layers.
        """
        if pos_encoding is None:
            pos_encoding = torch.zeros(x.size(0), self.pos_dim).to(x.device)
        x = self.initial_linear(x)
        
        for i in range(self.num_layers):
            x = self.kan_layers[i](x)
            x = self.gps_layers[i](x, edge_index, pos_encoding)
            x = torch.relu(x)
        
        x = self.final_linear(x)
        return x

class HybridKANGPS(KANGPSBase):
    """
    Hybrid KAN-GPS model.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of hidden layer channels.
        out_channels (int): Size of each output sample.
        num_layers (int): Number of layers in the model.
        pos_dim (int): Dimension of positional encoding (default: 64).
        activation (str): Activation function to use in KAN layers ('silu', 'relu', 'tanh', 'leaky_relu', 'elu') (default: 'silu').
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, pos_dim=64, activation='silu'):
        super(HybridKANGPS, self).__init__(in_channels, hidden_channels, out_channels, num_layers, pos_dim, activation)
        
        self.kan_layers = nn.ModuleList([
            KANLayer(hidden_channels, hidden_channels, activation=activation) for _ in range(num_layers)
        ])

class KANGPS(KANGPSBase):
    """
    KAN-GPS model.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of hidden layer channels.
        out_channels (int): Size of each output sample.
        num_layers (int): Number of layers in the model.
        pos_dim (int): Dimension of positional encoding (default: 64).
        activation (str): Activation function to use in KAN layers ('silu', 'relu', 'tanh', 'leaky_relu', 'elu') (default: 'silu').
        num_functions (int): Number of univariate functions in MultiKANLayer (default: 4).
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, pos_dim=64, activation='silu', num_functions=4):
        super(KANGPS, self).__init__(in_channels, hidden_channels, out_channels, num_layers, pos_dim, activation)
        
        self.kan_mpnn_layers = nn.ModuleList([
            MultiKANLayer(hidden_channels, hidden_channels, num_functions=num_functions, activation=activation) for _ in range(num_layers)
        ])
        
        self.kan_attn_layers = nn.ModuleList([
            MultiKANLayer(hidden_channels, hidden_channels, num_functions=num_functions, activation=activation) for _ in range(num_layers)
        ])
    
    def forward(self, x, edge_index, pos_encoding=None):
        """
        Forward pass of the KAN-GPS model.

        Args:
            x (torch.Tensor): Input node features.
            edge_index (torch.Tensor): Edge indices.
            pos_encoding (torch.Tensor): Positional encoding of nodes (default: None).

        Returns:
            torch.Tensor: Output node features after applying KAN-MPNN, KAN-Attention, and GPS layers.
        """
        if pos_encoding is None:
            pos_encoding = torch.zeros(x.size(0), self.pos_dim).to(x.device)
        x = self.initial_linear(x)
        
        for i in range(self.num_layers):
            x_mpnn = self.kan_mpnn_layers[i](x)
            x_attn = self.kan_attn_layers[i](x)
            x_mpnn = self.gps_layers[i](x_mpnn, edge_index, pos_encoding)
            x = x_mpnn + x_attn
            x = torch.relu(x)
        
        x = self.final_linear(x)
        return x
