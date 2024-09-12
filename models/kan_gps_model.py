import torch
import torch.nn as nn
from .kan_layer import KANLayer
from .gps_layer import GPSLayer

class KANGPSModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(KANGPSModel, self).__init__()
        self.num_layers = num_layers
        
        self.initial_linear = nn.Linear(in_channels, hidden_channels)
        
        self.kan_layers = nn.ModuleList([
            KANLayer(hidden_channels, hidden_channels) for _ in range(num_layers)
        ])
        
        self.gps_layers = nn.ModuleList([
            GPSLayer(hidden_channels, hidden_channels) for _ in range(num_layers)
        ])
        
        self.final_linear = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, pos_encoding):
        x = self.initial_linear(x)
        
        for i in range(self.num_layers):
            x = self.kan_layers[i](x)
            x = self.gps_layers[i](x, edge_index, pos_encoding)
            x = torch.relu(x)
        
        x = self.final_linear(x)
        return x
