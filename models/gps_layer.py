import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class GPSLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GPSLayer, self).__init__()
        self.conv = GCNConv(in_channels, out_channels)
        self.positional_encoding = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, pos_encoding):
        x = self.conv(x, edge_index)
        pos = self.positional_encoding(pos_encoding)
        return x + pos
