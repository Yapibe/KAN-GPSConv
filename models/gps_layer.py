import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class GPSLayer(nn.Module):
    def __init__(self, in_channels, out_channels, pos_dim=None):
        super(GPSLayer, self).__init__()
        self.conv = GCNConv(in_channels, out_channels)
        if pos_dim is not None:
            self.positional_encoding = nn.Linear(pos_dim, out_channels)
        else:
            self.positional_encoding = None

    def forward(self, x, edge_index, pos_encoding=None):
        x = self.conv(x, edge_index)
        if self.positional_encoding is not None and pos_encoding is not None:
            pos = self.positional_encoding(pos_encoding)
            x = x + pos
        return x
