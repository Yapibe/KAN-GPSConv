"""
This module contains the implementation of the GPS layer.
"""
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, SAGEConv


class GPSLayer(nn.Module):
    """
    Graph Positional Signature (GPS) Layer.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        conv_type (str): Type of graph convolution to use (default: 'GCN').
        pos_dim (int): Dimension of positional encoding (default: None).
        heads (int): Number of attention heads for GAT convolution (default: 1).
    """

    def __init__(
        self, in_channels, out_channels, conv_type="GCN", pos_dim=None, heads=1
    ):
        super(GPSLayer, self).__init__()
        self.conv_type = conv_type
        if conv_type == "GCN":
            self.conv = GCNConv(in_channels, out_channels)
        elif conv_type == "GAT":
            self.conv = GATConv(in_channels, out_channels, heads=heads, concat=False)
        elif conv_type == "SAGE":
            self.conv = SAGEConv(in_channels, out_channels)
        else:
            raise ValueError(f"Unsupported convolution type: {conv_type}")

        if pos_dim is not None:
            self.positional_encoding = nn.Linear(pos_dim, out_channels)
        else:
            self.positional_encoding = None

    def forward(self, x, edge_index, pos_encoding=None):
        """
        Forward pass of the GPS layer.

        Args:
            x (torch.Tensor): Input node features.
            edge_index (torch.Tensor): Edge indices.
            pos_encoding (torch.Tensor): Positional encoding of nodes (default: None).

        Returns:
            torch.Tensor: Output node features after applying graph convolution and positional encoding.
        """
        x = self.conv(x, edge_index)
        if self.positional_encoding is not None and pos_encoding is not None:
            pos = self.positional_encoding(pos_encoding)
            x = x + pos  # Add positional encoding to node features
        return x
