import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from .kan_layer import KANLayer

class GPSNetwork(nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_dim=64, task='node'):
        super(GPSNetwork, self).__init__()
        self.task = task
        
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.kan = KANLayer(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        if task == 'graph':
            self.fc = nn.Linear(hidden_dim, num_classes)
        else:  # node classification
            self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # First graph convolution
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        
        # Apply KAN layer
        x = self.kan(x)
        
        # Second graph convolution
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        if self.task == 'graph':
            # Global pooling for graph classification
            x = global_mean_pool(x, data.batch)
        
        # Final classification layer
        x = self.fc(x)
        
        return F.log_softmax(x, dim=1)
