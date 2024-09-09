import torch
import torch.nn as nn
import torch.nn.functional as F

class KANLayer(nn.Module):
    def __init__(self, in_features, out_features, num_neurons=10):
        super(KANLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_neurons = num_neurons
        
        # Initialize weights and biases
        self.weights = nn.Parameter(torch.Tensor(num_neurons, in_features))
        self.biases = nn.Parameter(torch.Tensor(num_neurons))
        self.output_weights = nn.Parameter(torch.Tensor(out_features, num_neurons))
        
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights)
        nn.init.xavier_uniform_(self.output_weights)
        nn.init.zeros_(self.biases)
    
    def forward(self, x):
        # Compute the inner products
        inner_products = F.linear(x, self.weights, self.biases)
        
        # Apply activation function (e.g., sine for KAN)
        activations = torch.sin(inner_products)
        
        # Compute output
        output = F.linear(activations, self.output_weights)
        
        return output
