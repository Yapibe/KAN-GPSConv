import torch
import torch.nn as nn
import torch.nn.functional as F

class KANLayer(nn.Module):
    def __init__(self, in_features, out_features, grid_features=100, use_bias=True):
        super(KANLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_features = grid_features
        
        self.linear_in = nn.Linear(in_features, grid_features, bias=use_bias)
        self.weight = nn.Parameter(torch.Tensor(out_features, grid_features))
        if use_bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        # Project input to grid space
        x = self.linear_in(input)
        # Apply periodic activation
        x = torch.sin(x)
        # Linear transformation
        output = F.linear(x, self.weight, self.bias)
        return output
