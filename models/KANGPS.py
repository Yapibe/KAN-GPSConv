import math
import torch
import torch.nn.functional as F
from torch.nn import ModuleList, BatchNorm1d
from torch_geometric.nn import GCNConv, GINConv,MessagePassing
from torch_geometric.utils import to_dense_batch,add_self_loops,scatter
from .eKAN import KAN as eKAN
import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_mean_pool
from ogb.graphproppred.mol_encoder import AtomEncoder as ogb_AtomEncoder

def make_kan(num_features, hidden_dim, out_dim, hidden_layers, grid_size, spline_order, base_activation=torch.nn.SiLU):
    sizes = [num_features] + [hidden_dim] * (hidden_layers - 1) + [out_dim]
    return eKAN(layers_hidden=sizes, grid_size=grid_size, spline_order=spline_order, base_activation=base_activation)


def make_kan_linear(num_features, out_dim, grid_size=2, spline_order=1, base_activation=nn.Identity):
    """
    Simplified KAN setup for linear behavior.
    Args:
        num_features (int): Input features.
        out_dim (int): Output features.
        grid_size (int): Reduced grid size for simplified interpolation.
        spline_order (int): Linear interpolation.
        base_activation: Identity activation for linearity.
    Returns:
        eKAN: Configured KAN instance.
    """
    return eKAN(
        layers_hidden=[num_features, out_dim],  # Single-layer KAN
        grid_size=grid_size,                   # Minimal grid size
        spline_order=spline_order,             # Linear splines
        base_activation=base_activation        # Linear activation
    )

def make_mlp(num_features, hidden_dim, out_dim, hidden_layers, grid_size, spline_order, base_activation=torch.nn.SiLU):
    layers = []
    in_dim = num_features
    for _ in range(hidden_layers - 1):
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(base_activation())
        in_dim = hidden_dim
    layers.append(nn.Linear(in_dim, out_dim))
    return nn.Sequential(*layers)

def optional_atom_encoder(use_atom_encoder, atom_encoder_dim):

    if not use_atom_encoder:
        return None
    
    return ogb_AtomEncoder(emb_dim=atom_encoder_dim)



class GIKANLayer_with_official_GIN(GINConv):
    def __init__(self,
                 in_feat: int,
                 out_feat: int,
                 grid_size: int = 4,
                 spline_order: int = 3,
                 kan_hidden_dim: int = 16,
                 kan_num_of_hidden_layers: int = 2,
                 kan_input_layer_dim: int = None,
                 kan_out_layer_dim: int = None):
        # If kan_input_layer_dim and kan_out_layer_dim are not specified, default to in_feat and out_feat
        kan_input_layer_dim = kan_input_layer_dim if kan_input_layer_dim is not None else in_feat
        kan_out_layer_dim = kan_out_layer_dim if kan_out_layer_dim is not None else out_feat

        kan = make_kan(num_features=kan_input_layer_dim,
                       hidden_dim=kan_hidden_dim,
                       out_dim=kan_out_layer_dim,
                       hidden_layers=kan_num_of_hidden_layers,
                       grid_size=grid_size,
                       spline_order=spline_order)
        GINConv.__init__(self, nn=kan,train_eps=True)


class GCN_post_kan_Layer(torch.nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 grid_size: int = 4,
                 spline_order: int = 3,
                 kan_hidden_dim: int = 16,
                 kan_num_of_hidden_layers: int = 2,
                 kan_input_layer_dim: int = None,
                 kan_out_layer_dim: int = None,
                 improved: bool = False,
                 cached: bool = False,
                 normalize: bool = True,
                 add_self_loops: bool = True,
                 bias: bool = True):
        super().__init__()
        # Initialize the GCNConv layer
        self.conv = GCNConv(
            in_channels,
            out_channels,
            improved=improved,
            cached=cached,
            normalize=normalize,
            add_self_loops=add_self_loops,
            bias=bias
        )
        # If kan_input_layer_dim and kan_out_layer_dim are not specified, default to out_channels
        kan_input_layer_dim = kan_input_layer_dim if kan_input_layer_dim is not None else out_channels
        kan_out_layer_dim = kan_out_layer_dim if kan_out_layer_dim is not None else out_channels

        # Create the KAN layer to apply after GCNConv
        self.kan = make_kan(
            num_features=kan_input_layer_dim,
            hidden_dim=kan_hidden_dim,
            out_dim=kan_out_layer_dim,
            hidden_layers=kan_num_of_hidden_layers,
            grid_size=grid_size,
            spline_order=spline_order
        )

    def forward(self, x, edge_index, edge_weight=None):
        # Apply GCNConv
        x = self.conv(x, edge_index, edge_weight)
        # Apply KAN layer on the output of GCNConv
        x = self.kan(x)
        return x

    def reset_parameters(self):
        self.conv.reset_parameters()
        if hasattr(self.kan, 'reset_parameters'):
            self.kan.reset_parameters()

    def __repr__(self):
        return f'{self.__class__.__name__}({self.conv.in_channels}, {self.conv.out_channels})'


class GCNKANLayer(MessagePassing):
    """
    Custom GCN-like layer that does:
      1. A_hat x   (where A_hat is normalized adjacency + self-loops)
      2. Pass the result to a KAN subnetwork for non-linear update

    This effectively replaces the typical 'A_hat x W' with 'KAN(A_hat x)'.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 grid_size: int = 4,
                 spline_order: int = 3,
                 kan_hidden_dim: int = 16,
                 kan_num_of_hidden_layers: int = 2,
                 kan_input_layer_dim: int = None,
                 kan_out_layer_dim: int = None,
                 base_activation=torch.nn.SiLU,
                 **kwargs):
        # We use 'add' aggregator (sum) for GCN
        super().__init__(aggr='add', **kwargs)

        # If user doesn't provide KAN input/output dims, default them
        kan_input_layer_dim = kan_input_layer_dim if kan_input_layer_dim is not None else in_channels
        kan_out_layer_dim = kan_out_layer_dim if kan_out_layer_dim is not None else out_channels

        # Build a KAN sub-network for final non-linear "update"
        self.kan = make_kan(
            num_features=kan_input_layer_dim,
            hidden_dim=kan_hidden_dim,
            out_dim=kan_out_layer_dim,
            hidden_layers=kan_num_of_hidden_layers,
            grid_size=grid_size,
            spline_order=spline_order,
            base_activation=base_activation
        )

    def forward(self, x, edge_index, edge_weight=None):
        """
        x: [num_nodes, in_channels]
        edge_index: [2, num_edges]
        edge_weight: Optional[Tensor] of shape [num_edges]
        """
        num_nodes = x.size(0)

        # 1) Add self loops so each node sees itself
        edge_index, edge_weight = add_self_loops(
            edge_index, edge_weight, fill_value=1.0, num_nodes=num_nodes
        )

        # 2) Compute deg and adjacency normalization factors
        row, col = edge_index
        # If edge_weight is None, create a ones vector
        if edge_weight is None:
            edge_weight = x.new_ones(edge_index.size(1))

        deg = scatter(edge_weight, row, dim=0, dim_size=num_nodes, reduce='add')
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        # norm[e] = 1 / sqrt(deg[row[e]] * deg[col[e]])
        self.norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

        # 3) Perform standard message passing
        #    => out[i] = sum_j (norm[e_ij] * x_j)
        out = self.propagate(edge_index, x=x)

        # 4) Pass the aggregated result to KAN
        out = self.kan(out)
        return out

    def message(self, x_j):
        """
        Multiply node j's feature by norm[e].
        """
        # 'x_j' has shape [num_edges, in_channels]
        # 'self.norm' has shape [num_edges]
        return self.norm.unsqueeze(-1) * x_j

    def update(self, aggr_out):
        """
        PyG calls update(...) on the aggregated output from message & aggregate.
        But we do nothing here, because we prefer to apply KAN after self.propagate returns.
        """
        return aggr_out

class GIKANLayer(MessagePassing):
    """
    A custom layer that:
      - Adds self-loops to ensure x_i is included in the neighbor sum
      - Messages: x_j for j in N(i)
      - Aggregates: sum of x_j over j in N(i) (including i itself)
      - Updates: Apply KAN to the summed result
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 grid_size: int = 4,
                 spline_order: int = 3,
                 kan_hidden_dim: int = 16,
                 kan_num_of_hidden_layers: int = 2,
                 kan_input_layer_dim: int = None,
                 kan_out_layer_dim: int = None,
                 **kwargs):
        super().__init__(aggr='add', **kwargs)  # "add" = sum aggregator

        # Default input/output dims for KAN, similar to your existing style:
        kan_input_layer_dim = kan_input_layer_dim if kan_input_layer_dim is not None else in_channels
        kan_out_layer_dim = kan_out_layer_dim if kan_out_layer_dim is not None else out_channels

        # Create a KAN subnetwork to perform the update step
        self.kan = make_kan(
            num_features=kan_input_layer_dim,
            hidden_dim=kan_hidden_dim,
            out_dim=kan_out_layer_dim,
            hidden_layers=kan_num_of_hidden_layers,
            grid_size=grid_size,
            spline_order=spline_order
        )

    def forward(self, x, edge_index):
        """
        x: [num_nodes, in_channels]
        edge_index: [2, num_edges]
        Returns: [num_nodes, out_channels]
        """
        # Ensure each node is in its own neighborhood by adding self-loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Run PyG's message passing:
        #   self.propagate -> message -> aggregate -> update
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        """
        For each edge (j -> i), the message is x_j.
        """
        return x_j  # shape: [num_edges, in_channels]

    def update(self, aggr_out):
        """
        Once we sum up messages (neighbors + self), pass the result to KAN.
        """
        return self.kan(aggr_out)  # shape: [num_nodes, out_channels]

class KANSageConvLayer(MessagePassing):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 grid_size: int = 4,
                 spline_order: int = 3,
                 kan_hidden_dim: int = 16,
                 kan_num_of_hidden_layers: int = 2,
                 kan_input_layer_dim: int = None,    
                 kan_out_layer_dim: int = None,
                 base_activation=torch.nn.SiLU,
                 add_self_loop: bool = True,
                 **kwargs):
        # "mean" aggregator = average of neighbors
        super().__init__(aggr='mean', **kwargs)

        kan_input_layer_dim = 2 * kan_input_layer_dim if kan_input_layer_dim is not None else 2 * in_channels
        kan_out_layer_dim = kan_out_layer_dim if kan_out_layer_dim is not None else out_channels

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_self_loop = add_self_loop

        # Build KAN subnetwork
        self.kan = make_kan(
            num_features=kan_input_layer_dim,
            hidden_dim=kan_hidden_dim,
            out_dim=kan_out_layer_dim,
            hidden_layers=kan_num_of_hidden_layers,
            grid_size=grid_size,
            spline_order=spline_order,
            base_activation=base_activation
        )

    def forward(self, x, edge_index, edge_weight=None):
        if self.add_self_loop:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # 1) propagate -> returns mean of neighbor features for each node
        aggr_out = self.propagate(edge_index, x=x)

        # 2) concat the node's original feature with the aggregated neighbor feature
        out = torch.cat([x, aggr_out], dim=-1)  # shape: [num_nodes, 2*in_channels]

        # 3) pass to KAN for the update
        out = self.kan(out)  # shape: [num_nodes, out_channels]

        return out

    def message(self, x_j):
        """
        For each edge (j -> i), the 'message' is just x_j.
        The aggregator='mean' will handle dividing by the number of neighbors.
        """
        return x_j

    def update(self, aggr_out):
        """
        PyG calls this after aggregation. But we prefer to handle the final
        KAN update outside in `forward()`. So no transform here.
        """
        return aggr_out


class KANLinear(nn.Module):

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 grid_size: int = 2,
                 spline_order: int = 1,
                 kan_hidden_dim: int = 16,  # Optional if eKAN requires it, but unused
                 base_activation=nn.Identity):
        """
        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            grid_size (int): Grid size for KAN (minimal for linearity).
            spline_order (int): Spline order (1 for linear interpolation).
            base_activation (nn.Module): Activation function (Identity for linear behavior).
        """
        super().__init__()
        self.kan = make_kan_linear(
            num_features=in_features,
            out_dim=out_features,
            grid_size=grid_size,
            spline_order=spline_order,
            base_activation=base_activation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the linear-like KAN.
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, in_features].
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, out_features].
        """
        return self.kan(x)


class KANLinear_old(nn.Module):
    """
    A drop-in replacement for nn.Linear that uses your make_kan(...) function.
    """
    def __init__(self,
                 in_features,
                 out_features,
                 grid_size=4,
                 spline_order=3,
                 kan_hidden_dim=16,
                 kan_num_layers=2,
                 base_activation=nn.SiLU):
        super().__init__()
        # Build a KAN subnetwork that goes from in_features -> out_features
        self.kan = make_kan(
            num_features=in_features,
            hidden_dim=kan_hidden_dim,
            out_dim=out_features,
            hidden_layers=kan_num_layers,
            grid_size=grid_size,
            spline_order=spline_order,
            base_activation=base_activation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: [batch_size, in_features]
        Returns shape: [batch_size, out_features]
        """
        return self.kan(x)
    

class KANMultiheadAttention(nn.Module):
    """
    A simplified multi-head attention module where:
      - Q, K, V projections are done via KANLinear instead of nn.Linear
      - The output projection is also a KANLinear
    """
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 grid_size: int = 4,
                 spline_order: int = 3,
                 kan_hidden_dim: int = 16,
                 kan_num_layers: int = 2,
                 base_activation=nn.SiLU,
                 dropout: float = 0.0):
        """
        Args:
            embed_dim (int): Dimensionality of input embeddings.
            num_heads (int): Number of attention heads.
            grid_size, spline_order, kan_hidden_dim, kan_num_layers, base_activation:
                Hyperparameters for the KANLinear layers.
            dropout (float): Dropout on attention weights.
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

        # Q, K, V transformations
        self.q_kan = KANLinear(in_features=embed_dim, out_features = embed_dim, grid_size = grid_size, spline_order=spline_order)
        self.k_kan = KANLinear(in_features=embed_dim, out_features = embed_dim, grid_size = grid_size, spline_order=spline_order)
        self.v_kan = KANLinear(in_features=embed_dim, out_features = embed_dim, grid_size = grid_size, spline_order=spline_order)

        # self.q_kan = KANLinear(embed_dim, embed_dim, grid_size, spline_order, kan_hidden_dim, kan_num_layers, base_activation)
        # self.k_kan = KANLinear(embed_dim, embed_dim, grid_size, spline_order, kan_hidden_dim, kan_num_layers, base_activation)
        # self.v_kan = KANLinear(embed_dim, embed_dim, grid_size, spline_order, kan_hidden_dim, kan_num_layers, base_activation)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask=None):
        """
        query, key, value: shape [batch_size, seq_len, embed_dim]
        mask: optional (batch_size, seq_len) or (batch_size, num_heads, seq_len, seq_len)

        Returns: (attn_output, attn_weights)
          attn_output: shape [batch_size, seq_len, embed_dim]
          attn_weights (optional): shape [batch_size, num_heads, seq_len, seq_len]
        """
        B, L, E = query.shape
        assert E == self.embed_dim, "query embed dim must match layer's embed_dim"

        # 1) Project Q, K, V using KAN
        # Flatten [B, L, E] => [B*L, E], pass through KAN, then reshape back
        Q = self.q_kan(query.reshape(-1, E)).view(B, L, E)
        K = self.k_kan(key.reshape(-1, E)).view(B, L, E)
        V = self.v_kan(value.reshape(-1, E)).view(B, L, E)

        # 2) Reshape for multi-head: [B, L, num_heads, head_dim]
        Q = Q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # => [B, num_heads, L, head_dim]
        K = K.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # 3) Scaled Dot-Product Attention
        #    attn_weights: [B, num_heads, L, L]
        attn_scores = torch.matmul(Q, K.transpose(-2, -1))  # shape [B, num_heads, L, L]
        attn_scores = attn_scores / math.sqrt(self.head_dim)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)  # softmax along last dim
        attn_weights = self.dropout(attn_weights)

        # 4) Weighted sum of V
        attn_output = torch.matmul(attn_weights, V)  # [B, num_heads, L, head_dim]

        # 5) Recombine heads: => [B, L, embed_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, E)

        return attn_output, attn_weights

class GraphormerKANLayer(torch.nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 attention_dropout: float,
                 kan_hidden_layer_dim: int,
                 kan_num_of_hidden_layers: int,
                 grid_size: int,
                 spline_order: int,
                 kan_input_layer_dim: int = None,
                 kan_out_layer_dim: int = None):
        super().__init__()
        self.input_norm = torch.nn.LayerNorm(embed_dim)

        self.attention = KANMultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            grid_size=grid_size,
            spline_order=spline_order,
            kan_hidden_dim=kan_hidden_layer_dim,
            kan_num_layers=kan_num_of_hidden_layers,
            dropout=attention_dropout
        )
        kan_input_layer_dim = kan_input_layer_dim if kan_input_layer_dim is not None else embed_dim
        kan_out_layer_dim = kan_out_layer_dim if kan_out_layer_dim is not None else embed_dim

        self.kan = make_kan(num_features=kan_input_layer_dim,
                            hidden_dim=kan_hidden_layer_dim,
                            out_dim=kan_out_layer_dim,
                            hidden_layers=kan_num_of_hidden_layers,
                            grid_size=grid_size,
                            spline_order=spline_order)

    def forward(self, x, batch=None):
        x = self.input_norm(x)

        if batch is not None:
            x_padded, mask = to_dense_batch(x, batch)
            attn_output, _ = self.attention(x_padded, x_padded, x_padded)
            x = attn_output[mask]
        else:
            x = x.unsqueeze(0) 
            attn_output, _ = self.attention(x, x, x)
            x = attn_output.squeeze(0)

        x = self.kan(x)
        return x

class KANGPSLayer(torch.nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 mpnn_type: str,
                 attn_num_heads: int,
                 attention_dropout: float,
                 dropout: float,

                 grid_size: int = 4,
                 spline_order: int = 3,
                 kan_hidden_dim: int = 16,
                 kan_num_of_hidden_layers: int = 2,
                 grid_size_attention: int = 4,

                 spline_order_attention: int = 3,
                 kan_hidden_dim_attention: int = 16,
                 kan_num_of_hidden_layers_attention: int = 2,

                 grid_size_mpnn: int = 4,
                 spline_order_mpnn: int = 3,
                 kan_hidden_dim_mpnn: int = 16,
                 kan_num_of_hidden_layers_mpnn: int = 2,

                 kan_input_layer_dim: int = None,
                 kan_out_layer_dim: int = None,

                 kan_input_layer_dim_attention: int = None,
                 kan_out_layer_dim_attention: int = None,

                 kan_input_layer_dim_mpnn: int = None,
                 kan_out_layer_dim_mpnn: int = None,

                 kan_base_activation: torch.nn = torch.nn.SiLU,
                 *args, **kwargs):

        super().__init__(*args, **kwargs)

        kan_input_layer_dim = kan_input_layer_dim if kan_input_layer_dim is not None else in_channels
        kan_out_layer_dim = kan_out_layer_dim if kan_out_layer_dim is not None else out_channels

        # KAN layer after combining MPNN and attention outputs
        self.kan = make_kan(num_features=kan_input_layer_dim,
                            hidden_dim=kan_hidden_dim,
                            out_dim=kan_out_layer_dim,
                            hidden_layers=kan_num_of_hidden_layers,
                            grid_size=grid_size,
                            spline_order=spline_order,
                            base_activation=kan_base_activation)

        # GraphormerKANLayer
        self.attention = GraphormerKANLayer(
            embed_dim=in_channels,
            num_heads=attn_num_heads,
            attention_dropout=attention_dropout,
            kan_hidden_layer_dim=kan_hidden_dim_attention,
            kan_num_of_hidden_layers=kan_num_of_hidden_layers_attention,
            grid_size=grid_size_attention,
            spline_order=spline_order_attention,
            kan_input_layer_dim=kan_input_layer_dim_attention,
            kan_out_layer_dim=kan_out_layer_dim_attention
        )

        # MPNN Layer
        if mpnn_type == 'GCN':
            self.mpnn = GCNKANLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                grid_size=grid_size_mpnn,
                spline_order=spline_order_mpnn,
                kan_hidden_dim=kan_hidden_dim_mpnn,
                kan_num_of_hidden_layers=kan_num_of_hidden_layers_mpnn,
                kan_input_layer_dim=in_channels,
                kan_out_layer_dim=kan_out_layer_dim_mpnn
            )
        elif mpnn_type == 'GIN':
            self.mpnn = GIKANLayer(
                # in_feat=in_channels,
                # out_feat=out_channels,
                in_channels=in_channels,
                out_channels=out_channels,
                grid_size=grid_size_mpnn,
                spline_order=spline_order_mpnn,
                kan_hidden_dim=kan_hidden_dim_mpnn,
                kan_num_of_hidden_layers=kan_num_of_hidden_layers_mpnn,
                kan_input_layer_dim= in_channels,
                kan_out_layer_dim=kan_out_layer_dim_mpnn
            )
        elif mpnn_type=='Sage':
            self.mpnn = KANSageConvLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                grid_size=grid_size_mpnn,
                spline_order=spline_order_mpnn,
                kan_hidden_dim=kan_hidden_dim_mpnn,
                kan_num_of_hidden_layers=kan_num_of_hidden_layers_mpnn,
                kan_input_layer_dim=in_channels,
                kan_out_layer_dim=kan_out_layer_dim_mpnn
            )
        else:
            raise ValueError(f"Unsupported MPNN type: {mpnn_type}")

        self.mpnn_batchnorm = torch.nn.BatchNorm1d(out_channels)
        self.attention_batchnorm = torch.nn.BatchNorm1d(out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index, batch=None):
        x_l = x

        # MPNN path
        x_m = self.mpnn(x, edge_index)
        x_m = F.dropout(x_m, p=self.dropout, training=self.training)
        x_m = self.mpnn_batchnorm(x_m)

        if x_m.shape == x_l.shape:
            x_m = x_m + x_l  # Residual connection
        

        # Attention path
        x_t = self.attention(x, batch)
        x_t = F.dropout(x_t, p=self.dropout, training=self.training)
        x_t = self.attention_batchnorm(x_t)
        
        if x_t.shape == x_l.shape:
             x_t = x_t + x_l  # Residual connection

        # Combine paths and apply final KAN
        out = self.kan(x_t + x_m)
        return out

class KANGPS(torch.nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_layers: int,
                 mpnn_type: str,
                 dropout: float = 0.1,
                 attn_num_heads: int = 1,
                 attention_dropout: float = 0.1,

                 grid_size: int = 4,
                 spline_order: int = 3,
                 kan_hidden_dim: int = 16,
                 kan_num_of_hidden_layers: int = 2,

                 grid_size_attention: int = 4,
                 spline_order_attention: int = 3,
                 kan_hidden_dim_attention: int = 16,
                 kan_num_of_hidden_layers_attention: int = 2,

                 grid_size_mpnn: int = 4,
                 spline_order_mpnn: int = 3,
                 kan_hidden_dim_mpnn: int = 16,
                 kan_num_of_hidden_layers_mpnn: int = 2,

                 kan_input_layer_dim: int = None,
                 kan_out_layer_dim: int = None,
                 return_embeddings: bool = False,
                 pool: bool = True,
                 final_layer_activation=None,

                 use_atom_encoder: bool = False,
                 atom_encoder_dim: int = None

                 ):
        super().__init__()

        self.layers = ModuleList()
        self.return_embeddings = return_embeddings

        atom_encoder_dim = hidden_dim if atom_encoder_dim is None else atom_encoder_dim

        self.atom_encoder = optional_atom_encoder(
            use_atom_encoder=use_atom_encoder,
            atom_encoder_dim=atom_encoder_dim
        )

        first_layer_input_dim = atom_encoder_dim if self.atom_encoder else input_dim


        self.layers.append(KANGPSLayer(
                in_channels=first_layer_input_dim,
                out_channels=hidden_dim,
                mpnn_type=mpnn_type,
                attn_num_heads=attn_num_heads,
                attention_dropout=attention_dropout,
                dropout=dropout,

                grid_size=grid_size,
                spline_order=spline_order,
                kan_hidden_dim=kan_hidden_dim,
                kan_num_of_hidden_layers=kan_num_of_hidden_layers,

                grid_size_attention=grid_size_attention,
                spline_order_attention=spline_order_attention,
                kan_hidden_dim_attention=kan_hidden_dim_attention,
                kan_num_of_hidden_layers_attention=kan_num_of_hidden_layers_attention,

                grid_size_mpnn=grid_size_mpnn,
                spline_order_mpnn=spline_order_mpnn,
                kan_hidden_dim_mpnn=kan_hidden_dim_mpnn,
                kan_num_of_hidden_layers_mpnn=kan_num_of_hidden_layers_mpnn,

                kan_input_layer_dim=hidden_dim,
                kan_out_layer_dim=hidden_dim,

                kan_input_layer_dim_attention= first_layer_input_dim,
                kan_out_layer_dim_attention = hidden_dim ,

                 kan_input_layer_dim_mpnn = hidden_dim,
                 kan_out_layer_dim_mpnn = hidden_dim,
            ))
        


        # Hidden KANGPSLayers
        for _ in range(num_layers - 2):
            self.layers.append(KANGPSLayer(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                mpnn_type=mpnn_type,
                attn_num_heads=attn_num_heads,
                attention_dropout=attention_dropout,
                dropout=dropout,

                grid_size=grid_size,
                spline_order=spline_order,
                kan_hidden_dim=kan_hidden_dim,
                kan_num_of_hidden_layers=kan_num_of_hidden_layers,

                grid_size_attention=grid_size_attention,
                spline_order_attention=spline_order_attention,
                kan_hidden_dim_attention=kan_hidden_dim_attention,
                kan_num_of_hidden_layers_attention=kan_num_of_hidden_layers_attention,

                grid_size_mpnn=grid_size_mpnn,
                spline_order_mpnn=spline_order_mpnn,
                kan_hidden_dim_mpnn=kan_hidden_dim_mpnn,
                kan_num_of_hidden_layers_mpnn=kan_num_of_hidden_layers_mpnn,

                kan_input_layer_dim=hidden_dim,
                kan_out_layer_dim=hidden_dim
            ))

        # Output KANGPSLayer
        self.layers.append(KANGPSLayer(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            mpnn_type=mpnn_type,
            attn_num_heads=attn_num_heads,
            attention_dropout=attention_dropout,
            dropout=dropout,

            grid_size=grid_size,
            spline_order=spline_order,
            kan_hidden_dim=kan_hidden_dim,
            kan_num_of_hidden_layers=kan_num_of_hidden_layers,

            grid_size_attention=grid_size_attention,
            spline_order_attention=spline_order_attention,
            kan_hidden_dim_attention=kan_hidden_dim_attention,
            kan_num_of_hidden_layers_attention=kan_num_of_hidden_layers_attention,

            grid_size_mpnn=grid_size_mpnn,
            spline_order_mpnn=spline_order_mpnn,
            kan_hidden_dim_mpnn=kan_hidden_dim_mpnn,
            kan_num_of_hidden_layers_mpnn=kan_num_of_hidden_layers_mpnn,

            kan_input_layer_dim=hidden_dim,
            kan_out_layer_dim=output_dim,
        ))
        self.pool = None
        if pool:
            self.pool = global_mean_pool

        self.final_layer_activation = final_layer_activation

    def forward(self, x, edge_index, positional_encoding=None, structural_encoding=None, batch=None):

        if self.atom_encoder is not None:
            x = x.long()
            x = self.atom_encoder(x)

        embeddings = []
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, batch)

            if self.return_embeddings:
                embeddings.append(x)

        if self.pool is not None:
            x = self.pool(x, batch)

        if self.final_layer_activation is not None:
            x = self.final_layer_activation(x)

        if self.return_embeddings:
            return x, embeddings
        else:
            return x


