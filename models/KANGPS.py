import torch
import torch.nn.functional as F
from torch.nn import ModuleList, BatchNorm1d
from torch_geometric.nn import GCNConv, GINConv
from torch_geometric.utils import to_dense_batch
from .eKAN import KAN as eKAN
import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_mean_pool


def make_kan(num_features, hidden_dim, out_dim, hidden_layers, grid_size, spline_order, base_activation=torch.nn.SiLU):
    sizes = [num_features] + [hidden_dim] * (hidden_layers - 1) + [out_dim]
    return eKAN(layers_hidden=sizes, grid_size=grid_size, spline_order=spline_order, base_activation=base_activation)


class GIKANLayer(GINConv):
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
        GINConv.__init__(self, kan)


class GCKANLayer(torch.nn.Module):
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
        self.attention = torch.nn.MultiheadAttention(embed_dim,
                                                     num_heads,
                                                     attention_dropout,
                                                     batch_first=True)
        # If kan_input_layer_dim and kan_out_layer_dim are not specified, default to embed_dim
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
            # Create a mask for padding
            x_padded, mask = to_dense_batch(x, batch)
            attn_output, _ = self.attention(x_padded, x_padded, x_padded, key_padding_mask=~mask)
            x = attn_output[mask]
        else:
            x = x.unsqueeze(0)  # Add batch dimension
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
            self.mpnn = GCKANLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                grid_size=grid_size_mpnn,
                spline_order=spline_order_mpnn,
                kan_hidden_dim=kan_hidden_dim_mpnn,
                kan_num_of_hidden_layers=kan_num_of_hidden_layers_mpnn,
                kan_input_layer_dim=kan_input_layer_dim_mpnn,
                kan_out_layer_dim=kan_out_layer_dim_mpnn
            )
        elif mpnn_type == 'GIN':
            self.mpnn = GIKANLayer(
                in_feat=in_channels,
                out_feat=out_channels,
                grid_size=grid_size_mpnn,
                spline_order=spline_order_mpnn,
                kan_hidden_dim=kan_hidden_dim_mpnn,
                kan_num_of_hidden_layers=kan_num_of_hidden_layers_mpnn,
                kan_input_layer_dim=kan_input_layer_dim_mpnn,
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
        x_m = x_m + x_l  # Residual connection

        # Attention path
        x_t = self.attention(x, batch)
        x_t = F.dropout(x_t, p=self.dropout, training=self.training)
        x_t = self.attention_batchnorm(x_t)
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
                 final_layer_activation=None
                 ):
        super().__init__()

        self.layers = ModuleList()
        self.return_embeddings = return_embeddings

        # Initial KAN layer
        self.layers.append(make_kan(num_features=input_dim,
                                    out_dim=hidden_dim,
                                    hidden_dim=kan_hidden_dim,
                                    hidden_layers=kan_num_of_hidden_layers,
                                    grid_size=grid_size,
                                    spline_order=spline_order))

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
        embeddings = []
        for i, layer in enumerate(self.layers):

            if i == 0:
                x = x.float()
                x = layer(x)

            elif i == (len(self.layers) - 1):

                if self.pool is not None:
                    x = layer(x, edge_index, batch)
                    x = self.pool(x, batch)
                    if self.return_embeddings:
                        embeddings.append(x)
                    if self.final_layer_activation is not None:
                        x = self.final_layer_activation(x)

                else:
                    x = layer(x, edge_index, batch)
                    if self.return_embeddings:
                        embeddings.append(x)
                    if self.final_layer_activation is not None:
                        x = self.final_layer_activation(x)
            else:
                x = layer(x, edge_index, batch)

        if self.return_embeddings:
            return x, embeddings
        else:
            return x
