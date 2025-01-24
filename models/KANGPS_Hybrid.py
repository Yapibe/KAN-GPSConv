import math
import torch
import torch.nn.functional as F
from torch.nn import ModuleList, BatchNorm1d
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_dense_batch, add_self_loops, scatter
import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_mean_pool
from ogb.graphproppred.mol_encoder import AtomEncoder as ogb_AtomEncoder


def make_kan(num_features, hidden_dim, out_dim, hidden_layers,
             grid_size, spline_order, base_activation):

    from .eKAN import KAN as eKAN
    base_activation = torch.nn.SiLU if base_activation is None else base_activation
    sizes = [num_features] + [hidden_dim] * (hidden_layers - 1) + [out_dim]
    return eKAN(
        layers_hidden=sizes,
        grid_size=grid_size,
        spline_order=spline_order,
        base_activation=base_activation
    )

def make_mlp(num_features, hidden_dim, out_dim, hidden_layers):
    layers = []
    in_dim = num_features
    for _ in range(hidden_layers - 1):
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.SiLU())  # Hard-coded SiLU
        in_dim = hidden_dim
    layers.append(nn.Linear(in_dim, out_dim))
    return nn.Sequential(*layers)

def optional_atom_encoder(use_atom_encoder, atom_encoder_dim):

    if not use_atom_encoder:
        return None
    
    return ogb_AtomEncoder(emb_dim=atom_encoder_dim)


class GCNLayer(MessagePassing):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 mlp_hidden_dim: int,
                 mlp_num_of_hidden_layers: int = 2,
                 **kwargs):
        super().__init__(aggr='add', **kwargs)

        self.mlp = make_mlp(
            num_features=in_channels,
            hidden_dim=mlp_hidden_dim,
            out_dim=out_channels,
            hidden_layers=mlp_num_of_hidden_layers
        )

    def forward(self, x, edge_index, edge_weight=None):
        num_nodes = x.size(0)
        edge_index, edge_weight = add_self_loops(
            edge_index, edge_weight, fill_value=1.0, num_nodes=num_nodes
        )
        row, col = edge_index
        if edge_weight is None:
            edge_weight = x.new_ones(edge_index.size(1))

        deg = scatter(edge_weight, row, dim=0, dim_size=num_nodes, reduce='add')
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        self.norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

        out = self.propagate(edge_index, x=x)
        return out

    def message(self, x_j):
        return self.norm.unsqueeze(-1) * x_j

    def update(self, aggr_out):
        return self.mlp(aggr_out)

class GINLayer(MessagePassing):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 mlp_num_of_hidden_layers: int = 2,
                 mlp_hidden_dim: int = 16,
                 **kwargs):
        super().__init__(aggr='add', **kwargs)

        self.mlp = make_mlp(
            num_features=in_channels,
            hidden_dim=mlp_hidden_dim,
            out_dim=out_channels,
            hidden_layers=mlp_num_of_hidden_layers
        )

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return x_j

    def update(self, aggr_out):
        return self.mlp(aggr_out)

class SageConvLayer(MessagePassing):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 mlp_num_of_hidden_layers: int = 2,
                 mlp_hidden_dim: int = 16,
                 add_self_loop: bool = True,
                 **kwargs):
        super().__init__(aggr='mean', **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_self_loop = add_self_loop

        self.mlp = make_mlp(
            num_features=2 * in_channels,  # x concat mean_of_neighbors
            hidden_dim=mlp_hidden_dim,
            out_dim=out_channels,
            hidden_layers=mlp_num_of_hidden_layers
        )

    def forward(self, x, edge_index, edge_weight=None):
        if self.add_self_loop:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        aggr_out = self.propagate(edge_index, x=x)
        out = torch.cat([x, aggr_out], dim=-1)
        out = self.mlp(out)
        return out

    def message(self, x_j):
        return x_j

    def update(self, aggr_out):
        return aggr_out

class GraphormerAttnLayer(torch.nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 attention_dropout: float,
                
                mlp_input_layer_dim:int = None,
                mlp_out_layer_dim:int = None,

                 mlp_num_of_hidden_layers: int = 2,
                 mlp_hidden_dim: int = 16):
        super().__init__()

        self.input_norm = torch.nn.LayerNorm(embed_dim)

        self.attention = torch.nn.MultiheadAttention(
            embed_dim, num_heads, attention_dropout, batch_first=True
        )
        mlp_input_layer_dim = embed_dim if mlp_input_layer_dim is None else mlp_input_layer_dim
        mlp_out_layer_dim = embed_dim if mlp_out_layer_dim is None else mlp_out_layer_dim

        self.mlp = make_mlp(
            num_features=mlp_input_layer_dim,
            hidden_dim=mlp_hidden_dim,
            out_dim=mlp_out_layer_dim,
            hidden_layers=mlp_num_of_hidden_layers
        )

    def forward(self, x, batch=None):
        x = self.input_norm(x)

        if batch is not None:
            x_padded, mask = to_dense_batch(x, batch)
            attn_output, _ = self.attention(
                x_padded, x_padded, x_padded, key_padding_mask=~mask
            )
            x = attn_output[mask]
        else:
            x = x.unsqueeze(0)
            attn_output, _ = self.attention(x, x, x)
            x = attn_output.squeeze(0)

        x = self.mlp(x)
        return x

class KANGPS_Hybrid_layer(torch.nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 mpnn_type: str,
                 attn_num_heads: int,
                 attention_dropout: float,
                 dropout: float,

                 # MLP hyperparams for MPNN:
                 mpnn_mlp_num_of_hidden_layers: int = 2,
                 mpnn_mlp_hidden_dim: int = 16,

                 # MLP hyperparams for attention block:
                 attn_mlp_num_of_hidden_layers: int = 2,
                 attn_mlp_hidden_dim: int = 16,
                 attn_mlp_input_layer_dim:int = None,
                 attn_mlp_out_layer_dim:int = None,

                 # final KAN
                 grid_size: int = 4,
                 spline_order: int = 3,
                 kan_hidden_dim: int = 16,
                 kan_num_of_hidden_layers: int = 2,

                 kan_base_activation: torch.nn.Module = nn.SiLU,
                 *args, **kwargs
                 ):
        super().__init__(*args, **kwargs)

        # 1) MPNN path
        if mpnn_type == 'GCN':
            self.mpnn = GCNLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                mlp_num_of_hidden_layers=mpnn_mlp_num_of_hidden_layers,
                mlp_hidden_dim=mpnn_mlp_hidden_dim
            )
        elif mpnn_type == 'GIN':
            self.mpnn = GINLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                mlp_num_of_hidden_layers=mpnn_mlp_num_of_hidden_layers,
                mlp_hidden_dim=mpnn_mlp_hidden_dim
            )
        elif mpnn_type == 'Sage':
            self.mpnn = SageConvLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                mlp_num_of_hidden_layers=mpnn_mlp_num_of_hidden_layers,
                mlp_hidden_dim=mpnn_mlp_hidden_dim
            )
        else:
            raise ValueError(f"Unsupported MPNN type: {mpnn_type}")

        # 2) Attention path
        self.attention = GraphormerAttnLayer(
            embed_dim=in_channels,
            num_heads=attn_num_heads,
            attention_dropout=attention_dropout,
            mlp_input_layer_dim = attn_mlp_input_layer_dim,
            mlp_out_layer_dim = attn_mlp_out_layer_dim,
            mlp_num_of_hidden_layers=attn_mlp_num_of_hidden_layers,
            mlp_hidden_dim=attn_mlp_hidden_dim
        )

        self.mpnn_batchnorm = nn.BatchNorm1d(out_channels)
        self.attention_batchnorm = nn.BatchNorm1d(out_channels)
        self.dropout = dropout

        # 3) The final KAN
        self.kan = make_kan(
            num_features=out_channels,
            hidden_dim=kan_hidden_dim,
            out_dim=out_channels,
            hidden_layers=kan_num_of_hidden_layers,
            grid_size=grid_size,
            spline_order=spline_order,
            base_activation=kan_base_activation
        )

    def forward(self, x, edge_index, batch=None):
        x_l = x

        # MPNN path
        x_m = self.mpnn(x, edge_index)
        x_m = F.dropout(x_m, p=self.dropout, training=self.training)
        x_m = self.mpnn_batchnorm(x_m)
        if x_m.shape == x_l.shape:
            x_m = x_m + x_l

        # Attention path
        x_t = self.attention(x, batch)
        x_t = F.dropout(x_t, p=self.dropout, training=self.training)
        x_t = self.attention_batchnorm(x_t)
        if x_t.shape == x_l.shape:
            x_t = x_t + x_l

        # Combine paths, apply final KAN
        out = self.kan(x_t + x_m)
        return out

class KANGPS_Hybrid(torch.nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_layers: int,
                 mpnn_type: str = 'GCN',
                 dropout: float = 0.1,
                 attn_num_heads: int = 1,
                 attention_dropout: float = 0.1,

                 # MLP hyperparams for the MPNN
                 mpnn_mlp_num_of_hidden_layers: int = 2,
                 mpnn_mlp_hidden_dim: int = 16,

                 # MLP hyperparams for the attention
                 attn_mlp_num_of_hidden_layers: int = 2,
                 attn_mlp_hidden_dim: int = 16,

                 # final KAN
                 grid_size: int = 4,
                 spline_order: int = 3,
                 kan_hidden_dim: int = 16,
                 kan_num_of_hidden_layers: int = 2,

                 return_embeddings: bool = False,
                 pool: bool = True,
                 final_layer_activation=None,

                 use_atom_encoder: bool = False,
                 atom_encoder_dim: int = None

                 ):
        super().__init__()

        self.return_embeddings = return_embeddings
        self.pool = global_mean_pool if pool else None
        self.final_layer_activation = final_layer_activation

        atom_encoder_dim = hidden_dim if atom_encoder_dim is None else atom_encoder_dim

        self.atom_encoder = optional_atom_encoder(
            use_atom_encoder=use_atom_encoder,
            atom_encoder_dim=atom_encoder_dim
        )

        first_layer_input_dim = atom_encoder_dim if self.atom_encoder else input_dim

        self.layers = nn.ModuleList()

        # 1) First layer: in->hidden
        self.layers.append(
            KANGPS_Hybrid_layer(

                in_channels=first_layer_input_dim,
                out_channels=hidden_dim,
                mpnn_type=mpnn_type,
                attn_num_heads=attn_num_heads,
                attention_dropout=attention_dropout,
                dropout=dropout,

                mpnn_mlp_num_of_hidden_layers=mpnn_mlp_num_of_hidden_layers,
                mpnn_mlp_hidden_dim=mpnn_mlp_hidden_dim,

                attn_mlp_num_of_hidden_layers=attn_mlp_num_of_hidden_layers,
                attn_mlp_hidden_dim=attn_mlp_hidden_dim,
                attn_mlp_input_layer_dim = first_layer_input_dim,
                attn_mlp_out_layer_dim = hidden_dim,

                grid_size=grid_size,
                spline_order=spline_order,
                kan_hidden_dim=kan_hidden_dim,
                kan_num_of_hidden_layers=kan_num_of_hidden_layers
            )
        )

        # 2) Middle layers: hidden->hidden
        for _ in range(num_layers - 2):
            self.layers.append(
                KANGPS_Hybrid_layer(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    mpnn_type=mpnn_type,
                    attn_num_heads=attn_num_heads,
                    attention_dropout=attention_dropout,
                    dropout=dropout,

                    mpnn_mlp_num_of_hidden_layers=mpnn_mlp_num_of_hidden_layers,
                    mpnn_mlp_hidden_dim=mpnn_mlp_hidden_dim,
                    attn_mlp_num_of_hidden_layers=attn_mlp_num_of_hidden_layers,
                    attn_mlp_hidden_dim=attn_mlp_hidden_dim,

                    grid_size=grid_size,
                    spline_order=spline_order,
                    kan_hidden_dim=kan_hidden_dim,
                    kan_num_of_hidden_layers=kan_num_of_hidden_layers
                )
            )

        # 3) Output layer: hidden->output_dim
        self.layers.append(
            KANGPS_Hybrid_layer(
                in_channels=hidden_dim,
                out_channels=output_dim,
                mpnn_type=mpnn_type,
                attn_num_heads=attn_num_heads,
                attention_dropout=attention_dropout,
                dropout=dropout,

                mpnn_mlp_num_of_hidden_layers=mpnn_mlp_num_of_hidden_layers,
                mpnn_mlp_hidden_dim=mpnn_mlp_hidden_dim,
                attn_mlp_num_of_hidden_layers=attn_mlp_num_of_hidden_layers,
                attn_mlp_hidden_dim=attn_mlp_hidden_dim,

                attn_mlp_input_layer_dim = hidden_dim,
                attn_mlp_out_layer_dim = output_dim,

                grid_size=grid_size,
                spline_order=spline_order,
                kan_hidden_dim=kan_hidden_dim,
                kan_num_of_hidden_layers=kan_num_of_hidden_layers
            )
        )

    def forward(self, x, edge_index, positional_encoding=None,
                structural_encoding=None, batch=None):
        
        if self.atom_encoder is not None:
            x = x.long()
            x = self.atom_encoder(x)
    
        embeddings = []

        for layer in self.layers:
            x = layer(x, edge_index, batch)
            if self.return_embeddings:
                embeddings.append(x)

        # Global pooling if desired
        if self.pool is not None:
            x = self.pool(x, batch)

        # Optional final activation
        if self.final_layer_activation is not None:
            x = self.final_layer_activation(x)

        if self.return_embeddings:
            return x, embeddings
        return x
