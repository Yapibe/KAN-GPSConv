import train_model
import argparse
import shlex

def cmd_string_to_args(cmd_string):
    """
    Parses a command-line string into an argparse.Namespace object.

    Args:
        cmd_string (str): Command-line string (e.g., the output of ' '.join(sys.argv[1:]))

    Returns:
        argparse.Namespace: Parsed arguments as a Namespace object.
    """
    # Remove 'python train_model.py' if present
    if cmd_string.startswith('python'):
        parts = cmd_string.split()
        # Remove 'python' and 'train_model.py'
        cmd_string = ' '.join(parts[2:])

    # Split the command-line string into arguments
    arg_list = shlex.split(cmd_string)

    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Train KAN-GPS Network on specified prediction level and task")

    parser.add_argument("--model_name_id", type=str, required=True, help="model name or id")

    parser.add_argument("--prediction_level", type=str, default="node", choices=["node", "graph"],
                        help="Level of prediction: node or graph")
    parser.add_argument("--prediction_task", type=str, default="classification-multiclass",
                        choices=["classification-binary", "classification-multiclass", "regression"],
                        help="Type of prediction task")

    parser.add_argument("--data_source", type=str, default="OGB",
                        choices=["Planetoid", "WebKB", "Actor", "TUDataset", "OGB", "torch_geometric_datasets"],
                        help="Data source")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name")

    # Data Loader arguments
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for graph level task")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--num_neighbors_sample_size", type=int, nargs="+", default=[5, 5],
                        help="Sample sizes for NeighborSampler")

    # Positional and Structural Encoding arguments
    parser.add_argument("--pe_dim", type=int, default=4, help="Dimension of positional encoding")
    parser.add_argument("--concat_pe", type=bool, default=True, help="Concat positional encoding into feature matrix?")
    parser.add_argument("--pos_encoding_type", type=str, default="laplacian_eigenvectors",
                        choices=["sinusoidal", "laplacian_eigenvectors", "RandomWalk"],
                        help="Type of positional encoding")

    parser.add_argument("--se_dim", type=int, default=4, help="Dimension of structural encoding")
    parser.add_argument("--concat_se", type=bool, default=True, help="Concat structural encoding into feature matrix?")
    parser.add_argument("--structural_encoding_type", type=str, default="RWSE",
                        choices=["RWSE", "laplacian_eigenvalues"],
                        help="Type of structural encoding")

    # Model arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay")
    parser.add_argument("--optimizer", type=str, default="SGD", choices=["SGD", "Adam"], help="Optimizer")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD optimizer")
    parser.add_argument("--hidden_channels_dim", type=int, default=16, help="hidden channels dimension")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of layers")
    parser.add_argument("--model_type", type=str, default="kangps", choices=["hybrid", "kangps"],
                        help="Model type: hybrid or kangps")
    parser.add_argument("--model_mpnn_type", type=str, default="GCN", choices=["GCN", "GIN"],
                        help="Model MPNN layer type")

    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--attn_num_heads", type=int, default=1, help="Number of attention heads")
    parser.add_argument("--attention_dropout", type=float, default=0.1, help="Dropout rate in attention layers")

    parser.add_argument("--grid_size", type=int, default=4, help="Grid size for KAN")
    parser.add_argument("--spline_order", type=int, default=3, help="Spline order for KAN")
    parser.add_argument("--kan_hidden_dim", type=int, default=16, help="Hidden dimension for KAN")
    parser.add_argument("--kan_num_of_hidden_layers", type=int, default=2, help="Number of hidden layers in KAN")

    parser.add_argument("--grid_size_attention", type=int, default=4, help="Grid size for attention KAN")
    parser.add_argument("--spline_order_attention", type=int, default=3, help="Spline order for attention KAN")
    parser.add_argument("--kan_hidden_dim_attention", type=int, default=16, help="Hidden dimension for attention KAN")
    parser.add_argument("--kan_num_of_hidden_layers_attention", type=int, default=2,
                        help="Number of hidden layers in attention KAN")

    parser.add_argument("--grid_size_mpnn", type=int, default=4, help="Grid size for MPNN KAN")
    parser.add_argument("--spline_order_mpnn", type=int, default=3, help="Spline order for MPNN KAN")
    parser.add_argument("--kan_hidden_dim_mpnn", type=int, default=16, help="Hidden dimension for MPNN KAN")
    parser.add_argument("--kan_num_of_hidden_layers_mpnn", type=int, default=2,
                        help="Number of hidden layers in MPNN KAN")

    parser.add_argument("--kan_input_layer_dim", type=int, default=None, help="KAN input layer dimension")
    parser.add_argument("--kan_out_layer_dim", type=int, default=None, help="KAN output layer dimension")

    parser.add_argument("--return_embeddings", type=bool, default=False,help="Return embeddings from the model")
    # Parse the arguments
    args = parser.parse_args(arg_list)

    return args

# cmd_string = 'python train_model.py --dataset_name ogbg-molhiv --prediction_level graph --prediction_task classification-binary --data_source OGB --pos_encoding_type laplacian_eigenvectors --structural_encoding_type RWSE --optimizer SGD --model_type kangps --model_mpnn_type GCN --attn_num_heads 1 --model_name_id model_ogbg-molhiv_graph_classification-binary_pos_laplacian_eigenvectors_se_RWSE_opt_SGD_mpnn_GCN_heads_1'
# cmd_string = 'python train_model.py --dataset_name ZINC --prediction_level graph --prediction_task regression --data_source torch_geometric_datasets --pos_encoding_type laplacian_eigenvectors --structural_encoding_type RWSE --optimizer Adam --model_type kangps --model_mpnn_type GCN --attn_num_heads 1 --model_name_id model_ZINC_graph_regression_pos_laplacian_eigenvectors_se_RWSE_opt_Adam_mpnn_GCN_heads_1'
cmd_string = 'python train_model.py --dataset_name ogbn-arxiv --prediction_level node --prediction_task classification-multiclass --data_source OGB --pos_encoding_type laplacian_eigenvectors --structural_encoding_type RWSE --optimizer SGD --model_type kangps --model_mpnn_type GCN --attn_num_heads 1 --model_name_id model_ogbn-arxiv_node_classification-multiclass_pos_laplacian_eigenvectors_se_RWSE_opt_SGD_mpnn_GCN_heads_1'
args = cmd_string_to_args(cmd_string)
train_model.main(args)

