import torch
import argparse
import wandb
import os
import pandas as pd
from datetime import datetime
from utils.data_loader import load_dataset
from models.kan_gps_model import HybridKANGPS
from models.KANGPS import KANGPS
from utils.train_utils import train_and_evaluate
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric import seed_everything


def main(args):
    seed_everything(args.seed)

    # wandb.init(mode="offline")
    wandb.init(project="KAN-GPSConv", config=args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    dataset, train_loader, valid_loader, test_loader, output_dim = load_dataset(
        root='./data', data_source=args.data_source,
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        prediction_task=args.prediction_task, prediction_level=args.prediction_level,
        concat_pe=args.concat_pe, concat_se=args.concat_se,
        pe_dim=args.pe_dim, pos_encoding_type=args.pos_encoding_type,
        se_dim=args.se_dim, structural_encoding_type=args.structural_encoding_type,
        num_neighbors_sample_size=args.num_neighbors_sample_size, device=device
    )
    # Determine the input dimension
    if args.prediction_level == "node":
        input_dim = dataset.num_features
    else:
        input_dim = dataset.num_node_features

    if args.model_type == "hybrid":
        model = HybridKANGPS(
            in_channels=input_dim,
            hidden_channels=args.hidden_channels,
            out_channels=output_dim,
            num_layers=args.num_layers,
            pos_dim=args.pe_dim,
        ).to(device)
    else:
        model = KANGPS(
            input_dim=input_dim,
            hidden_dim=args.hidden_channels_dim,
            output_dim=output_dim,  # Define output_dim based on your task
            num_layers=args.num_layers,
            mpnn_type=args.model_mpnn_type,
            dropout=args.dropout,
            attn_num_heads=args.attn_num_heads,
            attention_dropout=args.attention_dropout,

            grid_size=args.grid_size,
            spline_order=args.spline_order,
            kan_hidden_dim=args.kan_hidden_dim,
            kan_num_of_hidden_layers=args.kan_num_of_hidden_layers,

            grid_size_attention=args.grid_size_attention,
            spline_order_attention=args.spline_order_attention,
            kan_hidden_dim_attention=args.kan_hidden_dim_attention,
            kan_num_of_hidden_layers_attention=args.kan_num_of_hidden_layers_attention,

            grid_size_mpnn=args.grid_size_mpnn,
            spline_order_mpnn=args.spline_order_mpnn,
            kan_hidden_dim_mpnn=args.kan_hidden_dim_mpnn,
            kan_num_of_hidden_layers_mpnn=args.kan_num_of_hidden_layers_mpnn,

            kan_input_layer_dim=args.kan_input_layer_dim,
            kan_out_layer_dim=args.kan_out_layer_dim,

            return_embeddings=args.return_embeddings,
            pool=args.prediction_level == 'graph',
            final_layer_activation=None if args.prediction_task == 'regression' else torch.nn.Softmax(dim=1)
        ).to(device)

    wandb.config.update({
        "model_name_id": args.model_name_id,
        "model_type": args.model_type,
        "model_MPNN_layer_type": args.model_mpnn_type,
        "dataset": args.dataset_name,
        "data_source": args.data_source,
        "prediction_level": args.prediction_level,
        "prediction_task": args.prediction_task,
        "num_features": dataset.num_features if args.prediction_level == "node" else dataset.num_node_features,
        "num_classes": output_dim,
        "positional_encoding_dim": args.pe_dim,
        "positional_encoding_type": args.pos_encoding_type,
        "structural_encoding_dim": args.se_dim,
        "structural_encoding_type": args.structural_encoding_type,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "optimizer": args.optimizer,
        "momentum": args.momentum,
        "num_epochs": args.epochs,
        "model_number_of_params": params_count(model),

        # New KANGPS parameters
        "dropout": args.dropout,
        "attn_num_heads": args.attn_num_heads,
        "attention_dropout": args.attention_dropout,

        "grid_size": args.grid_size,
        "spline_order": args.spline_order,
        "kan_hidden_dim": args.kan_hidden_dim,
        "kan_num_of_hidden_layers": args.kan_num_of_hidden_layers,

        "grid_size_attention": args.grid_size_attention,
        "spline_order_attention": args.spline_order_attention,
        "kan_hidden_dim_attention": args.kan_hidden_dim_attention,
        "kan_num_of_hidden_layers_attention": args.kan_num_of_hidden_layers_attention,

        "grid_size_mpnn": args.grid_size_mpnn,
        "spline_order_mpnn": args.spline_order_mpnn,
        "kan_hidden_dim_mpnn": args.kan_hidden_dim_mpnn,
        "kan_num_of_hidden_layers_mpnn": args.kan_num_of_hidden_layers_mpnn,

        "kan_input_layer_dim": args.kan_input_layer_dim,
        "kan_out_layer_dim": args.kan_out_layer_dim,
        "return_embeddings": args.return_embeddings
    })

    if args.optimizer == "SGD":
        optimizer = getattr(torch.optim, args.optimizer)(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=args.momentum if args.optimizer == "SGD" else 0
        )
    else:
        optimizer = getattr(torch.optim, args.optimizer)(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )



    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    test_metric_result, best_model_state, test_res_df = train_and_evaluate(
        model=model,
        train_loader=train_loader, valid_loader=valid_loader, test_loader=test_loader,
        optimizer=optimizer, scheduler=scheduler, args=args, device=device,
        output_dim=output_dim)

    result_dir = f"data/{args.dataset_name}/results/{args.model_name_id}"
    os.makedirs(result_dir, exist_ok=True)
    torch.save(best_model_state, result_dir + "/best_model_state.pt")
    test_res_df.to_csv(result_dir + "/test_res_df.csv")
    wandb.log({"test_metric_result": test_metric_result})
    wandb.finish()


if __name__ == "__main__":
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
    parser.add_argument("--num_neighbors_sample_size", type=int, nargs="+", default=[10, 10],
                        help="Sample sizes for NeighborSampler")

    # Positional and Structural Encoding arguments
    parser.add_argument("--pe_dim", type=int, default=8, help="Dimension of positional encoding")
    parser.add_argument("--concat_pe", type=bool, default=True, help="Concat positional encoding into feature matrix?")
    parser.add_argument("--pos_encoding_type", type=str, default="laplacian_eigenvectors",
                        choices=["sinusoidal", "laplacian_eigenvectors", "RandomWalk"],
                        help="Type of positional encoding")

    parser.add_argument("--se_dim", type=int, default=8, help="Dimension of structural encoding")
    parser.add_argument("--concat_se", type=bool, default=True, help="Concat structural encoding into feature matrix?")
    parser.add_argument("--structural_encoding_type", type=str, default="RWSE",
                        choices=["RWSE", "laplacian_eigenvalues"],
                        help="Type of structural encoding")

    # Model arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay")
    parser.add_argument("--optimizer", type=str, default="SGD", choices=["SGD", "Adam"], help="Optimizer")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD optimizer")
    parser.add_argument("--hidden_channels_dim", type=int, default=64, help="hidden channels dimension")
    parser.add_argument("--num_layers", type=int, default=10, help="Number of layers")
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

    args = parser.parse_args()
    main(args)
