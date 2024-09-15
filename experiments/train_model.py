import torch
import argparse
import wandb
import os
from utils.data_loader import load_dataset, split_data, prepare_graph_data
from models.kan_gps_model import HybridKANGPS, KANGPS
from utils.train_utils import train_and_evaluate


def main(args):
    """
    Main function to run the training and evaluation process.

    Args:
        args (argparse.Namespace): Command-line arguments containing training parameters.
    """
    wandb.init(project="KAN-GPSConv", config=args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and preprocess data
    if args.task == "node":
        data, num_classes, pos_dim = load_dataset(
            args.dataset,
            task="node",
            pos_dim=args.pos_dim,
            pos_encoding_type=args.pos_encoding_type,
        )
        data = split_data(data).to(device)
    else:  # graph
        dataset, num_classes, pos_dim = load_dataset(
            args.dataset,
            task="graph",
            pos_dim=args.pos_dim,
            pos_encoding_type=args.pos_encoding_type,
        )
        train_loader, val_loader, test_loader = prepare_graph_data(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            sample_sizes=args.sample_sizes,
        )
        data = {"train": train_loader, "val": val_loader, "test": test_loader}

    # Update wandb config
    wandb.config.update(
        {
            "model_type": "KAN-GPS",
            "dataset": args.dataset,
            "task": args.task,
            "num_features": (
                data.num_features if args.task == "node" else dataset.num_node_features
            ),
            "num_classes": num_classes,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "optimizer": args.optimizer,
            "momentum": args.momentum,
            "num_epochs": args.epochs,
        }
    )

    # Initialize model
    if args.model_type == "hybrid":
        model = HybridKANGPS(
            in_channels=(
                data.num_features if args.task == "node" else dataset.num_node_features
            ),
            hidden_channels=args.hidden_channels,
            out_channels=num_classes,
            num_layers=args.num_layers,
            pos_dim=pos_dim,
        ).to(device)
    else:  # 'kangps'
        model = KANGPS(
            in_channels=(
                data.num_features if args.task == "node" else dataset.num_node_features
            ),
            hidden_channels=args.hidden_channels,
            out_channels=num_classes,
            num_layers=args.num_layers,
            pos_dim=pos_dim,
            num_functions=args.num_functions,
        ).to(device)

    # Initialize optimizer and scheduler
    optimizer = getattr(torch.optim, args.optimizer)(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum if args.optimizer == "SGD" else 0,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    # Train and evaluate
    test_acc, best_model_state = train_and_evaluate(
        model, data, optimizer, scheduler, args, device, task=args.task
    )

    # Save the best model checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(
        best_model_state, f"checkpoints/{args.dataset}_{args.task}_{args.model_type}.pt"
    )

    wandb.log({"test_accuracy": test_acc})
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train KAN-GPS Network on node or graph classification datasets"
    )
    parser.add_argument("--dataset", type=str, default="Cora", help="Dataset name")
    parser.add_argument(
        "--task",
        type=str,
        default="node",
        choices=["node", "graph"],
        help="Task type: node or graph classification",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs to train"
    )
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay")
    parser.add_argument(
        "--optimizer",
        type=str,
        default="SGD",
        choices=["SGD", "Adam"],
        help="Optimizer",
    )
    parser.add_argument(
        "--momentum", type=float, default=0.9, help="Momentum for SGD optimizer"
    )
    parser.add_argument(
        "--hidden_channels", type=int, default=64, help="Number of hidden channels"
    )
    parser.add_argument("--num_layers", type=int, default=3, help="Number of layers")
    parser.add_argument(
        "--model_type",
        type=str,
        default="hybrid",
        choices=["hybrid", "kangps"],
        help="Model type: hybrid or kangps",
    )
    parser.add_argument(
        "--num_functions",
        type=int,
        default=4,
        help="Number of univariate functions in MultiKANLayer (only for KANGPS model)",
    )
    parser.add_argument(
        "--conv_type",
        type=str,
        default="GCN",
        choices=["GCN", "GAT", "SAGE"],
        help="Convolution type for GPS layers",
    )
    parser.add_argument(
        "--pos_dim", type=int, default=64, help="Dimension of positional encoding"
    )
    parser.add_argument(
        "--pos_encoding_type",
        type=str,
        default="sinusoidal",
        choices=["sinusoidal", "laplacian"],
        help="Type of positional encoding",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for graph classification"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for data loading"
    )
    parser.add_argument(
        "--sample_sizes",
        type=int,
        nargs="+",
        default=[10, 10],
        help="Sample sizes for NeighborSampler",
    )
    args = parser.parse_args()

    main(args)
