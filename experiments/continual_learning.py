import torch
import argparse
import json
import os
from utils.data_loader import load_dataset, split_data, prepare_graph_data
from models.kan_gps_model import HybridKANGPS, KANGPS
from utils.train_utils import train_node, test_node, train_graph, test_graph


def train_on_task(model, data, optimizer, scheduler, args, device, task="node"):
    """
    Train the model on a single task.

    Args:
        model (torch.nn.Module): The neural network model.
        data (Data or dict): The graph data object for node classification or dictionary of DataLoaders for graph classification.
        optimizer (torch.optim.Optimizer): The optimizer for model parameters.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        args (argparse.Namespace): Command-line arguments.
        device (torch.device): The device to run the training on.
        task (str): Either 'node' or 'graph' to specify the task type.

    Returns:
        tuple: A tuple containing (trained_model, best_val_acc).
    """
    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(args.epochs):
        if task == "node":
            loss = train_node(model, data, optimizer)
            val_acc = test_node(model, data, data.val_mask)
        else:  # graph
            loss = train_graph(model, data["train"], optimizer, device)
            val_acc = test_graph(model, data["val"], device)

        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()

        if epoch % 10 == 0:
            print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}")

    model.load_state_dict(best_model_state)
    return model, best_val_acc


def evaluate_backward_transfer(model, data_list, device, task="node"):
    """
    Evaluate the model's backward transfer on previously learned tasks.

    Args:
        model (torch.nn.Module): The trained neural network model.
        data_list (list): List of graph data objects or dictionaries of DataLoaders for each task.
        device (torch.device): The device to run the evaluation on.
        task (str): Either 'node' or 'graph' to specify the task type.

    Returns:
        list: List of accuracies for each previously learned task.
    """
    model.eval()
    accuracies = []

    with torch.no_grad():
        for data in data_list:
            if task == "node":
                acc = test_node(model, data, data.test_mask)
            else:  # graph
                acc = test_graph(model, data["test"], device)
            accuracies.append(acc)

    return accuracies


def main(args):
    """
    Main function to run the continual learning evaluation.

    This function trains the model sequentially on multiple tasks and evaluates
    its performance on both new and previously learned tasks.

    Args:
        args (argparse.Namespace): Command-line arguments containing evaluation parameters.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and preprocess data for each task
    data_list = []
    for dataset_name in args.datasets:
        if args.task == "node":
            data, num_classes, pos_dim = load_dataset(dataset_name, task="node")
            data = split_data(data).to(device)
            data_list.append(data)
        else:  # graph
            dataset, num_classes, pos_dim = load_dataset(dataset_name, task="graph")
            data = prepare_graph_data(dataset)
            data_list.append(data)

    # Initialize model
    if args.model_type == "hybrid":
        model = HybridKANGPS(
            in_channels=(
                data_list[0].num_features
                if args.task == "node"
                else dataset.num_node_features
            ),
            hidden_channels=args.hidden_channels,
            out_channels=num_classes,
            num_layers=args.num_layers,
            pos_dim=pos_dim,
        ).to(device)
    else:  # 'kangps'
        model = KANGPS(
            in_channels=(
                data_list[0].num_features
                if args.task == "node"
                else dataset.num_node_features
            ),
            hidden_channels=args.hidden_channels,
            out_channels=num_classes,
            num_layers=args.num_layers,
            pos_dim=pos_dim,
            num_functions=args.num_functions,
        ).to(device)

    # Train and evaluate on each task sequentially
    accuracies = []
    backward_transfer_accuracies = []

    for i, data in enumerate(data_list):
        print(f"Training on task {i+1}...")

        # Initialize optimizer and scheduler for each task
        optimizer = getattr(torch.optim, args.optimizer)(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=args.momentum if args.optimizer == "SGD" else 0,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

        # Train on the current task
        model, best_val_acc = train_on_task(
            model, data, optimizer, scheduler, args, device, task=args.task
        )

        # Evaluate on the current task
        if args.task == "node":
            test_acc = test_node(model, data, data.test_mask)
        else:  # graph
            test_acc = test_graph(model, data["test"], device)
        accuracies.append(test_acc)

        # Evaluate backward transfer on previous tasks
        backward_transfer_accs = evaluate_backward_transfer(
            model, data_list[:i], device, task=args.task
        )
        backward_transfer_accuracies.append(backward_transfer_accs)

        print(f"Task {i+1} - Test Accuracy: {test_acc:.4f}")
        print(f"Backward Transfer Accuracies: {backward_transfer_accs}")

    # Print final results
    print("Final Accuracies:")
    for i, acc in enumerate(accuracies):
        print(f"Task {i+1}: {acc:.4f}")

    print("Backward Transfer Accuracies:")
    for i, accs in enumerate(backward_transfer_accuracies):
        print(f"Task {i+1}: {accs}")

    # Save results to JSON file
    results = {
        "datasets": args.datasets,
        "task": args.task,
        "accuracies": accuracies,
        "backward_transfer_accuracies": backward_transfer_accuracies,
    }

    os.makedirs("results", exist_ok=True)
    with open(f"results/continual_learning_{args.task}.json", "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to results/continual_learning_{args.task}.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Continual Learning Evaluation for KAN-GPS Network"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["Cora", "Citeseer", "PubMed"],
        help="List of dataset names",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="node",
        choices=["node", "graph"],
        help="Task type: node or graph classification",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="hybrid",
        choices=["hybrid", "kangps"],
        help="Model type: hybrid or kangps",
    )
    parser.add_argument(
        "--epochs", type=int, default=200, help="Number of epochs to train"
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
        "--num_functions",
        type=int,
        default=4,
        help="Number of univariate functions in MultiKANLayer (only for KANGPS model)",
    )
    args = parser.parse_args()

    main(args)
