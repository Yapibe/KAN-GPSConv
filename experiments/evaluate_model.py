import torch
import argparse
import json
import os
from utils.data_loader import load_dataset, split_data, prepare_graph_data
from models.kan_gps_model import KANGPSModel
from utils.train_utils import test_node, test_graph
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score


def evaluate(model, data, task, device):
    """
    Evaluate the model on the test set and compute various metrics.

    This function evaluates the model on either node or graph classification tasks
    and computes accuracy, precision, recall, and F1 score.

    Args:
        model (torch.nn.Module): The trained neural network model.
        data (Data or dict): The graph data object for node classification or dictionary of DataLoaders for graph classification.
        task (str): Either 'node' or 'graph' to specify the task type.
        device (torch.device): The device to run the evaluation on.

    Returns:
        tuple: A tuple containing (accuracy, precision, recall, f1).
    """
    if task == "node":
        accuracy = test_node(model, data, data.test_mask)
        y_true = data.y[data.test_mask].cpu().numpy()
        with torch.no_grad():
            out = model(data.x, data.edge_index, data.pos_encoding)
            y_pred = out.argmax(dim=1)[data.test_mask].cpu().numpy()
    else:  # graph
        accuracy = test_graph(model, data["test"], device)
        y_true = []
        y_pred = []
        for batch in data["test"]:
            batch = batch.to(device)
            with torch.no_grad():
                out = model(batch.x, batch.edge_index, batch.pos_encoding)
                y_true.extend(batch.y.cpu().numpy())
                y_pred.extend(out.argmax(dim=1).cpu().numpy())

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted"
    )
    auc = roc_auc_score(y_true, y_pred, average="weighted", multi_class="ovr")

    return accuracy, precision, recall, f1, auc


def main(args):
    """
    Main function to run the evaluation process.

    This function loads a trained model, evaluates it on the specified dataset,
    and saves the evaluation results to a JSON file.

    Args:
        args (argparse.Namespace): Command-line arguments containing evaluation parameters.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and preprocess data
    if args.task == "node":
        data, num_classes, pos_dim = load_dataset(args.dataset, task="node")
        data = split_data(data).to(device)
    else:  # graph
        dataset, num_classes, pos_dim = load_dataset(args.dataset, task="graph")
        _, _, test_loader = prepare_graph_data(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            sample_sizes=args.sample_sizes,
        )
        data = {"test": test_loader}

    # Initialize model
    model = KANGPSModel(
        in_channels=(
            data.num_features if args.task == "node" else dataset.num_node_features
        ),
        hidden_channels=args.hidden_channels,
        out_channels=num_classes,
        num_layers=args.num_layers,
        pos_dim=pos_dim,
    ).to(device)

    # Load trained model weights
    model.load_state_dict(torch.load(args.model_path))

    # Evaluate model
    accuracy, precision, recall, f1, auc = evaluate(model, data, args.task, device)

    # Print results
    print(f"Evaluation results on {args.dataset} dataset:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {auc:.4f}")

    # Save results to JSON file
    results = {
        "dataset": args.dataset,
        "task": args.task,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": auc,
    }

    os.makedirs("results", exist_ok=True)
    with open(f"results/{args.dataset}_{args.task}_evaluation.json", "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to results/{args.dataset}_{args.task}_evaluation.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate KAN-GPS Network on node or graph classification datasets"
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
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model weights",
    )
    parser.add_argument(
        "--hidden_channels", type=int, default=64, help="Number of hidden channels"
    )
    parser.add_argument("--num_layers", type=int, default=3, help="Number of layers")
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
