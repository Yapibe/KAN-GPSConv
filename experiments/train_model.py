import torch
from utils.data_loader import load_dataset, split_data, prepare_graph_data
from models.kan_gps_model import KANGPSModel
import torch.nn.functional as F
import argparse
import wandb

def train_node(model, data, optimizer):
    """
    Trains the model on a node classification task for one epoch.

    Args:
        model (nn.Module): The neural network model.
        data (Data): The data object containing the graph.
        optimizer (Optimizer): The optimizer used for training.

    Returns:
        float: The training loss for this epoch.
    """
    model.train()
    optimizer.zero_grad()
    pos_encoding = data.pos_encoding if hasattr(data, 'pos_encoding') else None
    out = model(data.x, data.edge_index, pos_encoding)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def train_graph(model, loader, optimizer, device):
    """
    Trains the model on a graph classification task for one epoch.

    Args:
        model (nn.Module): The neural network model.
        loader (DataLoader): DataLoader for the training data.
        optimizer (Optimizer): The optimizer used for training.
        device (torch.device): The device to run the training on.

    Returns:
        float: The average training loss for this epoch.
    """
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        pos_encoding = data.pos_encoding if hasattr(data, 'pos_encoding') else None
        out = model(data.x, data.edge_index, pos_encoding)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

def test_node(model, data, mask):
    """
    Evaluates the model on a node classification task.

    Args:
        model (nn.Module): The neural network model.
        data (Data): The data object containing the graph.
        mask (Tensor): The mask indicating which nodes to evaluate.

    Returns:
        float: The accuracy on the specified nodes.
    """
    model.eval()
    with torch.no_grad():
        pos_encoding = data.pos_encoding if hasattr(data, 'pos_encoding') else None
        out = model(data.x, data.edge_index, pos_encoding)
        pred = out.argmax(dim=1)
        correct = pred[mask] == data.y[mask]
        acc = int(correct.sum()) / int(mask.sum())
    return acc

def test_graph(model, loader, device):
    """
    Evaluates the model on a graph classification task.

    Args:
        model (nn.Module): The neural network model.
        loader (DataLoader): DataLoader for the evaluation data.
        device (torch.device): The device to run the evaluation on.

    Returns:
        float: The accuracy over the dataset.
    """
    model.eval()
    correct = 0
    total = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pos_encoding = data.pos_encoding if hasattr(data, 'pos_encoding') else None
            out = model(data.x, data.edge_index, pos_encoding)
            pred = out.argmax(dim=1)
            correct += (pred == data.y).float().sum().item()
            total += data.num_graphs
    return correct / total

def main(args):
    # Initialize wandb
    wandb.init(project="KAN-GPSConv", config=args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.task == 'node':
        # Load and preprocess data for node classification
        data, num_classes, pos_dim = load_dataset(args.dataset, task='node')
        data = split_data(data)
        data = data.to(device)
        
        wandb.config.update({
            "model_type": "KAN-GPS",
            "dataset": args.dataset,
            "task": args.task,
            "num_features": data.num_features,
            "num_classes": num_classes,
            "learning_rate": 0.01,
            "weight_decay": 5e-4,
            "optimizer": "SGD",
            "momentum": 0.9,
            "num_epochs": args.epochs
        })

        # Initialize model
        model = KANGPSModel(
            in_channels=data.num_features,
            hidden_channels=64,
            out_channels=num_classes,
            num_layers=3,
            pos_dim=pos_dim
        ).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

        # Training loop
        best_val_acc = 0.0
        best_model_state = None

        for epoch in range(args.epochs):
            loss = train_node(model, data, optimizer)
            train_acc = test_node(model, data, data.train_mask)
            val_acc = test_node(model, data, data.val_mask)
            scheduler.step()

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict()

            # Log metrics
            wandb.log({
                "epoch": epoch,
                "loss": loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "gpu_memory_allocated": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
                "gpu_memory_reserved": torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
            })

            if epoch % 10 == 0:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')

        # Load best model before testing
        model.load_state_dict(best_model_state)
        test_acc = test_node(model, data, data.test_mask)
        print(f'Test Accuracy: {test_acc:.4f}')

    elif args.task == 'graph':
        # Load and preprocess data for graph classification
        dataset, num_classes, pos_dim = load_dataset(args.dataset, task='graph')
        train_loader, val_loader, test_loader = prepare_graph_data(dataset)
        
        wandb.config.update({
            "model_type": "KAN-GPS",
            "dataset": args.dataset,
            "task": args.task,
            "num_features": dataset.num_node_features,
            "num_classes": num_classes,
            "learning_rate": 0.01,
            "weight_decay": 5e-4,
            "optimizer": "SGD",
            "momentum": 0.9,
            "num_epochs": args.epochs,
            "batch_size": train_loader.batch_size
        })

        # Initialize model
        model = KANGPSModel(
            in_channels=dataset.num_node_features,
            hidden_channels=64,
            out_channels=num_classes,
            num_layers=3,
            pos_dim=pos_dim
        ).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

        # Training loop
        best_val_acc = 0.0
        best_model_state = None

        for epoch in range(args.epochs):
            loss = train_graph(model, train_loader, optimizer, device)
            train_acc = test_graph(model, train_loader, device)
            val_acc = test_graph(model, val_loader, device)
            scheduler.step()

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict()

            # Log metrics
            wandb.log({
                "epoch": epoch,
                "loss": loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "gpu_memory_allocated": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
                "gpu_memory_reserved": torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
            })

            if epoch % 10 == 0:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')

        # Load best model before testing
        model.load_state_dict(best_model_state)
        test_acc = test_graph(model, test_loader, device)
        print(f'Test Accuracy: {test_acc:.4f}')

    else:
        raise ValueError("Task must be either 'node' or 'graph'")
    
    # Close wandb run
    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train KAN-GPS Network on node or graph classification datasets')
    parser.add_argument('--dataset', type=str, default='Cora', help='Dataset name')
    parser.add_argument('--task', type=str, default='node', choices=['node', 'graph'], help='Task type: node or graph classification')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    args = parser.parse_args()
    
    main(args)
