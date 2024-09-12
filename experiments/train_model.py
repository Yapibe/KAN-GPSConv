import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from utils.data_loader import load_dataset, split_data, prepare_graph_data
from models.gps_layer import GPSNetwork
import argparse
import wandb

def train_node(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def train_graph(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

def test_node(model, data, mask):
    model.eval()
    with torch.no_grad():
        out = model(data)
        pred = out.argmax(dim=1)
        correct = pred[mask] == data.y[mask]
        acc = int(correct.sum()) / int(mask.sum())
    return acc

def test_graph(model, loader, device):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out = model(data)
            pred = out.argmax(dim=1)
            correct += int((pred == data.y).sum())
    return correct / len(loader.dataset)

def main(args):
    # Initialize wandb
    wandb.init(project="KAN-GPSConv", config=args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.task == 'node':
        # Load and preprocess data for node classification
        data, num_classes = load_dataset(args.dataset, task='node')
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
        model = GPSNetwork(data.num_features, num_classes, task='node').to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        
        # Training loop
        for epoch in range(200):
            loss = train_node(model, data, optimizer)
            train_acc = test_node(model, data, data.train_mask)
            val_acc = test_node(model, data, data.val_mask)
            test_acc = test_node(model, data, data.test_mask)
            
            # Log metrics to wandb
            wandb.log({
                "epoch": epoch,
                "loss": loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "test_acc": test_acc,
                "gpu_memory_allocated": torch.cuda.memory_allocated(),
                "gpu_memory_cached": torch.cuda.memory_cached()
            })
            
            if epoch % 10 == 0:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')
    
    elif args.task == 'graph':
        # Load and preprocess data for graph classification
        dataset, num_classes = load_dataset(args.dataset, task='graph')
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
        model = GPSNetwork(dataset.num_node_features, num_classes, task='graph').to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        
        # Training loop
        for epoch in range(200):
            loss = train_graph(model, train_loader, optimizer, device)
            train_acc = test_graph(model, train_loader, device)
            val_acc = test_graph(model, val_loader, device)
            test_acc = test_graph(model, test_loader, device)
            
            # Log metrics to wandb
            wandb.log({
                "epoch": epoch,
                "loss": loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "test_acc": test_acc,
                "gpu_memory_allocated": torch.cuda.memory_allocated(),
                "gpu_memory_cached": torch.cuda.memory_cached()
            })
            
            if epoch % 10 == 0:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')
    
    else:
        raise ValueError("Task must be either 'node' or 'graph'")
    
    # Close wandb run
    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train GPS Network on node or graph classification datasets')
    parser.add_argument('--dataset', type=str, default='Cora', help='Dataset name')
    parser.add_argument('--task', type=str, default='node', choices=['node', 'graph'], help='Task type: node or graph classification')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train')
    args = parser.parse_args()
    
    main(args)
