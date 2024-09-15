import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

def train_node(model, data, optimizer):
    """
    Train the model for one epoch on node classification task.

    Args:
        model (torch.nn.Module): The neural network model.
        data (Data): The graph data object.
        optimizer (torch.optim.Optimizer): The optimizer for model parameters.

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
    Train the model for one epoch on graph classification task.

    Args:
        model (torch.nn.Module): The neural network model.
        loader (DataLoader): DataLoader for the graph dataset.
        optimizer (torch.optim.Optimizer): The optimizer for model parameters.
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
    Evaluate the model on node classification task.

    Args:
        model (torch.nn.Module): The neural network model.
        data (Data): The graph data object.
        mask (torch.Tensor): Boolean mask indicating which nodes to evaluate.

    Returns:
        float: The accuracy of the model on the specified nodes.
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
    Evaluate the model on graph classification task.

    Args:
        model (torch.nn.Module): The neural network model.
        loader (DataLoader): DataLoader for the graph dataset.
        device (torch.device): The device to run the evaluation on.

    Returns:
        float: The accuracy of the model on the graph classification task.
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

def train_and_evaluate(model, data, optimizer, scheduler, args, device, task='node'):
    """
    Train and evaluate the model for the specified number of epochs.

    Args:
        model (torch.nn.Module): The neural network model.
        data (Data or dict): The graph data object for node classification or dictionary of DataLoaders for graph classification.
        optimizer (torch.optim.Optimizer): The optimizer for model parameters.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        args (argparse.Namespace): Command-line arguments.
        device (torch.device): The device to run the training and evaluation on.
        task (str): Either 'node' or 'graph' to specify the task type.

    Returns:
        float: The test accuracy of the best model.
    """
    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(args.epochs):
        if task == 'node':
            loss = train_node(model, data, optimizer)
            train_acc = test_node(model, data, data.train_mask)
            val_acc = test_node(model, data, data.val_mask)
        else:  # graph
            loss = train_graph(model, data['train'], optimizer, device)
            train_acc = test_graph(model, data['train'], device)
            val_acc = test_graph(model, data['val'], device)

        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()

        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')

    model.load_state_dict(best_model_state)
    if task == 'node':
        test_acc = test_node(model, data, data.test_mask)
    else:  # graph
        test_acc = test_graph(model, data['test'], device)

    print(f'Test Accuracy: {test_acc:.4f}')
    return test_acc, best_model_state
