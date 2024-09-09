import torch
from torch_geometric.datasets import Planetoid, WebKB, Actor, TUDataset
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.loader import DataLoader
from ogb.nodeproppred import PygNodePropPredDataset

def load_dataset(name, root='./data', task='node'):
    """
    Load a dataset for node or graph classification.
    
    Args:
    - name (str): Name of the dataset
    - root (str): Root directory where the dataset should be saved
    - task (str): 'node' for node classification, 'graph' for graph classification
    
    Returns:
    - data: The loaded dataset
    - num_classes (int): Number of classes in the dataset
    """
    transform = NormalizeFeatures()
    
    if task == 'node':
        if name in ['Cora', 'Citeseer', 'PubMed']:
            dataset = Planetoid(root=root, name=name, transform=transform)
        elif name in ['Cornell', 'Texas', 'Wisconsin']:
            dataset = WebKB(root=root, name=name, transform=transform)
        elif name == 'Actor':
            dataset = Actor(root=root, transform=transform)
        elif name == 'ogbn-arxiv':
            dataset = PygNodePropPredDataset(name=name, root=root, transform=transform)
        else:
            raise ValueError(f"Node classification dataset {name} not recognized")
        
        data = dataset[0]
        num_classes = dataset.num_classes
        return data, num_classes
    
    elif task == 'graph':
        dataset = TUDataset(root=root, name=name, transform=transform)
        num_classes = dataset.num_classes
        return dataset, num_classes
    
    else:
        raise ValueError("Task must be either 'node' or 'graph'")

def split_data(data, val_ratio=0.1, test_ratio=0.1):
    """
    Split the data into train, validation, and test sets for node classification.
    
    Args:
    - data: PyG Data object
    - val_ratio (float): Ratio of validation set
    - test_ratio (float): Ratio of test set
    
    Returns:
    - data with train_mask, val_mask, and test_mask
    """
    num_nodes = data.num_nodes
    indices = torch.randperm(num_nodes)
    test_size = int(num_nodes * test_ratio)
    val_size = int(num_nodes * val_ratio)
    
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask[indices[:test_size]] = True
    
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask[indices[test_size:test_size+val_size]] = True
    
    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.train_mask[indices[test_size+val_size:]] = True
    
    return data
