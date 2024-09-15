import torch
from torch_sparse import SparseTensor
from torch_geometric.datasets import Planetoid, WebKB, Actor, TUDataset
from torch_geometric.transforms import NormalizeFeatures
import math
from torch_geometric.data import NeighborSampler
from ogb.nodeproppred import PygNodePropPredDataset
from ogb.graphproppred import PygGraphPropPredDataset
from torch.utils.data import random_split
from torch_geometric.utils import get_laplacian


def generate_positional_encoding(num_nodes, pos_dim):
    position = torch.arange(0, num_nodes, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, pos_dim, 2).float() * (-math.log(10000.0) / pos_dim)
    )
    pe = torch.zeros(num_nodes, pos_dim)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def generate_laplacian_positional_encoding(edge_index, num_nodes, pos_dim):
    edge_weight = torch.ones((edge_index.size(1),), dtype=torch.float)
    L = get_laplacian(edge_index, edge_weight, normalization="sym", num_nodes=num_nodes)
    _, eigenvectors = torch.linalg.eigh(L.to_dense())
    pe = eigenvectors[:, :pos_dim]
    return pe


def load_dataset(
    name, root="./data", task="node", pos_dim=64, pos_encoding_type="sinusoidal"
):
    """
    Load a dataset for node or graph classification.
    """
    transform = NormalizeFeatures()

    if task == "node":
        # Handle OGB node property prediction datasets
        if name.startswith("ogbn-"):
            dataset = PygNodePropPredDataset(name=name, root=root, transform=transform)
            data = dataset[0]
            num_classes = dataset.num_classes

            # Get split indices
            split_idx = dataset.get_idx_split()
            train_idx = split_idx["train"]
            val_idx = split_idx["valid"]
            test_idx = split_idx["test"]

            # Create masks
            num_nodes = data.num_nodes
            train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            train_mask[train_idx] = True

            val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            val_mask[val_idx] = True

            test_mask = torch.zeros(num_nodes, dtype=torch.bool)
            test_mask[test_idx] = True

            data.train_mask = train_mask
            data.val_mask = val_mask
            data.test_mask = test_mask

            # OGB datasets may have labels as an attribute 'y', but sometimes as a tuple
            if isinstance(data.y, torch.Tensor):
                data.y = data.y.squeeze()
            else:
                data.y = data.y[0].squeeze()

            # Generate positional encoding
            if pos_encoding_type == "sinusoidal":
                data.pos_encoding = generate_positional_encoding(
                    data.num_nodes, pos_dim
                )
            elif pos_encoding_type == "laplacian":
                data.pos_encoding = generate_laplacian_positional_encoding(
                    data.edge_index, data.num_nodes, pos_dim
                )
            else:
                raise ValueError(
                    f"Unsupported positional encoding type: {pos_encoding_type}"
                )

            # Convert edge_index to SparseTensor
            data.edge_index = SparseTensor(
                row=data.edge_index[0],
                col=data.edge_index[1],
                sparse_sizes=(data.num_nodes, data.num_nodes),
            )

            return data, num_classes, pos_dim

        elif name in ["Cora", "Citeseer", "PubMed"]:
            dataset = Planetoid(root=root, name=name, transform=transform)
            data = dataset[0]
            num_classes = dataset.num_classes
            # Generate positional encoding
            data.pos_encoding = generate_positional_encoding(data.num_nodes, pos_dim)
            return data, num_classes, pos_dim

        elif name in ["Cornell", "Texas", "Wisconsin"]:
            dataset = WebKB(root=root, name=name, transform=transform)
            data = dataset[0]
            num_classes = dataset.num_classes
            # Generate positional encoding
            data.pos_encoding = generate_positional_encoding(data.num_nodes, pos_dim)
            return data, num_classes, pos_dim

        elif name == "Actor":
            dataset = Actor(root=root, transform=transform)
            data = dataset[0]
            num_classes = dataset.num_classes
            # Generate positional encoding
            data.pos_encoding = generate_positional_encoding(data.num_nodes, pos_dim)
            return data, num_classes, pos_dim

        else:
            raise ValueError(f"Node classification dataset {name} not recognized")

    elif task == "graph":
        # Handle OGB graph property prediction datasets
        if name.startswith("ogbg-"):
            dataset = PygGraphPropPredDataset(name=name, root=root, transform=transform)
            num_classes = dataset.num_tasks  # For multi-task datasets
            # Generate positional encodings for each graph in the dataset
            if pos_encoding_type == "sinusoidal":
                for data in dataset:
                    data.pos_encoding = generate_positional_encoding(
                        data.num_nodes, pos_dim
                    )
            elif pos_encoding_type == "laplacian":
                for data in dataset:
                    data.pos_encoding = generate_laplacian_positional_encoding(
                        data.edge_index, data.num_nodes, pos_dim
                    )
            else:
                raise ValueError(
                    f"Unsupported positional encoding type: {pos_encoding_type}"
                )
            # OGB datasets provide predefined splits
            split_idx = dataset.get_idx_split()
            train_idx = split_idx["train"]
            val_idx = split_idx["valid"]
            test_idx = split_idx["test"]
            train_dataset = dataset[train_idx]
            val_dataset = dataset[val_idx]
            test_dataset = dataset[test_idx]

            # Convert edge_index to SparseTensor for each graph
            if isinstance(dataset, tuple):
                for data in train_dataset:
                    data.edge_index = SparseTensor(
                        row=data.edge_index[0],
                        col=data.edge_index[1],
                        sparse_sizes=(data.num_nodes, data.num_nodes),
                    )
                for data in val_dataset:
                    data.edge_index = SparseTensor(
                        row=data.edge_index[0],
                        col=data.edge_index[1],
                        sparse_sizes=(data.num_nodes, data.num_nodes),
                    )
                for data in test_dataset:
                    data.edge_index = SparseTensor(
                        row=data.edge_index[0],
                        col=data.edge_index[1],
                        sparse_sizes=(data.num_nodes, data.num_nodes),
                    )
            else:
                for data in dataset:
                    data.edge_index = SparseTensor(
                        row=data.edge_index[0],
                        col=data.edge_index[1],
                        sparse_sizes=(data.num_nodes, data.num_nodes),
                    )

            return (train_dataset, val_dataset, test_dataset), num_classes, pos_dim

        else:
            # Handle other graph datasets
            dataset = TUDataset(root=root, name=name, transform=transform)
            num_classes = dataset.num_classes
            # Generate positional encodings for each graph
            if pos_encoding_type == "sinusoidal":
                for data in dataset:
                    data.pos_encoding = generate_positional_encoding(
                        data.num_nodes, pos_dim
                    )
            elif pos_encoding_type == "laplacian":
                for data in dataset:
                    data.pos_encoding = generate_laplacian_positional_encoding(
                        data.edge_index, data.num_nodes, pos_dim
                    )
            else:
                raise ValueError(
                    f"Unsupported positional encoding type: {pos_encoding_type}"
                )
            return dataset, num_classes, pos_dim

    else:
        raise ValueError("Task must be either 'node' or 'graph'")


def split_data(data, val_ratio=0.1, test_ratio=0.1):
    """
    Split the data into train, validation, and test sets for node classification.
    If data already has train_mask, val_mask, and test_mask, it will return data as is.
    """
    if (
        hasattr(data, "train_mask")
        and hasattr(data, "val_mask")
        and hasattr(data, "test_mask")
    ):
        # Data already has predefined splits
        return data
    else:
        # Create new splits
        num_nodes = data.num_nodes
        indices = torch.randperm(num_nodes)
        test_size = int(num_nodes * test_ratio)
        val_size = int(num_nodes * val_ratio)

        data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.test_mask[indices[:test_size]] = True

        data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.val_mask[indices[test_size : test_size + val_size]] = True

        data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.train_mask[indices[test_size + val_size :]] = True

        return data


def prepare_graph_data(dataset, batch_size=32, num_workers=4, sample_sizes=[10, 10]):
    if isinstance(dataset, tuple):
        # For datasets with predefined splits (e.g., OGB)
        train_dataset, val_dataset, test_dataset = dataset

        # Create NeighborSampler for each dataset with multiple workers
        train_loader = NeighborSampler(
            train_dataset,
            sizes=sample_sizes,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        val_loader = NeighborSampler(
            val_dataset,
            sizes=sample_sizes,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        test_loader = NeighborSampler(
            test_dataset,
            sizes=sample_sizes,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
    else:
        # For other datasets, create random splits
        total_size = len(dataset)
        num_training = int(total_size * 0.8)
        num_val = int(total_size * 0.1)
        num_test = total_size - num_training - num_val

        train_dataset, val_dataset, test_dataset = random_split(
            dataset,
            [num_training, num_val, num_test],
            generator=torch.Generator().manual_seed(42),  # For reproducibility
        )

        # Create NeighborSampler for each dataset with multiple workers
        train_loader = NeighborSampler(
            train_dataset,
            sizes=sample_sizes,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        val_loader = NeighborSampler(
            val_dataset,
            sizes=sample_sizes,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        test_loader = NeighborSampler(
            test_dataset,
            sizes=sample_sizes,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

    return train_loader, val_loader, test_loader
