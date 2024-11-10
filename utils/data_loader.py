import torch
from torch_geometric.datasets import Planetoid, WebKB, Actor, TUDataset
from torch_geometric.datasets import ZINC
import math
from ogb.nodeproppred import PygNodePropPredDataset
from ogb.graphproppred import PygGraphPropPredDataset
from torch.utils.data import random_split
from torch_geometric.utils import get_laplacian,degree
from torch_geometric.loader import DataLoader
from torch_geometric.loader import NeighborLoader



class AddSinusoidalPE(object):
    def __init__(self, pe_dim, attr_name='positional_encoding'):
        self.pe_dim = pe_dim
        self.attr_name = attr_name

    def __call__(self, data):
        num_nodes = data.num_nodes
        position = torch.arange(0, num_nodes, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.pe_dim, 2).float() * (-math.log(10000.0) / self.pe_dim)
        )
        pe = torch.zeros(num_nodes, self.pe_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        setattr(data, self.attr_name, pe)
        return data
class AddLaplacianEigenvaluesSE(object):
    def __init__(self, se_dim, attr_name='structural_encoding'):
        self.se_dim = se_dim
        self.attr_name = attr_name

    def __call__(self, data):
        L_edge_index, L_edge_weight = get_laplacian(edge_index=data.edge_index, normalization="sym", num_nodes=data.num_nodes)
        L = torch.sparse.FloatTensor(L_edge_index, L_edge_weight, torch.Size([data.num_nodes, data.num_nodes])).to_dense()
        eigenvalues = torch.linalg.eigvalsh(L)
        eigenvalues = eigenvalues[:self.se_dim].unsqueeze(1)  # Shape: (se_dim, 1)
        setattr(data, self.attr_name, eigenvalues)
        return data
class AddNodeDegreeSE(object):
    def __init__(self, attr_name='structural_encoding'):
        self.attr_name = attr_name

    def __call__(self, data):
        deg = degree(data.edge_index[0], num_nodes=data.num_nodes)
        deg = deg.unsqueeze(1)  # Shape: (num_nodes, 1)
        setattr(data, self.attr_name, deg)
        return data
class ComputeRWSEPower(object):
    def __init__(self, attr_name='RWSE', exponent=8, se_attr_name='structural_encoding'):
        self.attr_name = attr_name
        self.exponent = exponent
        self.se_attr_name = se_attr_name

    def __call__(self, data):
        RWSE = getattr(data, self.attr_name, None)
        if RWSE is not None:
            RWSE_pow = torch.pow(RWSE, self.exponent)
            RWSE_diag = torch.diag(RWSE_pow).unsqueeze(0).expand(data.num_nodes, -1)
            setattr(data, self.se_attr_name, RWSE_diag)
            setattr(data, self.attr_name, None)
        return data
class ConcatAttributesToX(object):
    def __init__(self, attr_names):
        self.attr_names = attr_names

    def __call__(self, data):
        attrs = []
        for attr_name in self.attr_names:
            attr = getattr(data, attr_name, None)
            if attr is not None:
                attrs.append(attr.float())
                setattr(data, attr_name, None)  # Remove the attribute after concatenation
        if attrs:
            if data.x is not None:
                data.x = torch.cat([data.x.float()] + attrs, dim=-1)
            else:
                data.x = torch.cat(attrs, dim=-1)
        return data

class ConvertNodeFeaturesToFloat(object):
    def __call__(self, data):
        if data.x is not None:
            data.x = data.x.float()
        return data

def get_transform(pos_encoding_type, pe_dim,
                  structural_encoding_type, se_dim, concat_pe, concat_se):
    import torch_geometric.transforms as T

    # Start with base transformations that do not modify edge_index
    base_transformation = [T.RemoveIsolatedNodes(), T.AddSelfLoops()]
    transforms = base_transformation.copy()

    pe_attr_name = 'positional_encoding'
    se_attr_name = 'structural_encoding'
    concat_attrs = []

    # Positional Encoding
    if pos_encoding_type == 'sinusoidal':
        transforms.append(AddSinusoidalPE(pe_dim=pe_dim, attr_name=pe_attr_name))
        if concat_pe:
            concat_attrs.append(pe_attr_name)
    elif pos_encoding_type == 'laplacian_eigenvectors':
        transforms.append(T.AddLaplacianEigenvectorPE(k=pe_dim, attr_name=pe_attr_name))
        if concat_pe:
            concat_attrs.append(pe_attr_name)
    elif pos_encoding_type == 'RandomWalk':
        transforms.append(T.AddRandomWalkPE(walk_length=pe_dim, attr_name=pe_attr_name))
        if concat_pe:
            concat_attrs.append(pe_attr_name)
    elif pos_encoding_type is None:
        pass
    else:
        raise ValueError(f"Unsupported positional encoding type: {pos_encoding_type}")

    # Structural Encoding
    if structural_encoding_type == 'RWSE':
        transforms.append(T.AddRandomWalkPE(walk_length=se_dim, attr_name='RWSE'))
        transforms.append(ComputeRWSEPower(attr_name='RWSE', exponent=8, se_attr_name=se_attr_name))
        if concat_se:
            concat_attrs.append(se_attr_name)
    elif structural_encoding_type == 'laplacian_eigenvalues':
        transforms.append(AddLaplacianEigenvaluesSE(se_dim=se_dim, attr_name=se_attr_name))
        if concat_se:
            concat_attrs.append(se_attr_name)
    elif structural_encoding_type == 'Node degree':
        transforms.append(AddNodeDegreeSE(attr_name=se_attr_name))
        if concat_se:
            concat_attrs.append(se_attr_name)
    elif structural_encoding_type is None:
        pass
    else:
        raise ValueError(f"Unsupported structural encoding type: {structural_encoding_type}")

    # Now, convert to sparse tensor (after positional and structural encodings)
    transforms.append(ConvertNodeFeaturesToFloat())
    # transforms.append(T.ToSparseTensor())

    # Normalize features if needed
    transforms.append(T.NormalizeFeatures())

    # Concatenate attributes to data.x if needed
    if concat_attrs:
        transforms.append(ConcatAttributesToX(attr_names=concat_attrs))

    transform = T.Compose(transforms)
    return transform




def load_dataset(root, data_source, dataset_name,
                 batch_size,
                 prediction_task, prediction_level,
                 concat_pe, concat_se,
                 pe_dim, pos_encoding_type,
                 se_dim, structural_encoding_type, device,
                 num_neighbors_sample_size=None
                 ):
    transform = get_transform(pos_encoding_type=pos_encoding_type, pe_dim=pe_dim,
                              structural_encoding_type=structural_encoding_type, se_dim=se_dim,
                              concat_pe=concat_pe, concat_se=concat_se)
    if prediction_level == "node":

        if data_source == "Planetoid":
            dataset = Planetoid(root=root, name=dataset_name, transform=transform)
            data = dataset[0]
            train_loader, valid_loader, test_loader = preprocess_graph_data(data=data, batch_size=batch_size,
                                                                            device=device,
                                                                            sample_sizes=num_neighbors_sample_size)
            return dataset, train_loader, valid_loader, test_loader, dataset.num_classes
        elif data_source == "WebKB":
            dataset = WebKB(root=root, name=dataset_name, transform=transform)
            data = dataset[0]
            train_loader, valid_loader, test_loader = preprocess_graph_data(data=data, batch_size=batch_size,
                                                                            device=device,
                                                                            sample_sizes=num_neighbors_sample_size)
            return dataset, train_loader, valid_loader, test_loader, dataset.num_classes
        elif data_source == "Actor":
            dataset = Actor(root=root, transform=transform)
            data = dataset[0]
            train_loader, valid_loader, test_loader = preprocess_graph_data(data=data, batch_size=batch_size,
                                                                            device=device,
                                                                            sample_sizes=num_neighbors_sample_size)
            return dataset, train_loader, valid_loader, test_loader, dataset.num_classes
        elif data_source == "TUDataset":
            dataset = TUDataset(root=root, name=dataset_name, transform=transform)
            data = dataset[0]
            train_loader, valid_loader, test_loader = preprocess_graph_data(data=data, batch_size=batch_size,
                                                                            device=device,
                                                                            sample_sizes=num_neighbors_sample_size)
            return dataset, train_loader, valid_loader, test_loader, dataset.num_classes
        elif data_source == "OGB":
            dataset = PygNodePropPredDataset(name=dataset_name, root=root, transform=transform)
            data = dataset[0]
            data = data.to(device)

            split_idx = dataset.get_idx_split()
            train_idx, valid_idx, test_idx = split_idx["train"].to(device), split_idx["valid"].to(device), split_idx[
                "test"].to(device)

            num_nodes = data.num_nodes
            train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            train_mask[train_idx] = True

            val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            val_mask[valid_idx] = True

            test_mask = torch.zeros(num_nodes, dtype=torch.bool)
            test_mask[test_idx] = True

            data.train_mask = train_mask
            data.val_mask = val_mask
            data.test_mask = test_mask

            train_loader = NeighborLoader(
                data=data,
                input_nodes=data.train_mask,
                num_neighbors=num_neighbors_sample_size,
                batch_size=batch_size,
                shuffle=True, num_workers=0)

            valid_loader = NeighborLoader(
                data=data,
                input_nodes=data.val_mask,
                num_neighbors=[-1],
                batch_size=batch_size,
                shuffle=False, num_workers=0)

            test_loader = NeighborLoader(
                data=data,
                input_nodes=data.test_mask,
                num_neighbors=[-1],
                batch_size=batch_size,
                shuffle=False, num_workers=0)

            return dataset, train_loader, valid_loader, test_loader, dataset.num_classes
        else:
            raise ValueError(f"Unsupported data_source: {data_source}")
    else:
        if data_source == "TUDataset":
            dataset = TUDataset(root=root, name=dataset_name, transform=transform)
            dataset = dataset.to(device)
            total_size = len(dataset)
            num_training = int(total_size * 0.8)
            num_val = int(total_size * 0.1)
            num_test = total_size - num_training - num_val
            train_dataset, val_dataset, test_dataset = random_split(
                dataset,
                [num_training, num_val, num_test],
                generator=torch.Generator().manual_seed(42),
            )
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            valid_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            num_classes = 1 if prediction_task == "regression" else dataset.num_tasks
            return dataset, train_loader, valid_loader, test_loader, num_classes
        elif (data_source == "torch_geometric_datasets") & (dataset_name == 'ZINC'):

            dataset = ZINC(root=root, pre_transform=transform)
            train_dataset = ZINC(root=root, split='train', pre_transform=transform)
            val_dataset = ZINC(root=root, split='val', pre_transform=transform)
            test_dataset = ZINC(root=root, split='test', pre_transform=transform)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            valid_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            num_classes = 1

            return dataset, train_loader, valid_loader, test_loader, num_classes
        elif "OGB" in data_source:
            dataset = PygGraphPropPredDataset(name=dataset_name, root=root, transform=transform)
            dataset = dataset.to(device)
            split_idx = dataset.get_idx_split()
            train_loader = DataLoader(dataset[split_idx["train"]], batch_size=batch_size, shuffle=True)
            valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(dataset[split_idx["test"]], batch_size=batch_size, shuffle=False)

            num_classes = 1 if prediction_task == "regression" else dataset.num_classes
            return dataset, train_loader, valid_loader, test_loader, num_classes
        else:
            raise ValueError(f"Unsupported data_source: {data_source}")


def split_node_task_data(data, val_ratio=0.2, test_ratio=0.2):
    num_nodes = data.num_nodes
    indices = torch.randperm(num_nodes)
    test_size = int(num_nodes * test_ratio)
    val_size = int(num_nodes * val_ratio)

    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask[indices[:test_size]] = True

    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask[indices[test_size: test_size + val_size]] = True

    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.train_mask[indices[test_size + val_size:]] = True

    return data


def preprocess_graph_data(data, batch_size, device, num_workers=4, sample_sizes=None):
    sample_sizes = [5, 5] if sample_sizes is None else sample_sizes

    data = split_node_task_data(data=data)
    data = data.to(device)

    train_loader = NeighborLoader(
        data=data,
        input_nodes=data.train_mask,
        num_neighbors=sample_sizes,
        batch_size=batch_size,
        shuffle=True, num_workers=num_workers)

    valid_loader = NeighborLoader(
        data=data,
        input_nodes=data.val_mask,
        num_neighbors=sample_sizes,
        batch_size=batch_size,
        shuffle=False, num_workers=num_workers)

    test_loader = NeighborLoader(
        data=data,
        input_nodes=data.test_mask,
        num_neighbors=sample_sizes,
        batch_size=batch_size,
        shuffle=False, num_workers=num_workers)

    return train_loader, valid_loader, test_loader
