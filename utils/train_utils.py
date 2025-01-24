"""
This module contains utility functions for training and evaluating models.
"""

import torch
import pandas as pd
import torch.nn.functional as F
import math


def train_node(model, train_loader, optimizer, device, args, mode="batch"):
    model.train()
    total_loss = 0

    if mode == "batch":
        for batched_data in train_loader:
            batched_data = batched_data.to(device)
            x, edge_index = batched_data.x, batched_data.edge_index
            optimizer.zero_grad()
            y_pred = model(x=x, edge_index=edge_index)

            if args.prediction_task == "regression":
                y_pred = y_pred.squeeze()
                y_true = batched_data.y.to(torch.float32)
                loss = F.mse_loss(y_pred, y_true)

            elif args.prediction_task == "classification-binary":
                y_true = batched_data.y.to(torch.float32).squeeze().long()
                y_true = y_true[:batched_data.batch_size]
                y_pred = y_pred.squeeze()  # Shape: (batch_size,)
                y_pred = y_pred[:batched_data.batch_size]
                loss = F.cross_entropy(y_pred, y_true.squeeze().long())

            else:  # "classification-multiclass"
                y_true = batched_data.y.to(torch.float32).squeeze().long()
                y_true = y_true[:batched_data.batch_size]
                y_pred = y_pred.squeeze()  # Shape: (batch_size,)
                y_pred = y_pred[:batched_data.batch_size]
                loss = F.cross_entropy(y_pred, y_true.squeeze().long())

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)
    else:

        data = train_loader.data.to(device)
        optimizer.zero_grad()
        positional_encoding = data.positional_encoding if hasattr(data, 'positional_encoding') else None
        structural_encoding = data.structural_encoding if hasattr(data, 'structural_encoding') else None

        output = model(x=data.x, edge_index=data.edge_index, positional_encoding=positional_encoding,
                       structural_encoding=structural_encoding)
        mask = data.train_mask
        if args.prediction_task == "regression":
            loss = F.mse_loss(output[mask], data.y[mask].float())
        elif args.prediction_task == "classification-binary":
            loss = F.binary_cross_entropy_with_logits(output[mask], data.y[mask].float())
        else:  # "classification-multiclass"
            loss = F.cross_entropy(output[mask], data.y[mask])
        loss.backward()
        optimizer.step()
        return loss.item()


@torch.no_grad()
def test_node(model, loader, device, args, mode="batch", split="val", return_predictions=False):
    model.eval()
    y_true = []
    y_pred = []

    if mode == "batch":
        for batched_data in loader:

            batched_data = batched_data.to(device)
            x, edge_index, batch = batched_data.x, batched_data.edge_index, batched_data.batch

            positional_encoding = batched_data.positional_encoding if hasattr(batched_data,
                                                                              'positional_encoding') else None
            structural_encoding = batched_data.structural_encoding if hasattr(batched_data,
                                                                              'structural_encoding') else None

            y_pred_batch = model(x=x, edge_index=edge_index, positional_encoding=positional_encoding,
                                 structural_encoding=structural_encoding, batch=batch)

            y_true_batch = batched_data.y.to(torch.float32).squeeze().long()
            y_true_batch = y_true_batch[:batched_data.batch_size]
            y_pred_batch = y_pred_batch.squeeze()
            y_pred_batch = y_pred_batch[:batched_data.batch_size]

            if args.prediction_task == "regression":
                y_true.append(y_true_batch.cpu())
                y_pred.append(y_pred_batch.cpu())
            else:
                pred = y_pred_batch.argmax(dim=1)
                y_true.append(y_true_batch.cpu())
                y_pred.append(pred.cpu())
    else:

        data = loader.data.to(device)

        positional_encoding = data.positional_encoding if hasattr(data,'positional_encoding') else None
        structural_encoding = data.structural_encoding if hasattr(data,'structural_encoding') else None

        output = model(x=data.x, edge_index=data.edge_index, positional_encoding=positional_encoding,
                       structural_encoding=structural_encoding)
        if split == "train":
            mask = data.train_mask
        elif split == "val":
            mask = data.val_mask
        else:
            mask = data.test_mask

        if args.prediction_task == "regression":
            y_true.append(data.y[mask].float().cpu())
            y_pred.append(output[mask].cpu())
        else:
            pred = output.argmax(dim=1)
            y_true.append(data.y[mask].cpu())
            y_pred.append(pred[mask].cpu())

    y_true = torch.cat(y_true)
    y_pred = torch.cat(y_pred)

    if args.prediction_task == "regression":
        mse = F.mse_loss(y_pred, y_true, reduction='mean').item()
        if return_predictions:
            df = pd.DataFrame({'pred': y_pred.numpy(), 'truth': y_true.numpy()})
            return mse, df
        else:
            return mse
    else:
        correct = y_pred.eq(y_true).sum().item()
        accuracy = correct / y_true.size(0)
        if return_predictions:
            df = pd.DataFrame({'pred': y_pred.numpy(), 'truth': y_true.numpy()})
            return accuracy, df
        else:
            return accuracy


def train_graph(model, train_loader, optimizer, device, args, output_dim, mode="batch", current_epoch=None):
    model.train()
    total_loss = 0
    
    dataset_size = len(train_loader.dataset)
    batch_size = train_loader.batch_size
    batch_size = train_loader.batch_size
    expected_batches = math.ceil(dataset_size/batch_size)
    actual_batches = len(train_loader)
    
    print(f"\nEpoch {current_epoch} Training Details:")
    print(f"- Total samples in dataset: {dataset_size:,}")
    print(f"- Batch size: {batch_size}")
    print(f"- Expected batches: {expected_batches:,}")
    print(f"- Actual batches reported: {actual_batches:,}")
    
    if mode == "batch":
        batch_count = 0
        for step, batched_data in enumerate(train_loader):
            batch_count += 1
            
            # Print progress every 100 batches
            if batch_count % 100 == 0:
                samples_processed = batch_count * batch_size
                percent_complete = (batch_count / expected_batches) * 100
                print(f"Progress: {batch_count:,}/{expected_batches:,} batches "
                      f"({percent_complete:.1f}%) - "
                      f"Samples: {samples_processed:,}/{dataset_size:,}", end='\r')
            
            batched_data = batched_data.to(device)
            x, edge_index, batch = batched_data.x, batched_data.edge_index, batched_data.batch

            # Skip invalid batches
            if batched_data.x.shape[0] == 1 or batched_data.batch[-1] == 0:
                continue

            optimizer.zero_grad()

            # Model forward pass
            y_pred = model(x=x, edge_index=edge_index, batch=batch)
            
            # Reshape y_pred if needed to match batched_data.y shape
            if y_pred.shape != batched_data.y.shape:
                y_pred = y_pred.view(batched_data.y.shape)
            
            # Mask for labeled data (exclude nan values)
            is_labeled = (batched_data.y == batched_data.y)  # Mask for valid entries (non-nan)
            
            # No need to flatten - work with original shapes
            y_pred = y_pred[is_labeled]  # Filter valid predictions
            y_true = batched_data.y[is_labeled].to(torch.float32)  # Filter valid labels

            # Compute loss dynamically based on task
            if args.prediction_task == "regression":
                loss = F.mse_loss(y_pred, y_true)

            elif args.prediction_task == "classification-binary":
                y_true = y_true.long()
                loss = F.cross_entropy(y_pred, y_true)

            elif args.prediction_task == "classification-multiclass":
                y_true = y_true.long()
                loss = F.cross_entropy(y_pred, y_true)

            elif args.prediction_task == "classification-multitask-binary":
                loss = F.binary_cross_entropy_with_logits(y_pred, y_true)

            else:
                raise ValueError(f"Unsupported prediction task: {args.prediction_task}")

            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        return total_loss / len(train_loader.dataset)

    else:
        raise ValueError(f"Unsupported mode: {mode}")


@torch.no_grad()
def test_graph(model, loader, device, args, mode='batch', return_predictions=False):
    model.eval()

    if mode == 'batch':
        y_true = []
        y_pred = []

        for step, batched_data in enumerate(loader):
            batched_data = batched_data.to(device)
            x, edge_index, batch = batched_data.x, batched_data.edge_index, batched_data.batch
            if batched_data.x.shape[0] == 1 or batched_data.batch[-1] == 0:
                pass
            else:
                is_labeled = batched_data.y == batched_data.y
                is_labeled = is_labeled.squeeze()
                y_pred_batch = model(x=x, edge_index=edge_index, batch=batch)

                y_pred_batch = y_pred_batch[is_labeled]
                y_true_batch = batched_data.y[is_labeled].to(torch.float32).squeeze().long()

                if args.prediction_task == 'regression':
                    y_true.append(y_true_batch.view(y_pred_batch.shape).detach().cpu())
                    y_pred.append(y_pred_batch.detach().cpu())
                else:
                    y_pred_batch = y_pred_batch.argmax(dim=1).cpu()
                    y_true.append(y_true_batch.view(y_pred_batch.shape).detach().cpu())
                    y_pred.append(y_pred_batch.detach().cpu())

        y_true = torch.cat(y_true, dim=0)
        y_pred = torch.cat(y_pred, dim=0)

        if args.prediction_task == "regression":
            mse = F.mse_loss(y_pred, y_true, reduction='mean').item()
            y_true = y_true.squeeze()
            y_pred = y_pred.squeeze()
            if return_predictions:
                df = pd.DataFrame({'pred': y_pred.numpy(), 'truth': y_true.numpy()})
                return mse, df
            else:
                return mse

        else:
            correct = y_pred.eq(y_true).sum().item()
            accuracy = correct / y_true.size(0)
            if return_predictions:
                df = pd.DataFrame({'pred': y_pred.numpy(), 'truth': y_true.numpy()})
                return accuracy, df
            else:
                return accuracy

    else:
        data = loader.dataset
        y_pred = torch.zeros(data.y.shape[0], data.y.shape[1])
        y_true = torch.zeros(data.y.shape[0], data.y.shape[1])
        for i in range(len(data)):
            G = data[i]
            positional_encoding = G.positional_encoding if hasattr(G, 'positional_encoding') else None
            structural_encoding = G.structural_encoding if hasattr(G, 'structural_encoding') else None
            output = model(G.x, G.edge_index, positional_encoding, structural_encoding)
            output = torch.mean(output, dim=0, keepdim=True)
            if args.prediction_task == "regression":
                y_pred[i] = output.cpu()
                y_true[i] = G.y
            else:
                y_pred[i] = output.argmax(dim=1).cpu()
                y_true[i] = G.y

        if args.prediction_task == "regression":
            mse = F.mse_loss(y_pred, y_true, reduction='mean').item()
            if return_predictions:
                df = pd.DataFrame({'pred': y_pred.numpy(), 'truth': y_true.numpy()})
                return mse, df
            else:
                return mse

        else:
            correct = y_pred.eq(y_true).sum().item()
            accuracy = correct / y_true.size(0)
            if return_predictions:
                df = pd.DataFrame({'pred': y_pred.numpy(), 'truth': y_true.numpy()})
                return accuracy, df
            else:
                return accuracy


def train_and_evaluate(model, train_loader, valid_loader, test_loader, optimizer, scheduler, args, device,
                       output_dim, mode="batch"):
    print(f"\nStarting training for {args.epochs} epochs")
    
    if args.prediction_task == "regression":
        best_val_metric = float('inf')
    else:
        best_val_metric = 0.0

    best_model_state = None

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 50)
        
        if args.prediction_level == "graph":
            loss = train_graph(model, train_loader, optimizer, device, args, output_dim, mode, current_epoch=epoch+1)
            train_metric = test_graph(model, train_loader, device, args)
            val_metric = test_graph(model, valid_loader, device, args)
        else:
            loss = train_node(model, train_loader, optimizer, device, args, mode)
            train_metric = test_node(model, train_loader, device, args, mode, split="train")
            val_metric = test_node(model, valid_loader, device, args, mode, split="val")

        scheduler.step()

        # Update the best model based on validation metric
        if args.prediction_task == "regression":
            is_best = val_metric < best_val_metric
        else:
            is_best = val_metric > best_val_metric

        if is_best:
            best_val_metric = val_metric
            best_model_state = model.state_dict()

        metric_name = "MSE" if args.prediction_task == "regression" else "Accuracy"
        print(
            f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Train {metric_name}: {train_metric:.4f}, Val {metric_name}: {val_metric:.4f}"
        )

        if epoch % 10 == 0 or epoch == args.epochs - 1:
            metric_name = "MSE" if args.prediction_task == "regression" else "Accuracy"
            print(
                f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Train {metric_name}: {train_metric:.4f}, Val {metric_name}: {val_metric:.4f}"
            )

    # Load the best model and evaluate on the test set
    model.load_state_dict(best_model_state)
    if args.prediction_level == "node":
        test_metric, res_df = test_node(model, test_loader, device, args, mode, split="test", return_predictions=True)
    else:
        test_metric, res_df = test_graph(model, test_loader, device, args, mode, return_predictions=True)

    metric_name = "MSE" if args.prediction_task == "regression" else "Accuracy"
    print(f"Test {metric_name}: {test_metric:.4f}")
    return test_metric, best_model_state, res_df
