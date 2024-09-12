import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from utils.data_loader import load_dataset, split_data
from models.gps_layer import GPSNetwork
import argparse
import json
import os

def evaluate(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data)
        pred = out.argmax(dim=1)
        y_true = data.y[data.test_mask].cpu().numpy()
        y_pred = pred[data.test_mask].cpu().numpy()
        
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        
    return accuracy, precision, recall, f1

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load and preprocess data
    data, num_classes = load_dataset(args.dataset)
    data = split_data(data)
    data = data.to(device)
    
    # Initialize model
    model = GPSNetwork(data.num_features, num_classes).to(device)
    
    # Load trained model weights
    model.load_state_dict(torch.load(args.model_path))
    
    # Evaluate model
    accuracy, precision, recall, f1 = evaluate(model, data)
    
    # Print results
    print(f"Evaluation results on {args.dataset} dataset:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Save results to JSON file
    results = {
        "dataset": args.dataset,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }
    
    os.makedirs('results', exist_ok=True)
    with open(f'results/{args.dataset}_evaluation.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to results/{args.dataset}_evaluation.json")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate GPS Network on node classification datasets')
    parser.add_argument('--dataset', type=str, default='Cora', help='Dataset name')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model weights')
    args = parser.parse_args()
    
    main(args)
