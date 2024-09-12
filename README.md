# KAN-GPSConv: Integrating Kolmogorov-Arnold Networks with Graph Positional Signatures

## Project Overview

This project explores the integration of Kolmogorov-Arnold Network (KAN) layers into Graph Positional Signatures (GPS) networks for both node and graph classification tasks. Our goal is to enhance the expressiveness and interpretability of graph neural networks while maintaining or improving their performance on various benchmark datasets.

## Project Goals

1. Implement a flexible GPS network architecture that incorporates KAN layers.
2. Support both node classification and graph classification tasks.
3. Evaluate the performance of our KAN-GPS model on various benchmark datasets.
4. Compare the expressiveness, interpretability, and performance of our model with standard GPS networks and other state-of-the-art graph neural networks.
5. Provide a comprehensive analysis of the results and insights gained from integrating KAN layers into GPS networks.

## Directory Structure
```
KAN-GPSConv/
├── data/                   # Stores datasets (automatically downloaded)
├── external_libs/          # External libraries
│   ├── efficient-kan       # Efficient Kolmogorov-Arnold Network implementation
│   ├── fast-kan            # Fast Kolmogorov-Arnold Network implementation
│   ├── faster-kan          # Faster Kolmogorov-Arnold Network implementation
│   ├── FCN-KAN             # FCN-KAN implementation
│   ├── GraphGPS            # Graph Positional Signatures implementation
│   ├── GraphKAN            # Graph Kolmogorov-Arnold Network implementation
│   ├── LKAN                # LKAN implementation
│   └── pykan               # PyKAN implementation
├── models/                 # Model architectures
│   ├── gps_layer.py        # Implementation of the GPS layer
│   ├── kan_layer.py        # Implementation of the KAN layer
│   └── kan_gps_model.py    # Implementation of the KAN-GPS model
├── experiments/            # Experiment scripts and notebooks
│   ├── train_model.py      # Main training script
│   └── evaluate_model.py   # Evaluation script for trained models
├── results/                # Stores evaluation metrics and logs
├── utils/                  # Helper functions and utilities
│   ├── data_loader.py      # Dataset loading and preprocessing functions
│   └── train_utils.py      # Training and evaluation utilities
├── literature_review/      # Relevant papers and research summaries
├── README.md               # This file
├── requirements.txt        # Python dependencies
└── setup.py                # Package installation script
```
## Key Components

1. **GPS Network**: Our main model architecture, implemented in `models/gps_network.py`. It supports both node and graph classification tasks and incorporates KAN layers.

2. **KAN Layer**: Implemented in `models/kan_layer.py`, this layer adds the expressiveness of Kolmogorov-Arnold representation to our graph neural network.

3. **Data Loader**: Located in `utils/data_loader.py`, it supports loading various node and graph classification datasets, including Planetoid datasets (Cora, Citeseer, PubMed), WebKB datasets, Actor dataset, OGB datasets, and TUDatasets for graph classification.

4. **Training Script**: `experiments/train_model.py` handles the training process for both node and graph classification tasks.

5. **Evaluation Script**: `experiments/evaluate_model.py` is used to evaluate trained models on test sets and generate performance metrics.

## Recent Insights and Changes

1. **KAN Implementation**: We've learned that it's more effective to implement KAN for graphs in the latent feature space. This is achieved by using a linear layer to project input features into a latent space before applying the KAN layer.

2. **Optimizer Choice**: Recent experiments suggest that using SGD (with momentum) leads to more stable training compared to Adam, albeit with slower convergence. We've updated our training script to support both optimizers for comparison.

3. **Extended Training**: Due to the slower convergence of SGD, we've increased the default number of training epochs.

4. **Weights & Biases Integration**: We use Weights & Biases (wandb) for experiment tracking and visualization.

## Setup and Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/KAN-GPSConv.git
   cd KAN-GPSConv
   ```

2. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```

3. Install the project in editable mode:
   ```
   pip install -e .
   ```

## Running Experiments

### Training

To train the model on a node classification task:

```
python experiments/train_model.py --dataset Cora --task node --optimizer sgd --epochs 1000
```

To train the model on a graph classification task:

```
python experiments/train_model.py --dataset MUTAG --task graph --optimizer sgd --epochs 1000
```

Command-line arguments:
- `--dataset`: Name of the dataset (e.g., 'Cora', 'MUTAG')
- `--task`: 'node' for node classification, 'graph' for graph classification
- `--optimizer`: 'sgd' or 'adam'
- `--epochs`: Number of training epochs

## Evaluation

To evaluate a trained model:
```
python experiments/evaluate_model.py --dataset Cora --task node --model_path path/to/your/trained_model.pth
```

## Supported Datasets

### Node Classification
- Cora
- Citeseer
- PubMed
- Cornell
- Texas
- Wisconsin
- Actor
- ogbn-arxiv

### Graph Classification
- Any dataset available in TUDataset (e.g., MUTAG, PROTEINS, NCI1)

## Implementation Notes

1. The current KAN layer implementation is a simplified version and may need adjustments based on specific KAN architectures being targeted.

2. The GPS Network supports both node and graph classification tasks. The architecture automatically adjusts based on the specified task.

3. For graph classification tasks, we use global mean pooling to aggregate node features. This can be modified or extended with other pooling methods.

4. Hyperparameters (like hidden dimensions) are currently set as default values and may need tuning for optimal performance on different datasets.

5. The training script currently uses Adam optimizer with default learning rates. Learning rate scheduling or different optimizers can be implemented for potentially better performance.

6. For large datasets, especially in graph classification tasks, consider implementing mini-batch training and data parallelism for improved efficiency.

## Key Findings

1. The initial linear projection before KAN layers is crucial for effective training.
2. SGD optimizer shows more stable training but requires more epochs compared to Adam.
3. The integration of KAN layers into GPS networks shows promise in enhancing model expressiveness.

## Future Work

1. Implement and experiment with different variants of KAN layers.
2. Explore the interpretability aspects of the KAN-GPS integration.
3. Extend the model to support edge prediction and graph generation tasks.
4. Implement visualization tools for analyzing the learned representations.
5. Conduct ablation studies to quantify the impact of KAN layers on model performance.

## Contributing

We welcome contributions to the KAN-GPSConv project! Please read our CONTRIBUTING.md file for guidelines on how to submit issues, feature requests, and pull requests.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.

## Acknowledgments

- This project builds upon the work on Graph Positional Signatures and Kolmogorov-Arnold Networks.
- We thank the PyTorch Geometric team for their excellent framework for graph neural networks.

## Contact

For any questions or feedback, please open an issue in this repository or contact the project maintainers directly.
