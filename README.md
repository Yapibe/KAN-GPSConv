# KAN-GPS: Integrating Kolmogorov-Arnold Networks with Graph Positional Signatures

## Project Overview

This project explores the integration of Kolmogorov-Arnold Network (KAN) layers into Graph Positional Signatures (GPS) networks for various graph-based tasks. Our goal is to enhance the expressiveness and interpretability of graph neural networks while maintaining or improving their performance on benchmark datasets.

## Project Goals

1. Design and implement two KAN-GPS architectures: Hybrid-KAN-GPS and KAN-GPS.
2. Evaluate the performance of these architectures on benchmark datasets from the Open Graph Benchmark (OGB).
3. Compare the expressiveness, interpretability, and performance of our models with standard GPS networks and other state-of-the-art graph neural networks.
4. Analyze the continual learning capabilities of KAN-GPS models, focusing on knowledge retention.
5. Investigate the interpretability aspects of KAN-GPS by visualizing learned univariate functions and analyzing model components.

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

1. **Hybrid-KAN-GPS**: An architecture that replaces the final MLP in the standard GPS layer with a KAN layer.
2. **KAN-GPS**: An architecture that replaces both the MPNN and Global Attention mechanisms with KAN-based counterparts, in addition to using a KAN layer for final processing.
3. **Continual Learning Evaluation**: Focus on knowledge retention when models are trained sequentially on multiple tasks or datasets.
4. **Interpretability Analysis**: Visualization of learned univariate spline functions and exploration of the relationship between MPNN and Global Attention layers.

## Methodology

### Hybrid-KAN-GPS Layer

The Hybrid-KAN-GPS layer is defined as:

$X^{(l+1)}, E^{(l+1)} = \text{Hybrid-KAN-GPS}^{(l)}(X^{(l)}, E^{(l)}, A)$

computed as:

$$
\begin{align*}
X_M^{(l+1)}, E^{(l+1)} &= \text{MPNN}^{(l)}(X^{(l)}, E^{(l)}, A) \\
X_T^{(l+1)} &= \text{GlobalAttn}^{(l)}(X^{(l)}) \\
X_M^{(l+1)} &= \text{BatchNorm}(\text{Dropout}(X_M^{(l+1)}) + X^{(l)}) \\
X_T^{(l+1)} &= \text{BatchNorm}(\text{Dropout}(X_T^{(l+1)}) + X^{(l)}) \\
X^{(l+1)} &= \text{KAN}^{(l)}(X_M^{(l+1)} + X_T^{(l+1)})
\end{align*}
$$

### KAN-GPS Layer

The KAN-GPS layer is defined as:

$X^{(l+1)}, E^{(l+1)} = \text{KAN-GPS}^{(l)}(X^{(l)}, E^{(l)}, A)$

computed as:

$$
\begin{align*}
X_M^{(l+1)}, E^{(l+1)} &= \text{KAN-MPNN}^{(l)}(X^{(l)}, E^{(l)}, A) \\
X_T^{(l+1)} &= \text{KAN-GlobalAttn}^{(l)}(X^{(l)}) \\
X_M^{(l+1)} &= \text{BatchNorm}(\text{Dropout}(X_M^{(l+1)}) + X^{(l)}) \\
X_T^{(l+1)} &= \text{BatchNorm}(\text{Dropout}(X_T^{(l+1)}) + X^{(l)}) \\
X^{(l+1)} &= \text{KAN}^{(l)}(X_M^{(l+1)} + X_T^{(l+1)})
\end{align*}
$$

## Evaluation

We will evaluate our models on benchmark datasets from the Open Graph Benchmark (OGB) across different tasks, including:
- Node classification
- Link prediction
- Graph classification

For continual learning evaluation, we will simulate sequential learning scenarios and monitor performance on both new and previously learned tasks.

## Expected Outcomes

1. Improved performance on OGB benchmark tasks compared to standard GPS and baseline models.
2. Demonstration of practical utility and benefits in continual learning and interpretability, even if not achieving the best performance in all tasks.
3. Insights into the impact of KAN layers on model expressiveness and interpretability in graph neural networks.

## Setup and Installation

1. Clone the repository:
   ```
   git clone https://github.com/Yapibe/KAN-GPSConv.git
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

We welcome contributions to the KAN-GPS project! Please read our CONTRIBUTING.md file for guidelines on how to submit issues, feature requests, and pull requests.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.

## Acknowledgments

- This project builds upon the work on Graph Positional Signatures and Kolmogorov-Arnold Networks.
- We thank the PyTorch Geometric team for their excellent framework for graph neural networks.
- We acknowledge the Open Graph Benchmark (OGB) for providing standardized datasets and evaluation protocols.

## Contact

For any questions or feedback, please open an issue in this repository or contact the project maintainers directly.
