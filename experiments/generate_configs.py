import itertools

# Define the datasets and their fixed parameters
datasets = [
    {
        'data_source': 'OGB',
        'dataset_name': 'ogbg-molhiv',
        'prediction_level': 'graph',
        'prediction_task': 'classification-binary',
    },
    {
        'data_source': 'OGB',
        'dataset_name': 'ogbg-molpcba',
        'prediction_level': 'graph',
        'prediction_task': 'classification-multiclass',
    },
    {
        'data_source': 'OGB',
        'dataset_name': 'ogbn-arxiv',
        'prediction_level': 'node',
        'prediction_task': 'classification-multiclass',
    },
    {
        'data_source': 'torch_geometric_datasets',
        'dataset_name': 'ZINC',
        'prediction_level': 'graph',
        'prediction_task': 'regression',
    },
]

# Define the options to vary
pos_encoding_types = ['laplacian_eigenvectors', 'RandomWalk']
structural_encoding_types = ['RWSE', 'laplacian_eigenvalues']
optimizers = ['SGD', 'Adam']
model_types = ['kangps']  # Only 'kangps' is provided
model_mpnn_types = ['GCN', 'GIN']
attn_num_heads_options = [1, 2, 3]

# Generate all possible combinations of the options
combinations = list(itertools.product(
    pos_encoding_types,
    structural_encoding_types,
    optimizers,
    model_types,
    model_mpnn_types,
    attn_num_heads_options
))


# Function to generate a unique model_name_id
def generate_model_name_id(config):
    return f"model_{config['dataset_name']}_{config['prediction_level']}_{config['prediction_task']}_pos_{config['pos_encoding_type']}_se_{config['structural_encoding_type']}_opt_{config['optimizer']}_mpnn_{config['model_mpnn_type']}_heads_{config['attn_num_heads']}"


# Open the file in write mode
with open('configs.txt', 'w') as f:
    # Generate and write command-line arguments for each dataset and combination
    for dataset in datasets:
        for combination in combinations:
            config = {
                'data_source': dataset['data_source'],
                'dataset_name': dataset['dataset_name'],
                'prediction_level': dataset['prediction_level'],
                'prediction_task': dataset['prediction_task'],
                'pos_encoding_type': combination[0],
                'structural_encoding_type': combination[1],
                'optimizer': combination[2],
                'model_type': combination[3],
                'model_mpnn_type': combination[4],
                'attn_num_heads': combination[5],
                'model_name_id': '',  # Will be set below
            }
            config['model_name_id'] = generate_model_name_id(config)

            # Construct the command-line arguments
            cmd_args = [
                f"--dataset_name {config['dataset_name']}",
                f"--prediction_level {config['prediction_level']}",
                f"--prediction_task {config['prediction_task']}",
                f"--data_source {config['data_source']}",
                f"--pos_encoding_type {config['pos_encoding_type']}",
                f"--structural_encoding_type {config['structural_encoding_type']}",
                f"--optimizer {config['optimizer']}",
                f"--model_type {config['model_type']}",
                f"--model_mpnn_type {config['model_mpnn_type']}",
                f"--attn_num_heads {config['attn_num_heads']}",
                f"--model_name_id {config['model_name_id']}",
            ]

            # Join the arguments into a command string
            cmd = "python train_model.py " + ' '.join(cmd_args)

            # Write the command to the file
            f.write(cmd + '\n')

# OGB - ogbg-molhiv(graph,binary classif,AUROC),ogbg-molpcba(graph, 128-task classif,Avg. Precision),ogbn-products(node, Multi-class classification,Accuracy)
# ZINC(graph,regression,MAE)