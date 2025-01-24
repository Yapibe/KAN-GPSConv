#!/bin/bash
#SBATCH --job-name=kan_gps_training
#SBATCH --output=logs/kan_gps_training_%j.out
#SBATCH --error=logs/kan_gps_training_%j.err
#SBATCH --partition=gpu-ai
#SBATCH --account=gpu-research
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=50G


# Create logs directory if it doesn't exist
mkdir -p logs

# Activate Conda environment
source /home/yandex/MLWG2024/yairp1/anaconda3/etc/profile.d/conda.sh

conda activate kan-gps

export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH

# Navigate to your project directory
cd /home/yandex/MLWG2024/yairp1/KAN-GPSConv

export WANDB_API_KEY="c48c8a1278e99c4d9c66ecfe5d7d4151b7570ad8"

# Run your Python script

# python experiments/train_model.py \
#     --model_name_id ZINC_KANGPS_1 \
#     --prediction_level graph \
#     --prediction_task regression \
#     --data_source torch_geometric_datasets \
#     --dataset_name ZINC \
#     --batch_size 32 \
#     --num_workers 4 \
#     --epochs 150 \
#     --lr 0.001 \
#     --optimizer Adam \
#     --hidden_channels_dim 64 \
#     --num_layers 5 \
#     --model_type kangps \
#     --model_mpnn_type GIN \
#     --dropout 0.05 \
#     --attn_num_heads 1 \
#     --attention_dropout 0.1 \
#     --grid_size 3 \
#     --spline_order 3 \
#     --kan_hidden_dim 64 \
#     --kan_num_of_hidden_layers 3 \
#     --grid_size_attention 3 \
#     --spline_order_attention 3 \
#     --kan_hidden_dim_attention 64 \
#     --kan_num_of_hidden_layers_attention 3 \
#     --grid_size_mpnn 3 \
#     --spline_order_mpnn 3 \
#     --kan_hidden_dim_mpnn 64 \
#     --kan_num_of_hidden_layers_mpnn 3 \
#     --se_dim 3 \
#     --concat_se True \
#     --structural_encoding_type RWSE \
#     --pe_dim 4 \
#     --concat_pe True \
#     --pos_encoding_type laplacian_eigenvectors

# python experiments/train_model.py \
#     --model_name_id ZINC_KANGPS_Hybrid_1 \
#     --prediction_level graph \
#     --prediction_task regression \
#     --data_source torch_geometric_datasets \
#     --dataset_name ZINC \
#     --batch_size 32 \
#     --num_workers 4 \
#     --epochs 150 \
#     --lr 0.001 \
#     --optimizer Adam \
#     --hidden_channels_dim 64 \
#     --num_layers 5 \
#     --model_type hybrid \
#     --model_mpnn_type GIN \
#     --dropout 0.05 \
#     --attn_num_heads 2 \
#     --attention_dropout 0.1 \
#     --grid_size 3 \
#     --spline_order 3 \
#     --kan_hidden_dim 64 \
#     --kan_num_of_hidden_layers 3 \
#     --grid_size_attention 3 \
#     --spline_order_attention 3 \
#     --attn_mlp_hidden_dim 64 \
#     --attn_mlp_num_of_hidden_layers 3 \
#     --grid_size_mpnn 3 \
#     --spline_order_mpnn 3 \
#     --mpnn_mlp_hidden_dim 64 \
#     --mpnn_mlp_num_of_hidden_layers 3 \
#     --se_dim 3 \
#     --concat_se True \
#     --structural_encoding_type RWSE \
#     --pe_dim 4 \
#     --concat_pe True \
#     --pos_encoding_type laplacian_eigenvectors


# python experiments/train_model.py \
#     --model_name_id ZINC_KANGPS_2 \
#     --prediction_level graph \
#     --prediction_task regression \
#     --data_source torch_geometric_datasets \
#     --dataset_name ZINC \
#     --batch_size 32 \
#     --num_workers 4 \
#     --epochs 150 \
#     --lr 0.001 \
#     --optimizer Adam \
#     --hidden_channels_dim 64 \
#     --num_layers 5 \
#     --model_type kangps \
#     --model_mpnn_type GIN \
#     --dropout 0.05 \
#     --attn_num_heads 1 \
#     --attention_dropout 0.1 \
#     --grid_size 5 \
#     --spline_order 5 \
#     --kan_hidden_dim 64 \
#     --kan_num_of_hidden_layers 3 \
#     --grid_size_attention 5 \
#     --spline_order_attention 5 \
#     --kan_hidden_dim_attention 64 \
#     --kan_num_of_hidden_layers_attention 3 \
#     --grid_size_mpnn 5 \
#     --spline_order_mpnn 5 \
#     --kan_hidden_dim_mpnn 64 \
#     --kan_num_of_hidden_layers_mpnn 3 \
#     --se_dim 3 \
#     --concat_se True \
#     --structural_encoding_type RWSE \
#     --pe_dim 4 \
#     --concat_pe True \
#     --pos_encoding_type laplacian_eigenvectors

# python experiments/train_model.py \
#     --model_name_id ZINC_KANGPS_Hybrid_2 \
#     --prediction_level graph \
#     --prediction_task regression \
#     --data_source torch_geometric_datasets \
#     --dataset_name ZINC \
#     --batch_size 32 \
#     --num_workers 4 \
#     --epochs 150 \
#     --lr 0.001 \
#     --optimizer Adam \
#     --hidden_channels_dim 64 \
#     --num_layers 5 \
#     --model_type hybrid \
#     --model_mpnn_type GIN \
#     --dropout 0.05 \
#     --attn_num_heads 2 \
#     --attention_dropout 0.1 \
#     --grid_size 5 \
#     --spline_order 5 \
#     --kan_hidden_dim 64 \
#     --kan_num_of_hidden_layers 3 \
#     --grid_size_attention 5 \
#     --spline_order_attention 5 \
#     --attn_mlp_hidden_dim 64 \
#     --attn_mlp_num_of_hidden_layers 3 \
#     --grid_size_mpnn 5 \
#     --spline_order_mpnn 5 \
#     --mpnn_mlp_hidden_dim 64 \
#     --mpnn_mlp_num_of_hidden_layers 3 \
#     --se_dim 3 \
#     --concat_se True \
#     --structural_encoding_type RWSE \
#     --pe_dim 4 \
#     --concat_pe True \
#     --pos_encoding_type laplacian_eigenvectors

# python experiments/train_model.py \
#     --model_name_id ZINC_KANGPS_3 \
#     --prediction_level graph \
#     --prediction_task regression \
#     --data_source torch_geometric_datasets \
#     --dataset_name ZINC \
#     --batch_size 32 \
#     --num_workers 4 \
#     --epochs 100 \
#     --lr 0.001 \
#     --optimizer Adam \
#     --hidden_channels_dim 64 \
#     --num_layers 5 \
#     --model_type kangps \
#     --model_mpnn_type GIN \
#     --dropout 0.05 \
#     --attn_num_heads 1 \
#     --attention_dropout 0.1 \
#     --grid_size 3 \
#     --spline_order 1 \
#     --kan_hidden_dim 64 \
#     --kan_num_of_hidden_layers 3 \
#     --grid_size_attention 3 \
#     --spline_order_attention 1 \
#     --kan_hidden_dim_attention 64 \
#     --kan_num_of_hidden_layers_attention 3 \
#     --grid_size_mpnn 3 \
#     --spline_order_mpnn 1 \
#     --kan_hidden_dim_mpnn 64 \
#     --kan_num_of_hidden_layers_mpnn 3 \
#     --se_dim 3 \
#     --concat_se True \
#     --structural_encoding_type RWSE \
#     --pe_dim 4 \
#     --concat_pe True \
#     --pos_encoding_type laplacian_eigenvectors


# python experiments/train_model.py \
#     --model_name_id ZINC_KANGPS_Hybrid_3 \
#     --prediction_level graph \
#     --prediction_task regression \
#     --data_source torch_geometric_datasets \
#     --dataset_name ZINC \
#     --batch_size 32 \
#     --num_workers 4 \
#     --epochs 100 \
#     --lr 0.001 \
#     --optimizer Adam \
#     --hidden_channels_dim 64 \
#     --num_layers 5 \
#     --model_type hybrid \
#     --model_mpnn_type GIN \
#     --dropout 0.05 \
#     --attn_num_heads 2 \
#     --attention_dropout 0.1 \
#     --grid_size 3 \
#     --spline_order 1 \
#     --kan_hidden_dim 64 \
#     --kan_num_of_hidden_layers 3 \
#     --grid_size_attention 3 \
#     --spline_order_attention 1 \
#     --attn_mlp_hidden_dim 64 \
#     --attn_mlp_num_of_hidden_layers 3 \
#     --grid_size_mpnn 3 \
#     --spline_order_mpnn 1 \
#     --mpnn_mlp_hidden_dim 64 \
#     --mpnn_mlp_num_of_hidden_layers 3 \
#     --se_dim 3 \
#     --concat_se True \
#     --structural_encoding_type RWSE \
#     --pe_dim 4 \
#     --concat_pe True \
#     --pos_encoding_type laplacian_eigenvectors


python experiments/train_model.py \
    --model_name_id ZINC_KANGPS_4 \
    --prediction_level graph \
    --prediction_task regression \
    --data_source torch_geometric_datasets \
    --dataset_name ZINC \
    --batch_size 32 \
    --num_workers 4 \
    --epochs 100 \
    --lr 0.001 \
    --optimizer Adam \
    --hidden_channels_dim 64 \
    --num_layers 5 \
    --model_type kangps \
    --model_mpnn_type GIN \
    --dropout 0.05 \
    --attn_num_heads 1 \
    --attention_dropout 0.1 \
    --grid_size 5 \
    --spline_order 10 \
    --kan_hidden_dim 64 \
    --kan_num_of_hidden_layers 3 \
    --grid_size_attention 5 \
    --spline_order_attention 10 \
    --kan_hidden_dim_attention 64 \
    --kan_num_of_hidden_layers_attention 3 \
    --grid_size_mpnn 5 \
    --spline_order_mpnn 10 \
    --kan_hidden_dim_mpnn 64 \
    --kan_num_of_hidden_layers_mpnn 3 \
    --se_dim 3 \
    --concat_se True \
    --structural_encoding_type RWSE \
    --pe_dim 4 \
    --concat_pe True \
    --pos_encoding_type laplacian_eigenvectors


# python experiments/train_model.py \
#     --model_name_id ZINC_KANGPS_Hybrid_4 \
#     --prediction_level graph \
#     --prediction_task regression \
#     --data_source torch_geometric_datasets \
#     --dataset_name ZINC \
#     --batch_size 32 \
#     --num_workers 4 \
#     --epochs 150 \
#     --lr 0.001 \
#     --optimizer Adam \
#     --hidden_channels_dim 64 \
#     --num_layers 5 \
#     --model_type hybrid \
#     --model_mpnn_type GIN \
#     --dropout 0.05 \
#     --attn_num_heads 2 \
#     --attention_dropout 0.1 \
#     --grid_size 5 \
#     --spline_order 10 \
#     --kan_hidden_dim 64 \
#     --kan_num_of_hidden_layers 3 \
#     --grid_size_attention 5 \
#     --spline_order_attention 10 \
#     --attn_mlp_hidden_dim 64 \
#     --attn_mlp_num_of_hidden_layers 3 \
#     --grid_size_mpnn 5 \
#     --spline_order_mpnn 10 \
#     --mpnn_mlp_hidden_dim 64 \
#     --mpnn_mlp_num_of_hidden_layers 3 \
#     --se_dim 3 \
#     --concat_se True \
#     --structural_encoding_type RWSE \
#     --pe_dim 4 \
#     --concat_pe True \
#     --pos_encoding_type laplacian_eigenvectors

