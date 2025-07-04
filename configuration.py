import torch
import os

# Project configuration for Nonstationary Financial Prediction
# Using RNN and AEFIN (Attention-Enhanced Fourier-Integrated Network)

# File paths
dir_path = os.path.dirname(os.path.realpath(__file__))
data_path = dir_path + r"/data/financial_data.csv"
output_path = dir_path + r"/output/"

# Data configuration
date_column = 'Date'
target_column = 'Close'  # Can be price or returns
target_name = 'Financial Asset'

# Time series configuration for nonstationary data
sequence_length = 60        # 60 time steps (e.g., 60 days)
prediction_horizon = 1      # Predict 1 step ahead
overlap_ratio = 0.8         # 80% overlap between sequences

# Nonstationarity handling
use_differencing = True     # Convert to returns/differences
use_log_transform = False   # Log transformation for price data
detrend_data = True        # Remove linear trends
rolling_window = 30        # Rolling statistics window

# Model configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# RNN Configuration
rnn_config = {
    'hidden_size': 128,
    'num_layers': 3,
    'dropout': 0.2,
    'bidirectional': True,
    'rnn_type': 'LSTM'  # 'LSTM', 'GRU', or 'RNN'
}

# AEFIN Configuration  
aefin_config = {
    'hidden_size': 256,
    'num_attention_heads': 8,
    'fourier_modes': 32,        # Number of Fourier modes to keep
    'attention_dropout': 0.1,
    'fourier_dropout': 0.1,
    'integration_layers': 2,
    'use_positional_encoding': True
}

# Training configuration
batch_size = 64
learning_rate = 1e-4
num_epochs = 200
early_stopping_patience = 20
weight_decay = 1e-5

# Data splitting
train_split = 0.7
val_split = 0.15
test_split = 0.15

# Evaluation metrics
evaluation_metrics = ['mse', 'mae', 'mape', 'directional_accuracy', 'sharpe_ratio']

# Plotting configuration
fig_size = (15, 10)
dpi_display = 100
dpi_save = 300
font_size = 12
font_size_title = 16

# Random seed for reproducibility
random_seed = 42

# Advanced settings for nonstationary handling
adaptive_normalization = True  # Adapt normalization over time
regime_detection = True        # Detect regime changes
volatility_clustering = True   # Handle volatility clustering
