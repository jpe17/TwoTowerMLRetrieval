# Two-Tower ML Retrieval System

A modular, production-ready implementation of a two-tower neural retrieval model for information retrieval tasks. This system converts the original Jupyter notebook into well-structured Python modules with clear separation of concerns.

## 🏗️ Architecture

The system is built with a modular architecture:

```
backend/
├── data_loader.py      # Data loading and preprocessing
├── tokenizer.py        # Text tokenization
├── dataset.py          # PyTorch datasets and data loaders
├── model.py            # Neural network models and loss functions
├── trainer.py          # Training, validation, and testing logic
├── evaluator.py        # Evaluation metrics and search functionality
├── utils.py            # Utility functions and helpers
├── main.py             # Main training script
└── __init__.py         # Package initialization
```

## ✨ Features

- **Modular Design**: Clean separation of concerns with focused modules
- **Subsampling Support**: Train on a subset of your data for quick experiments
- **Multiple Loss Functions**: Triplet loss and cosine similarity loss
- **Flexible Architecture**: Support for different RNN types (GRU, LSTM, RNN)
- **Memory Optimization**: Built-in memory management for large datasets
- **Comprehensive Evaluation**: Precision@K, MRR, and interactive search demos
- **Device Flexibility**: Automatic detection and support for CPU, CUDA, and MPS (Apple Silicon)
- **Artifact Management**: Automatic saving and loading of trained models

## 🚀 Quick Start

### 1. Basic Training

Train with default configuration:
```bash
python backend/main.py
```

### 2. Quick Experiment with Subsampling

Train on 10% of the data for 3 epochs:
```bash
python backend/main.py --subsample 0.1 --epochs 3
```

### 3. Training with Interactive Demo

```bash
python backend/main.py --subsample 0.05 --epochs 2 --demo
```

### 4. Custom Batch Size

```bash
python backend/main.py --batch-size 32 --epochs 5
```

## 📖 Usage Examples

### Programmatic Usage

```python
from backend import (
    DataLoader, PretrainedTokenizer, DataLoaderFactory,
    ModelFactory, TrainerFactory, TwoTowerEvaluator,
    load_config, get_best_device
)

# Load configuration
config = load_config('backend/config.json')
config['EPOCHS'] = 5
config['BATCH_SIZE'] = 32

# Setup device
device = get_best_device()

# Load data with subsampling
data_loader = DataLoader(config)
datasets = data_loader.load_datasets(subsample_ratio=0.1)

# Create model and train
tokenizer = PretrainedTokenizer(config['WORD_TO_IDX_PATH'])
model = ModelFactory.create_two_tower_model(config, pretrained_embeddings)
trainer = TrainerFactory.create_trainer(config, model, device)

# Train the model
history = trainer.train(train_loader, val_loader)
```

### Custom Configuration

```python
custom_config = {
    'TRAIN_DATASET_PATH': 'data/ms_marco_train.parquet',
    'VAL_DATASET_PATH': 'data/ms_marco_validation.parquet',
    'TEST_DATASET_PATH': 'data/ms_marco_test.parquet',
    'EMBEDDINGS_PATH': 'data/embeddings.npy',
    'WORD_TO_IDX_PATH': 'data/word_to_idx.pkl',
    
    # Model settings
    'HIDDEN_DIM': 128,
    'RNN_TYPE': 'LSTM',
    'NUM_LAYERS': 2,
    'DROPOUT': 0.1,
    'LOSS_TYPE': 'cosine',
    'EPOCHS': 10,
    'BATCH_SIZE': 64,
    'LR': 0.001,
}

# Use programmatic training interface
from backend.main import train_with_custom_config
model, trainer, evaluator, artifacts_path = train_with_custom_config(
    config_dict=custom_config,
    subsample_ratio=0.2
)
```

### Search and Evaluation

```python
# Create evaluator
evaluator = TwoTowerEvaluator(trained_model, tokenizer, device)

# Search documents
documents = ["Document 1 text...", "Document 2 text...", ...]
query = "What is machine learning?"
results = evaluator.search(query, documents, top_k=5)

# Evaluate retrieval performance
metrics = evaluator.evaluate_retrieval(test_data, num_samples=100)
print(f"Precision@1: {metrics['precision_at_1']:.3f}")
print(f"MRR: {metrics['mrr']:.3f}")
```

## ⚙️ Configuration

Edit `backend/config.json` to customize training:

```json
{
    "TRAIN_DATASET_PATH": "data/ms_marco_train.parquet",
    "VAL_DATASET_PATH": "data/ms_marco_validation.parquet", 
    "TEST_DATASET_PATH": "data/ms_marco_test.parquet",
    "EMBEDDINGS_PATH": "data/embeddings.npy",
    "WORD_TO_IDX_PATH": "data/word_to_idx.pkl",
    
    "NUM_TRIPLETS_PER_QUERY": 1,
    "HIDDEN_DIM": 128,
    "RNN_TYPE": "GRU",
    "NUM_LAYERS": 1,
    "DROPOUT": 0.0,
    "MARGIN": 1.0,
    "LR": 0.001,
    "BATCH_SIZE": 64,
    "EPOCHS": 10,
    "LOSS_TYPE": "triplet"
}
```

### Key Parameters

- **SUBSAMPLE_RATIO**: Fraction of data to use (0.1 = 10%)
- **HIDDEN_DIM**: Size of RNN hidden state
- **RNN_TYPE**: Type of RNN ("GRU", "LSTM", "RNN")
- **NUM_LAYERS**: Number of RNN layers
- **LOSS_TYPE**: Loss function ("triplet", "cosine")
- **MARGIN**: Margin for triplet loss
- **DROPOUT**: Dropout rate for regularization

## 📊 Supported Datasets

The system works with MS MARCO dataset format:
- **Training**: ~800K query-passage pairs
- **Validation**: ~100K query-passage pairs  
- **Test**: ~100K query-passage pairs

Data should be in Parquet format with columns:
- `query`: Search query text
- `passages.passage_text`: List of relevant passages

## 🔧 Advanced Features

### Memory Optimization

Built-in memory management for large datasets:
- Gradient accumulation for effective larger batch sizes
- Periodic memory cleanup
- MPS (Apple Silicon) optimization
- Automatic fallback handling

### Model Variants

Configure different model architectures:

```python
config = {
    'RNN_TYPE': 'LSTM',        # GRU, LSTM, or RNN
    'NUM_LAYERS': 2,           # Multi-layer RNNs
    'DROPOUT': 0.1,            # Regularization
    'SHARED_ENCODER': False,   # Separate query/doc encoders
    'LOSS_TYPE': 'cosine'      # Triplet or cosine loss
}
```

### Evaluation Metrics

Comprehensive evaluation with:
- **Precision@K**: Precision at top-K results
- **Mean Reciprocal Rank (MRR)**: Average reciprocal rank of first relevant result
- **Interactive Search**: Demo queries with real-time results
- **Similarity Analysis**: Pairwise text similarity

## 🎯 Examples

Run the example script to see different usage patterns:

```bash
python example_usage.py
```

This demonstrates:
1. Quick training with subsampled data
2. Custom configuration training
3. Search and evaluation functionality

## 📁 Project Structure

```
TwoTowerMLRetrieval/
├── backend/              # Main package
│   ├── __init__.py
│   ├── data_loader.py    # Data loading
│   ├── tokenizer.py      # Tokenization
│   ├── dataset.py        # PyTorch datasets
│   ├── model.py          # Neural models
│   ├── trainer.py        # Training logic
│   ├── evaluator.py      # Evaluation
│   ├── utils.py          # Utilities
│   ├── main.py           # Main script
│   └── config.json       # Configuration
├── data/                 # Data files
│   ├── ms_marco_*.parquet
│   ├── embeddings.npy
│   └── word_to_idx.pkl
├── artifacts/            # Saved models
├── notebooks/            # Original notebook
├── example_usage.py      # Usage examples
└── README.md             # This file
```

## 🚀 Getting Started

1. **Install Dependencies**:
   ```bash
   pip install torch pandas fastparquet numpy
   ```

2. **Prepare Data**: Ensure your data files are in the `data/` directory

3. **Quick Test**:
   ```bash
   python backend/main.py --subsample 0.01 --epochs 1
   ```

4. **Full Training**:
   ```bash
   python backend/main.py --subsample 0.1 --epochs 5
   ```

5. **Custom Training**:
   ```bash
   python example_usage.py
   ```

## 🎛️ Command Line Options

```bash
python backend/main.py [options]

Options:
  --config PATH          Configuration file path (default: backend/config.json)
  --subsample RATIO      Subsample ratio 0.0 < ratio <= 1.0
  --epochs N             Number of training epochs
  --batch-size N         Batch size for training
  --eval-only            Only run evaluation (requires trained model)
  --demo                 Run interactive search demo
```

## 🏆 Key Improvements from Notebook

1. **Modularity**: Clear separation of concerns
2. **Reusability**: Easy to modify and extend
3. **Configurability**: JSON-based configuration
4. **Subsampling**: Train on data subsets
5. **Production Ready**: Error handling and logging
6. **Memory Efficient**: Optimized for large datasets
7. **Cross-Platform**: Works on CPU, CUDA, and MPS
8. **Comprehensive Testing**: Built-in evaluation suite

## 📈 Performance Tips

1. **Use Subsampling**: Start with `--subsample 0.1` for quick experiments
2. **Adjust Batch Size**: Use `--batch-size 32` for memory-constrained systems
3. **GPU Acceleration**: System automatically detects and uses available GPUs
4. **Memory Management**: Built-in cleanup handles large datasets efficiently

The modular design makes it easy to experiment with different configurations, data sizes, and model architectures while maintaining clean, maintainable code. 