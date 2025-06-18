import os
import gc
import torch
import json
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional
import shutil


def get_best_device() -> torch.device:
    """
    Get the best available device with informative feedback.
    
    Returns:
        The best available torch device
    """
    if torch.backends.mps.is_available():
        print("✅ MPS (Apple Silicon GPU) is available and will be used")
        return torch.device('mps')
    elif torch.cuda.is_available():
        print("✅ CUDA GPU is available and will be used")
        return torch.device('cuda')
    else:
        print("⚠️  No GPU acceleration available - using CPU")
        print("   💡 For better performance on Apple Silicon, run outside Docker")
        print("   💡 For CUDA support, configure Docker with GPU passthrough")
        return torch.device('cpu')


def setup_memory_optimization():
    """Setup memory optimization for different devices."""
    # Set MPS memory environment variable
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
    print("🔧 Set PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 (disables memory limit)")


def clean_memory():
    """Aggressive memory cleaning for MPS/CUDA."""
    # Multiple garbage collection passes
    for _ in range(3):
        gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    if torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
            torch.mps.synchronize()
        except AttributeError:
            # Older PyTorch versions might not have these methods
            pass


def get_memory_usage() -> str:
    """Get current memory usage string."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return f"CUDA: {allocated:.2f}GB / {reserved:.2f}GB"
    elif torch.backends.mps.is_available():
        # MPS doesn't have direct memory query methods
        return "MPS: Memory tracking not available"
    return "CPU: Memory tracking not available"


def safe_del(*tensors):
    """Safely delete tensors and clean memory."""
    for tensor in tensors:
        if tensor is not None:
            del tensor
    clean_memory()


def load_config(config_path: str = 'backend/config.json') -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def save_config(config: Dict[str, Any], config_path: str = 'backend/config.json'):
    """
    Save configuration to JSON file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save the configuration
    """
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)


def load_pretrained_embeddings(embeddings_path: str) -> np.ndarray:
    """
    Load pretrained embeddings from numpy file.
    
    Args:
        embeddings_path: Path to the embeddings file
        
    Returns:
        Pretrained embeddings array
    """
    embeddings = np.load(embeddings_path)
    print(f"Loaded pretrained embeddings: {embeddings.shape}")
    return embeddings


def save_model_artifacts(
    model, 
    optimizer, 
    config: Dict[str, Any], 
    epoch: int, 
    final_loss: float,
    datasets_stats: Dict[str, int],
    artifacts_dir: str = "artifacts",
    save_doc_embeddings: bool = False,
    datasets: Optional[Dict] = None,
    tokenizer = None
) -> str:
    """
    Save model artifacts with timestamped directory.
    
    Args:
        model: The trained model (TwoTowerModel)
        optimizer: The optimizer
        config: Training configuration
        epoch: Number of epochs trained
        final_loss: Final training loss
        datasets_stats: Dataset statistics
        artifacts_dir: Base directory for artifacts
        save_doc_embeddings: Whether to save document embeddings during training
        datasets: Dataset dictionaries (needed if save_doc_embeddings=True)
        tokenizer: Tokenizer instance (needed if save_doc_embeddings=True)
        
    Returns:
        Path to the saved artifacts directory
    """
    # Create artifacts directory
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # Create timestamped run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(artifacts_dir, f"two_tower_run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"💾 Saving artifacts to: {run_dir}")
    
    # Save model state dictionaries - handle different model types
    checkpoint_data = {
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'final_loss': final_loss,
        'config': config
    }
    
    # Handle different model architectures
    if hasattr(model, 'query_encoder') and hasattr(model, 'doc_encoder'):
        # Two-tower model
        checkpoint_data['query_encoder_state_dict'] = model.query_encoder.state_dict()
        checkpoint_data['doc_encoder_state_dict'] = model.doc_encoder.state_dict()
        checkpoint_data['model_type'] = 'two_tower'
    elif hasattr(model, 'shared_encoder'):
        # Merged/Ranking model
        checkpoint_data['shared_encoder_state_dict'] = model.shared_encoder.state_dict()
        if hasattr(model, 'classifier'):
            checkpoint_data['classifier_state_dict'] = model.classifier.state_dict()
        checkpoint_data['model_type'] = 'merged'
    else:
        # Fallback: save full model state dict
        checkpoint_data['model_state_dict'] = model.state_dict()
        checkpoint_data['model_type'] = 'unknown'
    
    torch.save(checkpoint_data, os.path.join(run_dir, 'model_checkpoint.pth'))
    
    # Save full model (for easy loading)
    torch.save(model, os.path.join(run_dir, 'full_model.pth'))
    
    # Save training configuration and results
    training_config = {
        'model_config': {
            'vocab_size': config.get('VOCAB_SIZE'),
            'embed_dim': config.get('EMBED_DIM'),
            'hidden_dim': config.get('HIDDEN_DIM', 128),
            'margin': config.get('MARGIN', 1.0),
            'rnn_type': config.get('RNN_TYPE', 'GRU'),
            'num_layers': config.get('NUM_LAYERS', 1),
            'dropout': config.get('DROPOUT', 0.0)
        },
        'training_config': {
            'epochs': epoch,
            'batch_size': config.get('BATCH_SIZE', 64),
            'learning_rate': config.get('LR', 0.001),
            'device': str(config.get('DEVICE', 'cpu')),
            'loss_type': config.get('LOSS_TYPE', 'triplet'),
            'task_mode': config.get('TASK_MODE', 'retrieval')
        },
        'data_config': {
            'subsample_ratio': config.get('SUBSAMPLE_RATIO'),
            'num_triplets_per_query': config.get('NUM_TRIPLETS_PER_QUERY', 1),
            **datasets_stats
        },
        'training_results': {
            'final_avg_loss': final_loss,
            'timestamp': timestamp
        }
    }
    
    with open(os.path.join(run_dir, 'training_config.json'), 'w') as f:
        json.dump(training_config, f, indent=2)
    
    # Copy tokenizer and embeddings if they exist
    if os.path.exists(config.get('WORD_TO_IDX_PATH', '')):
        shutil.copy2(config['WORD_TO_IDX_PATH'], os.path.join(run_dir, 'word_to_idx.pkl'))
    
    if os.path.exists(config.get('EMBEDDINGS_PATH', '')):
        shutil.copy2(config['EMBEDDINGS_PATH'], os.path.join(run_dir, 'embeddings.npy'))
    
    # Save document embeddings if requested
    if save_doc_embeddings and datasets and tokenizer:
        print("🔢 Generating and saving document embeddings...")
        _save_document_embeddings_during_training(model, datasets, tokenizer, run_dir, config)
    
    print(f"✅ Saved artifacts:")
    print(f"  📁 Directory: {run_dir}")
    print(f"  🧠 Models: model_checkpoint.pth, full_model.pth")
    print(f"  ⚙️  Config: training_config.json")
    print(f"  📝 Tokenizer: word_to_idx.pkl")
    print(f"  🔢 Embeddings: embeddings.npy")
    if save_doc_embeddings:
        print(f"  📄 Document Embeddings: document_embeddings.npy, doc_to_idx.pkl")
    
    return run_dir


def _save_document_embeddings_during_training(model, datasets, tokenizer, run_dir: str, config: Dict):
    """
    Helper function to save document embeddings during training.
    
    Args:
        model: Trained model
        datasets: Dictionary of datasets
        tokenizer: Tokenizer instance
        run_dir: Directory to save artifacts
        config: Configuration dictionary
    """
    import torch
    import numpy as np
    from torch.nn.utils.rnn import pad_sequence
    
    # Collect unique documents from all datasets
    all_documents = set()
    for split_name, dataset in datasets.items():
        if dataset:
            for query, pos_doc, neg_doc in dataset:
                all_documents.update([pos_doc, neg_doc])
    
    unique_docs = list(all_documents)
    print(f"  Found {len(unique_docs):,} unique documents across all datasets")
    
    # Create document-to-index mapping
    doc_to_idx = {doc: idx for idx, doc in enumerate(unique_docs)}
    
    # Encode documents in batches
    model.eval()
    device = next(model.parameters()).device
    batch_size = config.get('BATCH_SIZE', 64)
    doc_embeddings = []
    
    with torch.no_grad():
        for i in range(0, len(unique_docs), batch_size):
            if i % (batch_size * 10) == 0:
                print(f"    Encoding batch {i//batch_size + 1}/{(len(unique_docs) + batch_size - 1)//batch_size}")
            
            batch_docs = unique_docs[i:i + batch_size]
            
            # Tokenize batch
            batch_tokens = [
                torch.tensor(tokenizer.encode(doc), dtype=torch.long)
                for doc in batch_docs
            ]
            
            if batch_tokens:
                batch_tensor = pad_sequence(batch_tokens, batch_first=True).to(device)
                
                # Encode documents
                if hasattr(model, 'encode_document'):
                    embeddings = model.encode_document(batch_tensor)
                elif hasattr(model, 'encode_text'):
                    embeddings = model.encode_text(batch_tensor)
                else:
                    # Fallback for other model types
                    embeddings = model.doc_encoder(batch_tensor)
                
                doc_embeddings.append(embeddings.cpu().numpy())
    
    # Combine all embeddings
    if doc_embeddings:
        all_doc_embeddings = np.vstack(doc_embeddings)
        
        # Save document embeddings
        np.save(os.path.join(run_dir, 'document_embeddings.npy'), all_doc_embeddings)
        
        # Save document-to-index mapping
        import pickle
        with open(os.path.join(run_dir, 'doc_to_idx.pkl'), 'wb') as f:
            pickle.dump(doc_to_idx, f)
        
        # Save document texts
        with open(os.path.join(run_dir, 'documents.txt'), 'w', encoding='utf-8') as f:
            for doc in unique_docs:
                f.write(doc + '\n')
        
        print(f"  ✅ Saved {len(unique_docs):,} document embeddings ({all_doc_embeddings.shape})")
    else:
        print("  ⚠️  No document embeddings to save")


def load_model_artifacts(artifacts_dir: str, device: Optional[torch.device] = None):
    """
    Load model artifacts from a saved directory.
    
    Args:
        artifacts_dir: Path to the artifacts directory
        device: Device to load the model on
        
    Returns:
        Tuple of (model, config, training_info)
    """
    if device is None:
        device = get_best_device()
    
    # Load full model
    model_path = os.path.join(artifacts_dir, 'full_model.pth')
    model = torch.load(model_path, map_location=device)
    
    # Load training config
    config_path = os.path.join(artifacts_dir, 'training_config.json')
    with open(config_path, 'r') as f:
        training_info = json.load(f)
    
    # Load checkpoint for additional info
    checkpoint_path = os.path.join(artifacts_dir, 'model_checkpoint.pth')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    print(f"✅ Loaded model from {artifacts_dir}")
    print(f"   Final loss: {checkpoint['final_loss']:.4f}")
    print(f"   Epochs: {checkpoint['epoch']}")
    
    return model, checkpoint['config'], training_info


def print_model_summary(model, config: Dict[str, Any]):
    """Print a summary of the model architecture."""
    print("\n" + "="*50)
    print("📊 MODEL SUMMARY")
    print("="*50)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"🏗️  Architecture: Two-Tower RNN")
    print(f"📝 Vocabulary Size: {config.get('VOCAB_SIZE', 'N/A'):,}")
    print(f"📐 Embedding Dimension: {config.get('EMBED_DIM', 'N/A')}")
    print(f"🧠 Hidden Dimension: {config.get('HIDDEN_DIM', 'N/A')}")
    print(f"🔄 RNN Type: {config.get('RNN_TYPE', 'GRU')}")
    print(f"📚 RNN Layers: {config.get('NUM_LAYERS', 1)}")
    print(f"🚫 Dropout: {config.get('DROPOUT', 0.0)}")
    print(f"⚖️  Shared Encoders: {config.get('SHARED_ENCODER', False)}")
    print(f"📊 Total Parameters: {total_params:,}")
    print(f"🎯 Trainable Parameters: {trainable_params:,}")
    print("="*50)


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and fill in missing configuration values.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Validated configuration dictionary
    """
    # Required fields
    required_fields = [
        'TRAIN_DATASET_PATH', 'VAL_DATASET_PATH', 'TEST_DATASET_PATH',
        'EMBEDDINGS_PATH', 'WORD_TO_IDX_PATH'
    ]
    
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required configuration field: {field}")
    
    # Default values
    defaults = {
        'NUM_TRIPLETS_PER_QUERY': 1,
        'HIDDEN_DIM': 128,
        'MARGIN': 1.0,
        'LR': 0.001,
        'BATCH_SIZE': 64,
        'EPOCHS': 10,
        'RNN_TYPE': 'GRU',
        'NUM_LAYERS': 1,
        'DROPOUT': 0.0,
        'SHARED_ENCODER': False,
        'LOSS_TYPE': 'triplet',
        'GRADIENT_ACCUMULATION_STEPS': 1,
        'MEMORY_CLEANUP_FREQUENCY': 100
    }
    
    for key, value in defaults.items():
        if key not in config:
            config[key] = value
    
    return config 