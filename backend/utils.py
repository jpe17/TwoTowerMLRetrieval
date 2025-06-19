import os
import gc
import torch
import json
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional
import shutil
from torch.serialization import safe_globals
import argparse
import pickle


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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Two-Tower Model Training')
    parser.add_argument(
        '--checkpoint', '-c',
        type=str,
        help='Path to checkpoint directory to resume training from'
    )
    return parser.parse_args()


def load_embeddings_and_maps(artifacts_path, device):
    """Loads all necessary embeddings and mappings from an artifacts directory."""
    print(f"📂 Loading pre-computed embeddings from {artifacts_path}...")
    
    try:
        # Load documents
        doc_embeddings = torch.from_numpy(np.load(f"{artifacts_path}/document_embeddings.npy")).to(device)
        with open(f"{artifacts_path}/doc_to_idx.pkl", 'rb') as f:
            doc_to_idx = pickle.load(f)
        
        # Load queries
        query_embeddings = torch.from_numpy(np.load(f"{artifacts_path}/query_embeddings.npy")).to(device)
        with open(f"{artifacts_path}/query_to_idx.pkl", 'rb') as f:
            query_to_idx = pickle.load(f)
            
        # Create reverse mapping to get document text from its index
        idx_to_doc = {idx: doc for doc, idx in doc_to_idx.items()}

        print(f"  ✅ Loaded {len(doc_to_idx):,} document embeddings.")
        print(f"  ✅ Loaded {len(query_to_idx):,} query embeddings.")
        
        return doc_embeddings, doc_to_idx, idx_to_doc, query_embeddings, query_to_idx
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}.")
        print(f"   Ensure the path '{artifacts_path}' is correct and contains the required .npy and .pkl files.")
        print("   💡 Did you run training with the 'save_doc_embeddings=True' flag?")
        return None, None, None, None, None


def create_model_and_trainer(config, pretrained_embeddings, device, checkpoint_path=None):
    """Create two-tower model and trainer."""
    from model import ModelFactory
    from trainer import TwoTowerTrainer
    
    print("\n🏗️  Creating two-tower model...")
    
    if checkpoint_path:
        print(f"📥 Loading model from checkpoint: {checkpoint_path}")
        # Load model artifacts (includes model state, optimizer state, config)
        loaded_model, loaded_config, training_info = load_model_artifacts(checkpoint_path)
        
        # Use the loaded model directly
        model = loaded_model
        model = model.to(device)
        
        # Create trainer with loaded config
        trainer = TwoTowerTrainer(model, loaded_config, device)
        
        # Load checkpoint to get optimizer state
        checkpoint_path_file = os.path.join(checkpoint_path, 'model_checkpoint.pth')
        checkpoint = torch.load(checkpoint_path_file, map_location=device)
        if 'optimizer_state_dict' in checkpoint:
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        # Update config with loaded config
        config.update(loaded_config)
        
        print("✅ Model restored successfully from checkpoint")
    else:
        model = ModelFactory.create_two_tower_model(config, pretrained_embeddings)
        model = model.to(device)
        trainer = TwoTowerTrainer(model, config, device)
    
    return model, trainer


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
        print("🔢 Generating and saving document and query embeddings...")
        _save_embeddings_during_training(model, datasets, tokenizer, run_dir, config)
    
    print(f"✅ Saved artifacts:")
    print(f"  📁 Directory: {run_dir}")
    print(f"  🧠 Models: model_checkpoint.pth, full_model.pth")
    print(f"  ⚙️  Config: training_config.json")
    print(f"  📝 Tokenizer: word_to_idx.pkl")
    print(f"  🔢 Embeddings: embeddings.npy")
    if save_doc_embeddings:
        print(f"  📄 Document Embeddings: document_embeddings.npy, doc_to_idx.pkl")
        print(f"  ❓ Query Embeddings: query_embeddings.npy, query_to_idx.pkl")
    
    return run_dir


def _save_embeddings_during_training(model, datasets, tokenizer, run_dir: str, config: Dict):
    """
    Helper function to save document and query embeddings during training.
    
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
    import pickle
    
    # Collect unique documents and queries from all datasets
    all_documents = set()
    all_queries = set()
    for split_name, dataset in datasets.items():
        if dataset:
            for query, pos_doc, neg_doc in dataset:
                all_documents.update([pos_doc, neg_doc])
                all_queries.add(query)
    
    unique_docs = list(all_documents)
    unique_queries = list(all_queries)
    print(f"  Found {len(unique_docs):,} unique documents across all datasets")
    print(f"  Found {len(unique_queries):,} unique queries across all datasets")
    
    # CRITICAL FIX: Save and restore model training state
    was_training = model.training
    model.eval()  # Ensure model is in eval mode
    device = next(model.parameters()).device
    batch_size = min(32, config.get('BATCH_SIZE', 64))  # Use smaller batch size for stability

    # --- Create and Save Document Embeddings ---
    if unique_docs:
        print("  Processing document embeddings...")
        doc_to_idx = {doc: idx for idx, doc in enumerate(unique_docs)}
        doc_embeddings = []
        with torch.no_grad():
            for i in range(0, len(unique_docs), batch_size):
                if i % (batch_size * 10) == 0:
                    print(f"    Encoding doc batch {i//batch_size + 1}/{(len(unique_docs) + batch_size - 1)//batch_size}")
                batch_docs = unique_docs[i:i + batch_size]
                batch_tokens = [torch.tensor(tokenizer.encode(doc), dtype=torch.long) for doc in batch_docs]
                if batch_tokens:
                    # CRITICAL FIX: Ensure tensor is properly constructed
                    batch_tensor = pad_sequence(batch_tokens, batch_first=True).to(device)
                    
                    # CRITICAL FIX: Double-check model has the right method
                    if hasattr(model, 'encode_document'):
                        embeddings = model.encode_document(batch_tensor)
                    elif hasattr(model, 'doc_encoder'):
                        embeddings = model.doc_encoder(batch_tensor)
                    else: # Fallback
                        raise ValueError("Model does not have encode_document or doc_encoder method")
                    
                    # CRITICAL FIX: Ensure embeddings are properly detached
                    embeddings_np = embeddings.detach().cpu().numpy()
                    doc_embeddings.append(embeddings_np)
                    
                    # CRITICAL FIX: Add debug info for first batch
                    if i == 0 and len(embeddings_np) > 1:
                        sim_check = np.dot(embeddings_np[0], embeddings_np[1]) / (
                            np.linalg.norm(embeddings_np[0]) * np.linalg.norm(embeddings_np[1])
                        )
                        print(f"    🔍 First batch similarity check: {sim_check:.6f} (should be < 0.99)")
                        if sim_check > 0.99:
                            print(f"    ⚠️  WARNING: Identical embeddings detected in first batch!")
        
        if doc_embeddings:
            all_doc_embeddings = np.vstack(doc_embeddings)
            np.save(os.path.join(run_dir, 'document_embeddings.npy'), all_doc_embeddings)
            with open(os.path.join(run_dir, 'doc_to_idx.pkl'), 'wb') as f:
                pickle.dump(doc_to_idx, f)
            with open(os.path.join(run_dir, 'documents.txt'), 'w', encoding='utf-8') as f:
                for doc in unique_docs:
                    f.write(doc + '\n')
            print(f"    ✅ Saved {len(unique_docs):,} document embeddings ({all_doc_embeddings.shape})")

    # --- Create and Save Query Embeddings ---
    if unique_queries:
        print("  Processing query embeddings...")
        query_to_idx = {query: idx for idx, query in enumerate(unique_queries)}
        query_embeddings = []
        with torch.no_grad():
            for i in range(0, len(unique_queries), batch_size):
                if i % (batch_size * 10) == 0:
                    print(f"    Encoding query batch {i//batch_size + 1}/{(len(unique_queries) + batch_size - 1)//batch_size}")
                batch_queries = unique_queries[i:i + batch_size]
                batch_tokens = [torch.tensor(tokenizer.encode(q), dtype=torch.long) for q in batch_queries]
                if batch_tokens:
                    batch_tensor = pad_sequence(batch_tokens, batch_first=True).to(device)
                    if hasattr(model, 'encode_query'):
                        embeddings = model.encode_query(batch_tensor)
                    elif hasattr(model, 'query_encoder'):
                        embeddings = model.query_encoder(batch_tensor)
                    else: # Fallback
                        embeddings = model.encode_text(batch_tensor)
                    query_embeddings.append(embeddings.cpu().numpy())

        if query_embeddings:
            all_query_embeddings = np.vstack(query_embeddings)
            np.save(os.path.join(run_dir, 'query_embeddings.npy'), all_query_embeddings)
            with open(os.path.join(run_dir, 'query_to_idx.pkl'), 'wb') as f:
                pickle.dump(query_to_idx, f)
            with open(os.path.join(run_dir, 'queries.txt'), 'w', encoding='utf-8') as f:
                for q in unique_queries:
                    f.write(q + '\n')
            print(f"    ✅ Saved {len(unique_queries):,} query embeddings ({all_query_embeddings.shape})")
    
    # CRITICAL FIX: Restore model training state
    if was_training:
        model.train()
    
    # CRITICAL FIX: Validate saved embeddings to catch corruption early
    if os.path.exists(os.path.join(run_dir, 'document_embeddings.npy')):
        saved_doc_emb = np.load(os.path.join(run_dir, 'document_embeddings.npy'))
        if len(saved_doc_emb) > 1:
            # Check if first two embeddings are identical (sign of corruption)
            sim = np.dot(saved_doc_emb[0], saved_doc_emb[1]) / (
                np.linalg.norm(saved_doc_emb[0]) * np.linalg.norm(saved_doc_emb[1])
            )
            if sim > 0.999:
                print(f"    ⚠️  WARNING: Document embeddings may be corrupted (similarity: {sim:.6f})")
                print(f"    💡 Consider regenerating embeddings if retrieval performance is poor")
            else:
                print(f"    ✅ Document embeddings look diverse (similarity: {sim:.6f})")
        
        # Check variance
        total_variance = np.var(saved_doc_emb)
        if total_variance < 1e-6:
            print(f"    ❌ CRITICAL: Document embeddings have no variance ({total_variance:.10f})")
            print(f"    💡 This will cause zero retrieval performance - embeddings need regeneration")
        else:
            print(f"    ✅ Document embeddings have good variance ({total_variance:.6f})")


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
    
    # Import all model classes to make them available for loading
    from model import TwoTowerModel, RNNEncoder
    
    # Add all necessary classes to safe globals (PyTorch classes + our custom classes)
    with safe_globals([
        TwoTowerModel, 
        RNNEncoder,
        torch.nn.modules.sparse.Embedding,
        torch.nn.modules.rnn.GRU,
        torch.nn.modules.rnn.LSTM,
        torch.nn.modules.rnn.RNN,
        torch.nn.modules.activation.ReLU,
        torch.nn.modules.dropout.Dropout,
        torch.nn.modules.normalization.LayerNorm
    ]):
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
    
    # Ensure backward compatibility for loaded models
    if hasattr(model, '_ensure_backward_compatibility'):
        model._ensure_backward_compatibility()
    
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
        'LOSS_TYPE': 'triplet',
        'GRADIENT_ACCUMULATION_STEPS': 1,
        'MEMORY_CLEANUP_FREQUENCY': 100
    }
    
    for key, value in defaults.items():
        if key not in config:
            config[key] = value
    
    return config 