#!/usr/bin/env python3
"""
Two-Tower ML Training Script

Focused on triplet-based learning using two-tower architecture.
"""

import sys
import torch
from pathlib import Path
from dotenv import load_dotenv
import os
import wandb
import argparse
import numpy as np
import pickle
import torch.nn.functional as F

load_dotenv()  # Loads .env file

# Only login, don't init here (trainer will handle init)
wandb.login(key=os.getenv("WANDB_API_KEY"))

# Add backend to path for imports
sys.path.append(str(Path(__file__).parent))

from data_loader import DataLoader
from tokenizer import PretrainedTokenizer
from dataset import DataLoaderFactory
from model import ModelFactory, TwoTowerModel
from trainer import TwoTowerTrainer
from evaluate import run_evaluation
from utils import (
    load_config, validate_config, get_best_device, setup_memory_optimization,
    load_pretrained_embeddings, print_model_summary, load_model_artifacts,
    save_model_artifacts
)


def create_model_and_trainer(config, pretrained_embeddings, device, checkpoint_path=None):
    """Create two-tower model and trainer."""
    print("\n🏗️  Creating two-tower model...")
    
    if checkpoint_path:
        print(f"📥 Loading model from checkpoint: {checkpoint_path}")
        # Load model artifacts (includes model state, optimizer state, config)
        loaded_artifacts = load_model_artifacts(checkpoint_path)
        
        # Create model with loaded config
        model = ModelFactory.create_two_tower_model(loaded_artifacts['config'], pretrained_embeddings)
        model = model.to(device)
        
        # Load model state
        model.load_state_dict(loaded_artifacts['model_state'])
        
        # Create trainer and restore optimizer state
        trainer = TwoTowerTrainer(model, loaded_artifacts['config'], device)
        if 'optimizer_state' in loaded_artifacts:
            trainer.optimizer.load_state_dict(loaded_artifacts['optimizer_state'])
            
        # Update config with loaded config
        config.update(loaded_artifacts['config'])
        
        print("✅ Model restored successfully from checkpoint")
    else:
        model = ModelFactory.create_two_tower_model(config, pretrained_embeddings)
        model = model.to(device)
        trainer = TwoTowerTrainer(model, config, device)
    
    return model, trainer


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


def main():
    """Main training function."""
    # Parse command line arguments
    args = parse_args()
    
    # Load and validate configuration
    print("📋 Loading configuration...")
    config = load_config('backend/config.json')
    config = validate_config(config)
    
    print(f"🚀 Two-Tower Model Training Pipeline")
    print("=" * 60)

    print(f"✅ Configuration loaded and validated")
    if config.get('SUBSAMPLE_RATIO'):
        print(f"   📊 Subsampling ratio: {config['SUBSAMPLE_RATIO']}")

    # Setup device and memory optimization
    setup_memory_optimization()
    device = get_best_device()

    # Load tokenizer
    print("\n📝 Loading tokenizer...")
    tokenizer = PretrainedTokenizer(config['WORD_TO_IDX_PATH'])

    # Load pretrained embeddings
    print("\n🔢 Loading pretrained embeddings...")
    pretrained_embeddings = load_pretrained_embeddings(config['EMBEDDINGS_PATH'])

    # VOCAB REDUCTION: Drop last 100k words (keep most frequent)
    words_to_drop = 0
    original_vocab_size = tokenizer.vocab_size()
    new_vocab_size = original_vocab_size - words_to_drop
    print(f"🔪 Reducing vocab from {original_vocab_size:,} to {new_vocab_size:,} words")
    
    # Truncate embeddings to new vocab size
    pretrained_embeddings = pretrained_embeddings[:new_vocab_size]
    
    # Filter tokenizer vocabulary to keep only first new_vocab_size words
    filtered_word2idx = {word: idx for word, idx in tokenizer.word2idx.items() if idx < new_vocab_size}
    tokenizer.word2idx = filtered_word2idx
    tokenizer.idx2word = {idx: word for word, idx in filtered_word2idx.items()}

    # Add dimensions to config
    config['VOCAB_SIZE'] = tokenizer.vocab_size()
    config['EMBED_DIM'] = pretrained_embeddings.shape[1]

    # Load datasets using unified DataLoader
    print(f"\n📚 Loading datasets...")
    data_loader = DataLoader(config)
    datasets = data_loader.load_datasets(subsample_ratio=config.get('SUBSAMPLE_RATIO'))
    dataset_stats = data_loader.get_dataset_stats(datasets)

    print(f"\n📊 Dataset Statistics:")
    for split, count in dataset_stats.items():
        print(f"   {split.capitalize()}: {count:,} samples")

    # Create data loaders
    print("\n🔄 Creating data loaders...")
    dataloader_factory = DataLoaderFactory(config)
    dataloaders = dataloader_factory.create_dataloaders(datasets, tokenizer)

    # Create model and trainer
    model, trainer = create_model_and_trainer(
        config, 
        pretrained_embeddings, 
        device,
        checkpoint_path=args.checkpoint
    )
    print_model_summary(model, config)

    # Train the model
    epochs = config.get('EPOCHS', 10)
    print(f"\n🚀 Starting training...")
    if args.checkpoint:
        print(f"   📌 Continuing training from checkpoint: {args.checkpoint}")
    
    history = trainer.train(
        train_loader=dataloaders['train'],
        val_loader=dataloaders.get('validation'),
        epochs=epochs
    )

    print(f"\n📈 Training Summary:")
    print(f"   Final Training Loss: {history['train_losses'][-1]:.4f}")
    if 'val_losses' in history and history['val_losses']:
        print(f"   Final Validation Loss: {history['val_losses'][-1]:.4f}")
        if 'best_val_loss' in history:
            print(f"   Best Validation Loss: {history['best_val_loss']:.4f}")

    # Save the trained model
    print(f"\n💾 Saving model...")
    
    artifacts_dir = save_model_artifacts(
        model=model,
        optimizer=trainer.optimizer,
        config=config,
        epoch=epochs,
        final_loss=history['train_losses'][-1],
        datasets_stats=dataset_stats,
        artifacts_dir="artifacts",
        save_doc_embeddings=True,
        datasets=datasets,
        tokenizer=tokenizer
    )
    
    print(f"✅ Model saved to: {artifacts_dir}")

    # Run the full evaluation on the test set
    print(f"\n\n📊 RUNNING FULL EVALUATION")
    print("=" * 80)
    run_evaluation(artifacts_dir, config, device)

    print(f"\n🎉 Training completed successfully!")
    print(f"📁 Trained model saved in: {artifacts_dir}")


if __name__ == "__main__":
    main() 