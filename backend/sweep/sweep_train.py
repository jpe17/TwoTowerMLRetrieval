#!/usr/bin/env python3
"""
WandB Sweep Training Script for Two-Tower ML Retrieval

Simple sweep training that works with the simplified trainer.
"""

import sys
import os
import wandb
from pathlib import Path
from dotenv import load_dotenv

# Add backend to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import modules using direct imports (not relative)
sys.path.append('backend')
from data_loader import DataLoader
from tokenizer import PretrainedTokenizer
from dataset import DataLoaderFactory
from model import ModelFactory
from trainer import TwoTowerTrainer
from utils import (
    load_config, validate_config, get_best_device, setup_memory_optimization,
    load_pretrained_embeddings, print_model_summary
)


def sweep_train():
    """Simple training function for WandB sweeps."""
    # Initialize WandB run
    wandb.init()
    
    # Get sweep config from wandb
    sweep_config = wandb.config
    
    print("🔄 WandB Sweep Training - Two-Tower ML Retrieval")
    print("=" * 60)
    print(f"🎯 Sweep parameters: {dict(sweep_config)}")

    # Load base configuration
    print("📋 Loading base configuration...")
    config = load_config('backend/config.json')
    
    # Override config with sweep parameters
    for key, value in sweep_config.items():
        if key in config:
            print(f"   🔧 Overriding {key}: {config[key]} → {value}")
            config[key] = value
        else:
            print(f"   ➕ Adding {key}: {value}")
            config[key] = value
    
    # Validate the updated configuration
    config = validate_config(config)

    # Setup device and memory optimization
    setup_memory_optimization()
    device = get_best_device()

    # Load tokenizer
    print("\n📝 Loading tokenizer...")
    tokenizer = PretrainedTokenizer(config['WORD_TO_IDX_PATH'])

    # Load pretrained embeddings
    print("\n🔢 Loading pretrained embeddings...")
    pretrained_embeddings = load_pretrained_embeddings(config['EMBEDDINGS_PATH'])

    # VOCAB REDUCTION: Drop last words if needed (keep most frequent)
    words_to_drop = 0
    original_vocab_size = tokenizer.vocab_size()
    new_vocab_size = original_vocab_size - words_to_drop
    
    # Truncate embeddings to new vocab size
    pretrained_embeddings = pretrained_embeddings[:new_vocab_size]
    
    # Filter tokenizer vocabulary to keep only first new_vocab_size words
    if words_to_drop > 0:
        filtered_word2idx = {word: idx for word, idx in tokenizer.word2idx.items() if idx < new_vocab_size}
        tokenizer.word2idx = filtered_word2idx
        tokenizer.idx2word = {idx: word for word, idx in filtered_word2idx.items()}

    # Add dimensions to config
    config['VOCAB_SIZE'] = tokenizer.vocab_size()
    config['EMBED_DIM'] = pretrained_embeddings.shape[1]

    # Load datasets
    print("\n📚 Loading datasets...")
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

    # Create model
    print("\n🏗️  Creating model...")
    model = ModelFactory.create_two_tower_model(config, pretrained_embeddings)
    model = model.to(device)

    print_model_summary(model, config)

    # Create trainer
    print("\n🎯 Setting up trainer...")
    trainer = TwoTowerTrainer(model, config, device)

    # Train the model
    print("\n🚀 Starting training...")
    history = trainer.train(
        train_loader=dataloaders['train'],
        val_loader=dataloaders.get('validation'),
        epochs=config.get('EPOCHS')
    )

    # Log final metrics to WandB
    final_metrics = {
        'final_train_loss': history['train_losses'][-1] if history['train_losses'] else 0.0,
        'best_val_loss': history['best_val_loss'],
        'total_params': sum(p.numel() for p in model.parameters()),
        'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad)
    }
    
    if history['val_losses']:
        final_metrics['final_val_loss'] = history['val_losses'][-1]
    
    wandb.log(final_metrics)

    print(f"\n📈 Training Summary:")
    print(f"   Final Training Loss: {final_metrics['final_train_loss']:.4f}")
    if 'final_val_loss' in final_metrics:
        print(f"   Final Validation Loss: {final_metrics['final_val_loss']:.4f}")
        print(f"   Best Validation Loss: {final_metrics['best_val_loss']:.4f}")

    print(f"\n✅ Sweep run completed!")


if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Login to WandB (using API key from environment)
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    
    # Run the sweep training
    sweep_train() 