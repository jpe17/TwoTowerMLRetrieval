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
import numpy as np
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
from eval_test import run_evaluation
from utils import (
    load_config, validate_config, get_best_device, setup_memory_optimization,
    load_pretrained_embeddings, print_model_summary, load_model_artifacts,
    save_model_artifacts, parse_args, load_embeddings_and_maps, create_model_and_trainer
)



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