#!/usr/bin/env python3
"""
WandB Sweep Training Script for Two-Tower ML Retrieval

This script is designed to work with WandB sweeps without modifying the main training pipeline.
It overrides configuration parameters with sweep parameters from WandB.
"""

import sys
import os
import wandb
from pathlib import Path
from dotenv import load_dotenv

# Add backend to path for imports
sys.path.append(str(Path(__file__).parent))

from data_loader import DataLoader
from tokenizer import PretrainedTokenizer
from dataset import DataLoaderFactory
from model import TwoTowerModel, ModelFactory
from trainer import TrainerFactory
from utils import (
    load_config, validate_config, get_best_device, setup_memory_optimization,
    load_pretrained_embeddings, save_model_artifacts, print_model_summary
)


def sweep_train():
    """Training function for WandB sweeps."""
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
    trainer = TrainerFactory.create_trainer(config, model, device)

    # Train the model
    print("\n🚀 Starting training...")
    history = trainer.train(
        train_loader=dataloaders['train'],
        val_loader=dataloaders.get('validation'),
        epochs=config.get('EPOCHS')
    )

    # Test the model
    test_results = {}
    if 'test' in dataloaders:
        print("\n🧪 Testing model...")
        test_results = trainer.test(dataloaders['test'])

    # Get best validation metrics
    best_val_metrics = None
    if trainer.val_metrics_history:
        # Find epoch with best validation loss
        best_epoch_idx = history['val_losses'].index(history['best_val_loss'])
        best_val_metrics = trainer.val_metrics_history[best_epoch_idx]

    # Log final metrics to WandB
    final_metrics = {
        'final_train_loss': history['train_losses'][-1] if history['train_losses'] else 0.0,
        'best_val_loss': history['best_val_loss'],
        'total_params': sum(p.numel() for p in model.parameters()),
        'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad)
    }
    
    if history['val_losses']:
        final_metrics['final_val_loss'] = history['val_losses'][-1]
    
    if test_results:
        final_metrics['test_loss'] = test_results['test_loss']
    
    # Add best validation metrics
    if best_val_metrics:
        final_metrics.update({
            'best_val_triplet_accuracy': best_val_metrics['triplet_accuracy'],
            'best_val_margin_violations': best_val_metrics['margin_violations'],
            'best_val_similarity_gap': best_val_metrics['similarity_gap'],
            'best_val_avg_pos_similarity': best_val_metrics['avg_pos_similarity'],
            'best_val_avg_neg_similarity': best_val_metrics['avg_neg_similarity'],
            'best_val_distance_ratio': best_val_metrics['distance_ratio'],
            'best_val_query_magnitude': best_val_metrics['query_magnitude'],
            'best_val_pos_magnitude': best_val_metrics['pos_magnitude'],
            'best_val_neg_magnitude': best_val_metrics['neg_magnitude']
        })
    
    # Add final training metrics
    if trainer.train_metrics_history:
        final_train_metrics = trainer.train_metrics_history[-1]
        final_metrics.update({
            'final_train_triplet_accuracy': final_train_metrics['triplet_accuracy'],
            'final_train_similarity_gap': final_train_metrics['similarity_gap'],
            'final_train_margin_violations': final_train_metrics['margin_violations'],
            'final_train_query_magnitude': final_train_metrics['query_magnitude'],
            'final_train_pos_magnitude': final_train_metrics['pos_magnitude'],
            'final_train_neg_magnitude': final_train_metrics['neg_magnitude']
        })
    
    wandb.log(final_metrics)

    print(f"\n📈 Training Summary:")
    print(f"   Final Training Loss: {final_metrics['final_train_loss']:.4f}")
    if 'final_val_loss' in final_metrics:
        print(f"   Final Validation Loss: {final_metrics['final_val_loss']:.4f}")
        print(f"   Best Validation Loss: {final_metrics['best_val_loss']:.4f}")
    if 'test_loss' in final_metrics:
        print(f"   Test Loss: {final_metrics['test_loss']:.4f}")
    
    # Print best validation metrics
    if best_val_metrics:
        print(f"\n🎯 Best Validation Metrics (at best loss epoch):")
        print(f"   Triplet Accuracy: {best_val_metrics['triplet_accuracy']:.3f}")
        print(f"   Similarity Gap: {best_val_metrics['similarity_gap']:.3f}")
        print(f"   Margin Violations: {best_val_metrics['margin_violations']:.3f}")

    print(f"\n✅ Sweep run completed!")


if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Login to WandB (using API key from environment)
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    
    # Run the sweep training
    sweep_train() 