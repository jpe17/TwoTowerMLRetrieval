#!/usr/bin/env python3
"""
Two-Tower ML Retrieval - Main Training Script

This script orchestrates the complete training pipeline with modular components.
Supports subsampling datasets and full train/validation/test splits.
"""

import argparse
import sys
import torch
from pathlib import Path
from dotenv import load_dotenv
import os
import wandb

load_dotenv()  # Loads .env file

wandb.login(key=os.getenv("WANDB_API_KEY"))
wandb.init(project="two-tower-ml-retrieval")  # Set your project name

# Add backend to path for imports
sys.path.append(str(Path(__file__).parent))

from data_loader import DataLoader
from tokenizer import PretrainedTokenizer
from dataset import DataLoaderFactory
from model import TwoTowerModel, ModelFactory
from trainer import TrainerFactory
from evaluator import TwoTowerEvaluator
from utils import (
    load_config, validate_config, get_best_device, setup_memory_optimization,
    load_pretrained_embeddings, save_model_artifacts, print_model_summary
)


def main():
    """Main training function (config-only version)."""
    print("🚀 Two-Tower ML Retrieval Training Pipeline")
    print("=" * 60)

    # Load and validate configuration
    print("📋 Loading configuration...")
    config = load_config('backend/config.json')
    config = validate_config(config)

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
    if 'test' in dataloaders:
        print("\n🧪 Testing model...")
        test_results = trainer.test(dataloaders['test'])
    else:
        test_results = {}

    # Save model artifacts
    print("\n💾 Saving model artifacts...")
    final_loss = history['train_losses'][-1] if history['train_losses'] else 0.0
    artifacts_path = save_model_artifacts(
        model=trainer.get_model_for_inference(),
        optimizer=trainer.optimizer,
        config=config,
        epoch=config.get('EPOCHS', 0),
        final_loss=final_loss,
        datasets_stats=dataset_stats
    )

    print(f"\n📈 Training Summary:")
    print(f"   Final Training Loss: {final_loss:.4f}")
    if history['val_losses']:
        print(f"   Final Validation Loss: {history['val_losses'][-1]:.4f}")
        print(f"   Best Validation Loss: {history['best_val_loss']:.4f}")
    if test_results:
        print(f"   Test Loss: {test_results['test_loss']:.4f}")

    print(f"\n✅ Training completed! Artifacts saved to: {artifacts_path}")

    # Get best model for evaluation
    best_model = trainer.get_model_for_inference()

    # Run evaluation
    if 'test' in datasets and datasets['test']:
        print("\n🔍 Running comprehensive evaluation...")
        evaluator = TwoTowerEvaluator(best_model, tokenizer, device)

        # Retrieval evaluation
        eval_metrics = evaluator.evaluate_retrieval(
            test_data=datasets['test'][:500],  # Sample for faster evaluation
            num_samples=config.get('NUM_SAMPLES_EVAL'),
            num_distractors=config.get('NUM_DISTRACTORS_EVAL')
        )

    print("\n🎉 Pipeline completed successfully!")


if __name__ == "__main__":
    main() 