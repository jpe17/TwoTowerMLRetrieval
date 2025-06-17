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
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Two-Tower ML Retrieval Model')
    parser.add_argument('--config', type=str, default='backend/config.json',
                        help='Path to configuration file')
    parser.add_argument('--subsample', type=float, default=None,
                        help='Subsample ratio (0.0 < ratio <= 1.0)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (overrides config)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size (overrides config)')
    parser.add_argument('--eval-only', action='store_true',
                        help='Only run evaluation (requires trained model)')
    parser.add_argument('--demo', action='store_true',
                        help='Run interactive search demo')
    
    args = parser.parse_args()
    
    print("🚀 Two-Tower ML Retrieval Training Pipeline")
    print("=" * 60)
    
    # Load and validate configuration
    print("📋 Loading configuration...")
    config = load_config(args.config)
    config = validate_config(config)
    
    # Override config with command line arguments
    if args.subsample is not None:
        config['SUBSAMPLE_RATIO'] = args.subsample
    if args.epochs is not None:
        config['EPOCHS'] = args.epochs
    if args.batch_size is not None:
        config['BATCH_SIZE'] = args.batch_size
    
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
    
    if not args.eval_only:
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
    else:
        print("⚠️  Evaluation-only mode requires a pre-trained model!")
        print("   Please train a model first or provide model loading functionality.")
        return
    
    # Run evaluation
    if 'test' in datasets and datasets['test']:
        print("\n🔍 Running comprehensive evaluation...")
        evaluator = TwoTowerEvaluator(best_model, tokenizer, device)
        
        # Retrieval evaluation
        eval_metrics = evaluator.evaluate_retrieval(
            test_data=datasets['test'][:500],  # Sample for faster evaluation
            num_samples=100,
            num_distractors=20
        )
        
        # Interactive demo
        if args.demo:
            print("\n🎯 Running interactive search demo...")
            # Extract documents from test data for demo
            demo_documents = []
            for _, pos_doc, neg_doc in datasets['test'][:200]:
                demo_documents.extend([pos_doc, neg_doc])
            
            evaluator.interactive_search_demo(demo_documents, num_demos=3)
    
    print("\n🎉 Pipeline completed successfully!")


def train_with_custom_config(
    config_dict: dict,
    subsample_ratio: float = None,
    device: str = None
):
    """
    Programmatic training interface for custom configurations.
    
    Args:
        config_dict: Configuration dictionary
        subsample_ratio: Optional subsample ratio
        device: Optional device specification
        
    Returns:
        Tuple of (trained_model, trainer, evaluator, artifacts_path)
    """
    print("🔧 Running programmatic training...")
    
    # Validate config
    config = validate_config(config_dict)
    if subsample_ratio:
        config['SUBSAMPLE_RATIO'] = subsample_ratio
    
    # Setup
    setup_memory_optimization()
    if device:
        device = torch.device(device)
    else:
        device = get_best_device()
    
    # Load components
    tokenizer = PretrainedTokenizer(config['WORD_TO_IDX_PATH'])
    pretrained_embeddings = load_pretrained_embeddings(config['EMBEDDINGS_PATH'])
    
    config['VOCAB_SIZE'] = tokenizer.vocab_size()
    config['EMBED_DIM'] = pretrained_embeddings.shape[1]
    
    # Load data
    data_loader = DataLoader(config)
    datasets = data_loader.load_datasets(subsample_ratio=config.get('SUBSAMPLE_RATIO'))
    dataset_stats = data_loader.get_dataset_stats(datasets)
    
    # Create dataloaders
    dataloader_factory = DataLoaderFactory(config)
    dataloaders = dataloader_factory.create_dataloaders(datasets, tokenizer)
    
    # Create and train model
    model = ModelFactory.create_two_tower_model(config, pretrained_embeddings).to(device)
    trainer = TrainerFactory.create_trainer(config, model, device)
    
    # Train
    history = trainer.train(
        train_loader=dataloaders['train'],
        val_loader=dataloaders.get('validation'),
        epochs=config.get('EPOCHS')
    )
    
    # Save artifacts
    final_loss = history['train_losses'][-1] if history['train_losses'] else 0.0
    artifacts_path = save_model_artifacts(
        model=trainer.get_model_for_inference(),
        optimizer=trainer.optimizer,
        config=config,
        epoch=config.get('EPOCHS', 0),
        final_loss=final_loss,
        datasets_stats=dataset_stats
    )
    
    # Create evaluator
    evaluator = TwoTowerEvaluator(trainer.get_model_for_inference(), tokenizer, device)
    
    return trainer.get_model_for_inference(), trainer, evaluator, artifacts_path


if __name__ == "__main__":
    main() 