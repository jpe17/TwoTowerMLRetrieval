#!/usr/bin/env python3
"""
Two-Tower ML Retrieval - Main Training Script

Simple main script that works with the simplified trainer.
"""

import sys
import torch
from pathlib import Path
from dotenv import load_dotenv
import os
import wandb

load_dotenv()  # Loads .env file

# Only login, don't init here (trainer will handle init)
wandb.login(key=os.getenv("WANDB_API_KEY"))

# Add backend to path for imports
sys.path.append(str(Path(__file__).parent))

from data_loader import DataLoader
from tokenizer import PretrainedTokenizer
from dataset import DataLoaderFactory
from model import ModelFactory
from trainer import TrainerFactory
from evaluator import SimpleEvaluator
from utils import (
    load_config, validate_config, get_best_device, setup_memory_optimization,
    load_pretrained_embeddings, print_model_summary
)


def main():
    """Simple main training function."""
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

    print(f"\n📈 Training Summary:")
    print(f"   Final Training Loss: {history['train_losses'][-1]:.4f}")
    if history['val_losses']:
        print(f"   Final Validation Loss: {history['val_losses'][-1]:.4f}")
        print(f"   Best Validation Loss: {history['best_val_loss']:.4f}")

    # Demo the evaluator with test data
    if 'test' in datasets and datasets['test']:
        print("\n🔍 Testing Evaluator with Sample Queries...")
        print("=" * 60)
        
        # Create evaluator
        evaluator = SimpleEvaluator(trainer.model, tokenizer, device)
        
        # Get sample data from test set
        test_sample = datasets['test'][:50]  # Use first 50 test samples
        all_docs = []
        test_queries = []
        positive_docs = {}
        
        for query, pos_doc, neg_doc in test_sample:
            all_docs.extend([pos_doc, neg_doc])
            if query not in test_queries[:3]:  # Limit to 3 demo queries
                test_queries.append(query)
                positive_docs[query] = [pos_doc]
        
        # Remove duplicates
        unique_docs = list(dict.fromkeys(all_docs))
        
        # Test 2-3 sample queries
        for i, demo_query in enumerate(test_queries[:3], 1):
            print(f"\n📝 Demo Query {i}:")
            results = evaluator.evaluate_query(
                query=demo_query,
                documents=unique_docs,
                positive_docs=positive_docs[demo_query]
            )
            evaluator.print_query_results(demo_query, results)
        
        print("\n" + "=" * 60)

    print("\n🎉 Training completed successfully!")


if __name__ == "__main__":
    main() 