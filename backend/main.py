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
from evaluator import SimpleEvaluator
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

    # Demo the evaluator with actual data
    if 'test' in datasets and datasets['test']:
        print("\n🔍 Running evaluator demo...")
        
        # Get the trained model for evaluation
        best_model = trainer.get_model_for_inference()
        evaluator = SimpleEvaluator(best_model, tokenizer, device)
        
        # Extract documents from test data for demo
        test_sample = datasets['test'][:50]  # Use first 50 test samples
        all_docs = []
        test_queries = []
        positive_docs = {}
        
        for query, pos_doc, neg_doc in test_sample:
            all_docs.extend([pos_doc, neg_doc])
            if query not in test_queries:
                test_queries.append(query)
                positive_docs[query] = [pos_doc]
            else:
                positive_docs[query].append(pos_doc)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_docs = []
        for doc in all_docs:
            if doc not in seen:
                seen.add(doc)
                unique_docs.append(doc)
        
        # 1. Query evaluation demo
        if test_queries:
            demo_query = test_queries[0]
            results = evaluator.evaluate_query(
                query=demo_query,
                documents=unique_docs,
                positive_docs=positive_docs[demo_query]
            )
            evaluator.print_query_results(demo_query, results)
        
        # 2. Similarity search demo
        demo_text = "What is the capital of France?"
        similar_docs = evaluator.search_similar(
            text=demo_text,
            documents=unique_docs[:20]  # Use subset for demo
        )
        evaluator.print_search_results(demo_text, similar_docs)

    print("\n🎉 Pipeline completed successfully!")


if __name__ == "__main__":
    main() 