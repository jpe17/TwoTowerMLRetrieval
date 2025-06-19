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
from evaluator import SimpleEvaluator, AdvancedEvaluator
from utils import (
    load_config, validate_config, get_best_device, setup_memory_optimization,
    load_pretrained_embeddings, print_model_summary, load_model_artifacts,
    save_model_artifacts
)


def create_model_and_trainer(config, pretrained_embeddings, device):
    """Create two-tower model and trainer."""
    print("\n🏗️  Creating two-tower model...")
    model = ModelFactory.create_two_tower_model(config, pretrained_embeddings)
    model = model.to(device)
    
    # Create trainer using factory
    trainer = TwoTowerTrainer(model, config, device)
    
    return model, trainer


def demo_evaluator(model, trainer, tokenizer, device, datasets, config):
    """Demo the evaluator with test data."""
    if 'test' not in datasets or not datasets['test']:
        return
    
    print(f"\n🔍 Testing Evaluator...")
    print("=" * 60)
    
    # Create retrieval evaluator
    evaluator = SimpleEvaluator(trainer.model, tokenizer, device, config)
    
    # Get sample data from test set - create proper query-specific document sets
    test_sample = datasets['test'][:20]  # Use fewer samples for cleaner demo
    
    # Process each demo query individually to ensure correct positive docs are included
    demo_queries_data = []
    seen_queries = set()
    
    for query, pos_doc, neg_doc in test_sample:
        if query not in seen_queries and len(demo_queries_data) < 3:
            seen_queries.add(query)
            
            # Create a focused document set for this query
            query_docs = [pos_doc]  # Start with the correct positive document
            
            # Add some other documents from the test set as distractors
            for other_query, other_pos, other_neg in test_sample[:15]:
                if other_query != query:  # Don't add docs from same query
                    query_docs.extend([other_pos, other_neg])
            
            # Remove duplicates while preserving order
            unique_query_docs = []
            seen = set()
            for doc in query_docs:
                if doc not in seen:
                    unique_query_docs.append(doc)
                    seen.add(doc)
            
            demo_queries_data.append({
                'query': query,
                'documents': unique_query_docs[:20],  # Limit to 20 docs for manageable demo
                'positive_docs': [pos_doc]
            })
    
    # Test the demo queries
    for i, query_data in enumerate(demo_queries_data, 1):
        print(f"\n📝 Demo Query {i}:")
        results = evaluator.evaluate_query(
            query=query_data['query'],
            documents=query_data['documents'],
            positive_docs=query_data['positive_docs']
        )
        evaluator.print_query_results(query_data['query'], results)
    
    print("\n" + "=" * 60)


def main():
    """Main training function."""
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
    model, trainer = create_model_and_trainer(config, pretrained_embeddings, device)
    print_model_summary(model, config)

    # Train the model
    epochs = config.get('EPOCHS', 10)
    print(f"\n🚀 Starting training...")
    history = trainer.train(
        train_loader=dataloaders['train'],
        val_loader=dataloaders.get('validation'),
        epochs=epochs
    )

    print(f"\n📈 Training Summary:")
    print(f"   Final Training Loss: {history['train_losses'][-1]:.4f}")
    if history.get('val_losses'):
        print(f"   Final Validation Loss: {history['val_losses'][-1]:.4f}")
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

    # Demo the evaluator
    demo_evaluator(model, trainer, tokenizer, device, datasets, config)

    print(f"\n🎉 Training completed successfully!")
    print(f"📁 Trained model saved in: {artifacts_dir}")


if __name__ == "__main__":
    main() 