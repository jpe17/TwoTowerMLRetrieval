#!/usr/bin/env python3
"""
Example Usage of Two-Tower ML Retrieval System

This script demonstrates how to use the modular system for training
and evaluating the two-tower retrieval model with different configurations.
"""

import json
from pathlib import Path

# Import the backend modules
from backend import (
    DataLoader, PretrainedTokenizer, DataLoaderFactory,
    ModelFactory, TrainerFactory, TwoTowerEvaluator,
    load_config, validate_config, get_best_device, setup_memory_optimization,
    load_pretrained_embeddings, save_model_artifacts, print_model_summary
)


def example_quick_training():
    """Example: Quick training with subsampled data."""
    print("🚀 Example: Quick Training with Subsampled Data")
    print("=" * 60)
    
    # Load and modify config for quick training
    config = load_config('backend/config.json')
    config['EPOCHS'] = 2  # Quick training
    config['BATCH_SIZE'] = 32  # Smaller batch size
    config = validate_config(config)
    
    # Setup
    setup_memory_optimization()
    device = get_best_device()
    
    # Load components
    print("📝 Loading tokenizer and embeddings...")
    tokenizer = PretrainedTokenizer(config['WORD_TO_IDX_PATH'])
    pretrained_embeddings = load_pretrained_embeddings(config['EMBEDDINGS_PATH'])
    
    config['VOCAB_SIZE'] = tokenizer.vocab_size()
    config['EMBED_DIM'] = pretrained_embeddings.shape[1]
    
    # Load data with subsampling (10% of data for quick demo)
    print("📚 Loading datasets with 10% subsampling...")
    data_loader = DataLoader(config)
    datasets = data_loader.load_datasets(subsample_ratio=0.1)
    dataset_stats = data_loader.get_dataset_stats(datasets)
    
    print(f"📊 Dataset sizes after subsampling:")
    for split, count in dataset_stats.items():
        print(f"   {split}: {count:,} samples")
    
    # Create dataloaders
    dataloader_factory = DataLoaderFactory(config)
    dataloaders = dataloader_factory.create_dataloaders(datasets, tokenizer)
    
    # Create model
    print("🏗️  Creating model...")
    model = ModelFactory.create_two_tower_model(config, pretrained_embeddings).to(device)
    print_model_summary(model, config)
    
    # Train
    print("🎯 Training model...")
    trainer = TrainerFactory.create_trainer(config, model, device)
    history = trainer.train(
        train_loader=dataloaders['train'],
        val_loader=dataloaders.get('validation'),
        epochs=config['EPOCHS']
    )
    
    # Quick evaluation
    print("🔍 Quick evaluation...")
    evaluator = TwoTowerEvaluator(trainer.get_model_for_inference(), tokenizer, device)
    
    # Test search functionality
    demo_docs = [datasets['test'][i][1] for i in range(min(50, len(datasets['test'])))]
    demo_query = datasets['test'][0][0]
    
    print(f"\n🔎 Demo Search:")
    print(f"Query: {demo_query[:100]}...")
    results = evaluator.search(demo_query, demo_docs, top_k=3)
    
    for i, (doc, score) in enumerate(results):
        print(f"{i+1}. Score: {score:.4f} - {doc[:80]}...")
    
    print("✅ Quick training example completed!")
    return trainer.get_model_for_inference(), evaluator


def example_custom_configuration():
    """Example: Training with custom configuration."""
    print("\n🔧 Example: Custom Configuration Training")
    print("=" * 60)
    
    # Create custom configuration
    custom_config = {
        'TRAIN_DATASET_PATH': 'data/ms_marco_train.parquet',
        'VAL_DATASET_PATH': 'data/ms_marco_validation.parquet',
        'TEST_DATASET_PATH': 'data/ms_marco_test.parquet',
        'EMBEDDINGS_PATH': 'data/embeddings.npy',
        'WORD_TO_IDX_PATH': 'data/word_to_idx.pkl',
        
        # Custom model settings
        'NUM_TRIPLETS_PER_QUERY': 2,  # More triplets per query
        'HIDDEN_DIM': 64,              # Smaller model for demo
        'MARGIN': 0.5,                 # Smaller margin
        'LR': 0.005,                   # Higher learning rate
        'BATCH_SIZE': 16,              # Smaller batch
        'EPOCHS': 1,                   # Very quick training
        'RNN_TYPE': 'LSTM',            # Use LSTM instead of GRU
        'NUM_LAYERS': 2,               # Multi-layer RNN
        'DROPOUT': 0.1,                # Add some dropout
        'LOSS_TYPE': 'cosine',         # Use cosine similarity loss
    }
    
    print("⚙️  Custom configuration:")
    for key, value in custom_config.items():
        if key not in ['TRAIN_DATASET_PATH', 'VAL_DATASET_PATH', 'TEST_DATASET_PATH', 
                       'EMBEDDINGS_PATH', 'WORD_TO_IDX_PATH']:
            print(f"   {key}: {value}")
    
    # Use the programmatic training interface
    from backend.main import train_with_custom_config
    
    model, trainer, evaluator, artifacts_path = train_with_custom_config(
        config_dict=custom_config,
        subsample_ratio=0.05,  # Use 5% of data
        device='auto'  # Auto-detect device
    )
    
    print(f"✅ Custom configuration training completed!")
    print(f"   Artifacts saved to: {artifacts_path}")
    
    return model, evaluator


def example_evaluation_only():
    """Example: Evaluation and search with existing model components."""
    print("\n🔍 Example: Search and Similarity Analysis")
    print("=" * 60)
    
    # Load configuration and components
    config = load_config('backend/config.json')
    tokenizer = PretrainedTokenizer(config['WORD_TO_IDX_PATH'])
    pretrained_embeddings = load_pretrained_embeddings(config['EMBEDDINGS_PATH'])
    
    config['VOCAB_SIZE'] = tokenizer.vocab_size()
    config['EMBED_DIM'] = pretrained_embeddings.shape[1]
    
    # Create a simple model for demo (in practice, you'd load a trained model)
    device = get_best_device()
    model = ModelFactory.create_two_tower_model(config, pretrained_embeddings).to(device)
    evaluator = TwoTowerEvaluator(model, tokenizer, device)
    
    # Example documents for search
    documents = [
        "Machine learning is a subset of artificial intelligence that enables computers to learn.",
        "Deep learning uses neural networks with multiple layers to process data.",
        "Natural language processing helps computers understand and generate human language.",
        "Computer vision allows machines to interpret and analyze visual information.",
        "Reinforcement learning teaches agents to make decisions through trial and error.",
        "The weather today is sunny with a temperature of 75 degrees.",
        "Cooking pasta requires boiling water and adding salt for flavor.",
        "Basketball is a popular sport played with two teams of five players each."
    ]
    
    # Example queries
    queries = [
        "What is machine learning?",
        "How does deep learning work?",
        "Tell me about sports"
    ]
    
    print("🔎 Search Examples:")
    for query in queries:
        print(f"\nQuery: {query}")
        results = evaluator.search(query, documents, top_k=3)
        for i, (doc, score) in enumerate(results):
            print(f"  {i+1}. Score: {score:.4f} - {doc}")
    
    # Similarity analysis
    print("\n📊 Similarity Analysis:")
    text_pairs = [
        ("machine learning", "artificial intelligence"),
        ("deep learning", "neural networks"),
        ("cooking pasta", "machine learning"),
        ("basketball", "sports")
    ]
    
    similarities = evaluator.similarity_analysis(text_pairs)
    for text1, text2, sim in similarities:
        print(f"  '{text1}' vs '{text2}': {sim:.4f}")
    
    print("✅ Evaluation example completed!")


def main():
    """Run all examples."""
    print("🎯 Two-Tower ML Retrieval - Usage Examples")
    print("=" * 80)
    
    try:
        # Example 1: Quick training with subsampling
        model1, evaluator1 = example_quick_training()
        
        # Example 2: Custom configuration
        model2, evaluator2 = example_custom_configuration()
        
        # Example 3: Evaluation and search
        example_evaluation_only()
        
        print("\n🎉 All examples completed successfully!")
        print("\nTo run the full training pipeline, use:")
        print("  python backend/main.py --subsample 0.1 --epochs 5")
        print("  python backend/main.py --subsample 0.01 --epochs 2 --demo")
        
    except Exception as e:
        print(f"❌ Error running examples: {str(e)}")
        print("Make sure all data files are available in the 'data/' directory.")


if __name__ == "__main__":
    main() 