#!/usr/bin/env python3
"""
Vocabulary Reduction Script

Reduces vocabulary size by removing the least frequent words and creates
corresponding reduced embeddings.
"""

import pickle
import numpy as np
import os
from typing import Dict, Tuple
import json


def reduce_vocabulary(
    word_to_idx_path: str,
    embeddings_path: str,
    words_to_drop: int = 100000,
    output_dir: str = "data/reduced_vocab"
) -> Tuple[str, str]:
    """
    Reduce vocabulary by dropping the last N words (assumed to be least frequent).
    
    Args:
        word_to_idx_path: Path to original word_to_idx pickle file
        embeddings_path: Path to original embeddings numpy file
        words_to_drop: Number of words to drop from the end
        output_dir: Directory to save reduced vocabulary files
        
    Returns:
        Tuple of (new_word_to_idx_path, new_embeddings_path)
    """
    print(f"🔪 Reducing vocabulary by dropping {words_to_drop:,} words...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load original vocabulary
    print("📚 Loading original vocabulary...")
    with open(word_to_idx_path, 'rb') as f:
        original_word2idx = pickle.load(f)
    
    original_vocab_size = len(original_word2idx)
    print(f"   Original vocabulary size: {original_vocab_size:,}")
    
    # Load original embeddings
    print("🔢 Loading original embeddings...")
    original_embeddings = np.load(embeddings_path)
    print(f"   Original embeddings shape: {original_embeddings.shape}")
    
    # Verify consistency
    if original_embeddings.shape[0] != original_vocab_size:
        raise ValueError(f"Vocabulary size ({original_vocab_size}) doesn't match embeddings shape ({original_embeddings.shape[0]})")
    
    # Calculate new vocabulary size
    new_vocab_size = original_vocab_size - words_to_drop
    print(f"   New vocabulary size: {new_vocab_size:,}")
    
    # Create reduced vocabulary
    # Assuming the word2idx is ordered by frequency (most frequent first)
    # We keep the first new_vocab_size words
    print("✂️  Creating reduced vocabulary...")
    
    # Sort by index to maintain order (assuming lower indices = more frequent)
    sorted_items = sorted(original_word2idx.items(), key=lambda x: x[1])
    
    # Keep only the first new_vocab_size words
    reduced_items = sorted_items[:new_vocab_size]
    
    # Create new word2idx with continuous indices
    new_word2idx = {}
    old_to_new_idx = {}  # Mapping from old indices to new indices
    
    for new_idx, (word, old_idx) in enumerate(reduced_items):
        new_word2idx[word] = new_idx
        old_to_new_idx[old_idx] = new_idx
    
    print(f"   Kept {len(new_word2idx):,} words")
    
    # Create reduced embeddings
    print("🔢 Creating reduced embeddings...")
    new_embeddings = np.zeros((new_vocab_size, original_embeddings.shape[1]))
    
    for old_idx, new_idx in old_to_new_idx.items():
        new_embeddings[new_idx] = original_embeddings[old_idx]
    
    print(f"   New embeddings shape: {new_embeddings.shape}")
    
    # Save reduced vocabulary
    new_word_to_idx_path = os.path.join(output_dir, "word_to_idx_reduced.pkl")
    print(f"💾 Saving reduced vocabulary to {new_word_to_idx_path}")
    with open(new_word_to_idx_path, 'wb') as f:
        pickle.dump(new_word2idx, f)
    
    # Save reduced embeddings
    new_embeddings_path = os.path.join(output_dir, "embeddings_reduced.npy")
    print(f"💾 Saving reduced embeddings to {new_embeddings_path}")
    np.save(new_embeddings_path, new_embeddings)
    
    # Save reduction info
    reduction_info = {
        "original_vocab_size": original_vocab_size,
        "new_vocab_size": new_vocab_size,
        "words_dropped": words_to_drop,
        "reduction_ratio": new_vocab_size / original_vocab_size,
        "original_word_to_idx_path": word_to_idx_path,
        "original_embeddings_path": embeddings_path,
        "new_word_to_idx_path": new_word_to_idx_path,
        "new_embeddings_path": new_embeddings_path
    }
    
    info_path = os.path.join(output_dir, "reduction_info.json")
    with open(info_path, 'w') as f:
        json.dump(reduction_info, f, indent=2)
    
    print(f"📊 Vocabulary reduction summary:")
    print(f"   Original size: {original_vocab_size:,}")
    print(f"   New size: {new_vocab_size:,}")
    print(f"   Reduction: {(1 - new_vocab_size/original_vocab_size)*100:.1f}%")
    print(f"   Info saved to: {info_path}")
    
    return new_word_to_idx_path, new_embeddings_path


def update_config_for_reduced_vocab(
    config_path: str,
    new_word_to_idx_path: str,
    new_embeddings_path: str,
    output_config_path: str = None
):
    """Update configuration to use reduced vocabulary."""
    if output_config_path is None:
        output_config_path = config_path.replace('.json', '_reduced.json')
    
    # Load original config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Update paths
    config['WORD_TO_IDX_PATH'] = new_word_to_idx_path
    config['EMBEDDINGS_PATH'] = new_embeddings_path
    
    # Save updated config
    with open(output_config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"⚙️  Updated configuration saved to: {output_config_path}")
    return output_config_path


if __name__ == "__main__":
    # Configuration
    config_path = "backend/config.json"
    
    # Load config to get current paths
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Reduce vocabulary
    new_word_to_idx_path, new_embeddings_path = reduce_vocabulary(
        word_to_idx_path=config['WORD_TO_IDX_PATH'],
        embeddings_path=config['EMBEDDINGS_PATH'],
        words_to_drop=100000,
        output_dir="data/reduced_vocab"
    )
    
    # Update config
    new_config_path = update_config_for_reduced_vocab(
        config_path=config_path,
        new_word_to_idx_path=new_word_to_idx_path,
        new_embeddings_path=new_embeddings_path
    )
    
    print(f"\n✅ Vocabulary reduction complete!")
    print(f"📁 Use new config: {new_config_path}") 