import pandas as pd
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from backend.tokenizer import PretrainedTokenizer
from backend.model import TripletDataset, collate_fn, RNNEncoder, triplet_loss_function
from backend.data_processing import flatten_data, add_negative_samples_fast, to_triplets, filter_valid_data, convert_to_training_format
import random

# Load config
with open('backend/config.json', 'r') as f:
    config = json.load(f)

# Dataset paths
datasets = {
    'train': config['TRAIN_DATASET_PATH'],
    'validation': config['VAL_DATASET_PATH'], 
    'test': config['TEST_DATASET_PATH']
}

# Sampling configuration - modify these as needed
print(f"📊 SAMPLING CONFIGURATION:")
print(f"  Total samples to process: {config['TOTAL_SAMPLES']:,}")
print(f"  Train split: {config['TRAIN_SPLIT']*100:.0f}% ({int(config['TOTAL_SAMPLES']*config['TRAIN_SPLIT']):,} samples)")
print(f"  Test split: {config['TEST_SPLIT']*100:.0f}% ({int(config['TOTAL_SAMPLES']*config['TEST_SPLIT']):,} samples)")
print(f"  Validation split: {(1-config['TRAIN_SPLIT']-config['TEST_SPLIT'])*100:.0f}% ({int(config['TOTAL_SAMPLES']*(1-config['TRAIN_SPLIT']-config['TEST_SPLIT'])):,} samples)")

results = {}

print("\nProcessing datasets to triplet format...")
print("="*50)

# Calculate samples per dataset
samples_per_dataset = {
    'train': int(config['TOTAL_SAMPLES'] * config['TRAIN_SPLIT']),
    'test': int(config['TOTAL_SAMPLES'] * config['TEST_SPLIT']),
    'validation': int(config['TOTAL_SAMPLES'] * (1 - config['TRAIN_SPLIT'] - config['TEST_SPLIT']))
}

for name, input_path in datasets.items():
    target_samples = samples_per_dataset[name]
    print(f"\n📁 Processing {name.upper()} dataset (target: {target_samples:,} samples)...")
    print(f"Loading: {input_path}")
    
    # Step 1: Load data
    df = pd.read_parquet(input_path, engine='fastparquet')
    print(f"  Loaded: {len(df):,} samples")
    
    # Step 2: Early sampling - cut here to save processing time
    if len(df) > target_samples:
        df = df.sample(n=target_samples, random_state=42).reset_index(drop=True)
        print(f"  ✂️ Sampled down to: {len(df):,} samples")
    
    # Step 3: Filter valid data
    df_filtered = filter_valid_data(df)
    print(f"  After filtering: {len(df_filtered):,} samples")
    
    # Step 4: Flatten data (nested passages to flat rows)
    print("  🔄 Flattening data...")
    flattened = flatten_data(df_filtered)
    print(f"  Flattened: {len(flattened):,} rows")
    
    # Step 5: Add negative samples
    print("  ➕ Adding negative samples...")
    with_negatives = add_negative_samples_fast(flattened)
    print(f"  With negatives: {len(with_negatives):,} rows")
    print(f"    - Positive: {sum(with_negatives['passage_sign_de'] == 1):,}")
    print(f"    - Negative: {sum(with_negatives['passage_sign_de'] == 0):,}")
    
    # Step 6: Convert to triplets
    print("  🔄 Converting to triplets...")
    triplets = to_triplets(with_negatives, triplets_per_query=10)
    print(f"  Final triplets: {len(triplets):,}")
    print(f"  Unique queries: {triplets['query'].nunique()}")
    
    # Store result
    results[name] = triplets
    print(f"  ✅ {name.upper()} datasetcompleted!")

print("\n" + "="*50)
print("FINAL SUMMARY")
print("="*50)
total_triplets = 0
for name, triplets_df in results.items():
    triplets_count = len(triplets_df)
    total_triplets += triplets_count
    print(f"{name.upper()}: {triplets_count:,} triplets, {triplets_df['query'].nunique():,} unique queries")

print(f"\n🎯 TOTAL TRIPLETS: {total_triplets:,}")

print("\n🎯 Sample triplet from train dataset:")
if 'train' in results and len(results['train']) > 0:
    sample = results['train'].iloc[0]
    print(f"Query: {sample['query'][:80]}...")
    print(f"Positive: {sample['positive_example'][:80]}...")
    print(f"Negative: {sample['negative_example'][:80]}...")

print("\n✅ All datasets processed! Results stored in 'results' dictionary.")
print("Access with: results['train'], results['validation'], results['test']")


from backend.data_processing import convert_to_training_format

# Convert processed results to training format (no subsampling needed - already done!)
train_data = convert_to_training_format(results['train'])
val_data = convert_to_training_format(results['validation']) 
test_data = convert_to_training_format(results['test'])

# Print sample to verify format
print("Sample training triplet:")
print(f"Query: {train_data[0][0][:100]}...")
print(f"Positive: {train_data[0][1][:100]}...")  
print(f"Negative: {train_data[0][2][:100]}...")
print(f"\nDataset sizes:")
print(f"  Training: {len(train_data):,} triplets")
print(f"  Validation: {len(val_data):,} triplets")
print(f"  Test: {len(test_data):,} triplets")

# Use training data for the model
data = train_data

# Load pretrained tokenizer
tokenizer = PretrainedTokenizer(config['WORD_TO_IDX_PATH'])

# --- Training Setup ---
VOCAB_SIZE = tokenizer.vocab_size()

# Load pretrained embeddings
pretrained_embeddings = np.load(config['EMBEDDINGS_PATH'])
EMBED_DIM = pretrained_embeddings.shape[1]  # Get embedding dimension from loaded embeddings

print(f"Loaded pretrained embeddings: {pretrained_embeddings.shape}")
print(f"Vocabulary size: {VOCAB_SIZE}")
print(f"Embedding dimension: {EMBED_DIM}")

# Check if CUDA is available and set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize encoders with pretrained embeddings and move to GPU
query_encoder = RNNEncoder(VOCAB_SIZE, EMBED_DIM, config['HIDDEN_DIM'], pretrained_embeddings).to(device)
doc_encoder = RNNEncoder(VOCAB_SIZE, EMBED_DIM, config['HIDDEN_DIM'], pretrained_embeddings).to(device)

optimizer = torch.optim.Adam(list(query_encoder.parameters()) + list(doc_encoder.parameters()), lr=config['LR'])

# CRITICAL FIX: Increase batch size dramatically for much faster training
dataset = TripletDataset(data, tokenizer)
dataloader = DataLoader(dataset, batch_size=config['BATCH_SIZE'], shuffle=True, collate_fn=collate_fn, 
                       num_workers=2, pin_memory=True if device.type == 'cuda' else False)


# --- Training Loop ---
import time

print("🚀 Starting training...")
start_time = time.time()

for epoch in range(config['EPOCHS']):
    epoch_start = time.time()
    total_loss = 0
    num_batches = 0
    
    for query_batch, pos_batch, neg_batch in dataloader:
        # Move tensors to GPU
        query_batch = query_batch.to(device)
        pos_batch = pos_batch.to(device)
        neg_batch = neg_batch.to(device)
        
        q_vec = query_encoder(query_batch)
        pos_vec = doc_encoder(pos_batch)
        neg_vec = doc_encoder(neg_batch)

        loss = triplet_loss_function((q_vec, pos_vec, neg_vec), F.pairwise_distance, config['MARGIN'])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        
        # Progress indicator every 50 batches
        if num_batches % 50 == 0:
            print(f"  Batch {num_batches}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    epoch_time = time.time() - epoch_start
    avg_loss = total_loss / num_batches
    print(f"Epoch {epoch+1}/{config['EPOCHS']}, Avg Loss: {avg_loss:.4f}, Time: {epoch_time:.1f}s")

total_time = time.time() - start_time
print(f"\n✅ Training completed! Total time: {total_time/60:.1f} minutes")


# --- Automatic Model Saving ---
import os
import json
from datetime import datetime

# Create artifacts directory
artifacts_dir = "../artifacts"
os.makedirs(artifacts_dir, exist_ok=True)

# Create timestamped run directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = os.path.join(artifacts_dir, f"two_tower_run_{timestamp}")
os.makedirs(run_dir, exist_ok=True)

print(f"💾 Saving artifacts to: {run_dir}")

# Save model state dictionaries
torch.save({
    'query_encoder_state_dict': query_encoder.state_dict(),
    'doc_encoder_state_dict': doc_encoder.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': config['EPOCHS'],
    'final_loss': avg_loss
}, os.path.join(run_dir, 'model_checkpoint.pth'))

# Save model architectures (for easy loading later)
torch.save(query_encoder, os.path.join(run_dir, 'query_encoder_full.pth'))
torch.save(doc_encoder, os.path.join(run_dir, 'doc_encoder_full.pth'))

# Save training configuration
training_config = {
    'model_config': {
        'vocab_size': VOCAB_SIZE,
        'embed_dim': EMBED_DIM,
        'hidden_dim': config['HIDDEN_DIM'],
        'margin': config['MARGIN']
    },
    'training_config': {
        'epochs': config['EPOCHS'],
        'batch_size': config['BATCH_SIZE'],
        'learning_rate': config['LR'],
        'device': str(device)
    },
    'data_config': {
        'train_samples': len(train_data),
        'val_samples': len(val_data),
        'test_samples': len(test_data),
        'total_triplets': len(train_data) + len(val_data) + len(test_data)
    },
    'training_results': {
        'final_avg_loss': avg_loss
    }
}

with open(os.path.join(run_dir, 'training_config.json'), 'w') as f:
    json.dump(training_config, f, indent=2)

# Save tokenizer (copy the word2idx file or save the object)
import shutil
if os.path.exists(config['WORD_TO_IDX_PATH']):
    shutil.copy2(config['WORD_TO_IDX_PATH'], os.path.join(run_dir, 'word_to_idx.pkl'))

print(f"✅ Saved artifacts:")
print(f"  📁 Directory: {run_dir}")
print(f"  🧠 Models: model_checkpoint.pth, *_encoder_full.pth")
print(f"  ⚙️  Config: training_config.json")
print(f"  📝 Tokenizer: word_to_idx.pkl")
