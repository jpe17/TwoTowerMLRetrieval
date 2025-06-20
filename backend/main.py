#!/usr/bin/env python3
"""
Simplified Two-Tower ML Training and Evaluation Script
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import json
import time
from typing import List, Tuple, Dict
from pathlib import Path
import sys
import os
import gc
import argparse
import pickle
import shutil
from sklearn.feature_extraction.text import TfidfVectorizer
import wandb

# Add backend to path for imports
sys.path.append(str(Path(__file__).parent))

from data_loader import DataLoader as RetrievalDataLoader
from tokenizer import PretrainedTokenizer
from model import TwoTowerModel, triplet_loss_cosine
from evaluators import BatchEvaluator, CorpusEvaluator, TestEvaluator

# --- DATASET AND COLLATE FUNCTION ---

class TripletDataset(Dataset):
    """Dataset for (query, positive_doc, negative_doc) triplets."""
    def __init__(self, data: List[Tuple[str, str, str]], tokenizer: PretrainedTokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query, pos_doc, neg_doc = self.data[idx]
        return (
            torch.tensor(self.tokenizer.encode(query), dtype=torch.long),
            torch.tensor(self.tokenizer.encode(pos_doc), dtype=torch.long),
            torch.tensor(self.tokenizer.encode(neg_doc), dtype=torch.long)
        )

def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pads sequences to the max length in a batch."""
    queries, pos_docs, neg_docs = zip(*batch)
    queries_padded = pad_sequence(queries, batch_first=True, padding_value=0)
    pos_docs_padded = pad_sequence(pos_docs, batch_first=True, padding_value=0)
    neg_docs_padded = pad_sequence(neg_docs, batch_first=True, padding_value=0)
    return queries_padded, pos_docs_padded, neg_docs_padded

# --- HELPER FUNCTIONS ---

def get_best_device() -> torch.device:
    """Gets the best available device."""
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

def clean_memory():
    """Aggressively cleans memory for MPS/CUDA devices."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

def load_config(path: str) -> Dict:
    """Loads a JSON config file."""
    with open(path, 'r') as f:
        return json.load(f)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Simplified Two-Tower ML Training & Evaluation')
    parser.add_argument(
        '--model_path', '-m',
        type=str,
        help='Path to a saved model state_dict (.pth) to run evaluation on, skipping training.'
    )
    return parser.parse_args()

# --- ARTIFACT SAVING ---
def save_inference_artifacts(output_dir: Path, model: TwoTowerModel, config: Dict, tokenizer: PretrainedTokenizer, datasets: Dict):
    """Saves all artifacts required for inference and indexing to a specified directory."""
    print(f"\n💾 Saving inference artifacts to: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Save model state dict
    torch.save(model.state_dict(), output_dir / "model.pth")
    
    # 2. Save config
    # Add runtime-determined values to the config before saving
    config_to_save = config.copy()
    config_to_save['VOCAB_SIZE'] = tokenizer.vocab_size()
    config_to_save['EMBED_DIM'] = model.query_encoder.embedding.embedding_dim
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config_to_save, f, indent=4)
        
    # 3. Save tokenizer data
    shutil.copyfile(config['WORD_TO_IDX_PATH'], output_dir / 'word_to_idx.pkl')
    
    # 4. Generate and save document embeddings
    print("  Generating and saving document embeddings for indexing...")
    model.eval()
    
    # Collect unique documents from all datasets
    all_docs = set()
    for split_data in datasets.values():
        for _, pos_doc, neg_doc in split_data:
            all_docs.add(pos_doc)
            all_docs.add(neg_doc)
    
    unique_docs = list(all_docs)
    doc_embeddings = []
    
    with torch.no_grad():
        for i in range(0, len(unique_docs), config.get('BATCH_SIZE', 64)):
            batch_docs = unique_docs[i:i+config.get('BATCH_SIZE', 64)]
            batch_tokens = [torch.tensor(tokenizer.encode(doc), dtype=torch.long) for doc in batch_docs]
            padded_batch = pad_sequence(batch_tokens, batch_first=True, padding_value=0).to(model.device)
            embeddings = model.encode_document(padded_batch)
            doc_embeddings.append(embeddings.cpu().numpy())
            
    all_doc_embeddings = np.vstack(doc_embeddings)
    
    # Save documents and their embeddings
    with open(output_dir / 'documents.pkl', 'wb') as f:
        pickle.dump(unique_docs, f)
    np.save(output_dir / 'document_embeddings.npy', all_doc_embeddings)
    
    # 5. Create and save TF-IDF model and document matrix
    print("  Creating and saving TF-IDF model...")
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=20000)
    doc_tfidf_matrix = tfidf_vectorizer.fit_transform(unique_docs)
    
    with open(output_dir / 'tfidf_artifacts.pkl', 'wb') as f:
        pickle.dump({
            'vectorizer': tfidf_vectorizer,
            'matrix': doc_tfidf_matrix
        }, f)

    print(f"  ✅ Saved {len(unique_docs)} documents and their {all_doc_embeddings.shape} embeddings.")
    print("  ✅ Saved TF-IDF vectorizer and document matrix.")
    print(f"✅ Artifacts ready for frontend use.")


# --- MAIN ---

def main():
    """Main function to run the training and evaluation pipeline."""
    # --- ARGUMENT PARSING ---
    args = parse_args()

    # --- CONFIGURATION ---
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
    print("📋 Loading configuration...")
    config = load_config('backend/config.json')
    device = get_best_device()
    print(f"✅ Using device: {device}")
    
    # --- TOKENIZER & EMBEDDINGS ---
    print("\n📝 Loading tokenizer and pretrained embeddings...")
    tokenizer = PretrainedTokenizer(config['WORD_TO_IDX_PATH'])
    pretrained_embeddings = np.load(config['EMBEDDINGS_PATH'])
    
    # Handle <UNK> token embedding
    if tokenizer.vocab_size() > len(pretrained_embeddings):
        print("Mismatch between vocab size and embeddings. Adding vector for <UNK> token.")
        embed_dim = pretrained_embeddings.shape[1]
        # Use a small random vector for the <UNK> token
        unk_embedding = np.random.rand(1, embed_dim).astype(np.float32) * 0.1
        pretrained_embeddings = np.vstack([pretrained_embeddings, unk_embedding])
        print(f"New embedding shape: {pretrained_embeddings.shape}")

    config['VOCAB_SIZE'] = tokenizer.vocab_size()
    config['EMBED_DIM'] = pretrained_embeddings.shape[1]

    # --- DATA LOADING ---
    print("\n📚 Loading datasets...")
    data_loader = RetrievalDataLoader(config)
    datasets = data_loader.load_datasets(subsample_ratio=config.get('SUBSAMPLE_RATIO'))

    # --- MODEL CREATION ---
    print("\n🏗️  Creating model...")
    model = TwoTowerModel(config, pretrained_embeddings).to(device)
    # Add device to model for later access
    model.device = device

    if args.model_path:
        print(f"📥 Loading model for evaluation from: {args.model_path}")
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    else:
        # --- TRAINING & VALIDATION ---
        train_dataset = TripletDataset(datasets['train'], tokenizer)
        val_dataset = TripletDataset(datasets['validation'], tokenizer)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.get('BATCH_SIZE', 64),
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=2
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.get('BATCH_SIZE', 64),
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=1
        )
        print(f"  Train samples: {len(train_dataset):,}, Val samples: {len(val_dataset):,}")

        optimizer = torch.optim.Adam(model.parameters(), lr=config.get('LR', 1e-4))
        
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Trainable parameters: {total_params:,}")

        # Initialize W&B
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        wandb.init(
            project="two-tower-retrieval-last",
            config=config,
            name=f"run-{timestamp}"
        )
        wandb.watch(model, log_freq=50)

        print("\n🚀 Starting training...")
        start_time = time.time()
        global_step = 0
        
        for epoch in range(config.get('EPOCHS', 3)):
            model.train()
            total_loss = 0
            
            for i, (queries, pos_docs, neg_docs) in enumerate(train_loader):
                queries, pos_docs, neg_docs = queries.to(device), pos_docs.to(device), neg_docs.to(device)
                
                optimizer.zero_grad()
                
                query_emb = model.encode_query(queries)
                pos_emb = model.encode_document(pos_docs)
                neg_emb = model.encode_document(neg_docs)
                
                loss = triplet_loss_cosine((query_emb, pos_emb, neg_emb), margin=config.get('MARGIN', 0.2))
                loss.backward()
                
                # Gradient clipping for RNN stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                total_loss += loss.item()
                global_step += 1
                
                if (i + 1) % 50 == 0:
                    print(f"   Epoch {epoch+1}/{config.get('EPOCHS', 3)}, Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
                    wandb.log({"train_loss_batch": loss.item()}, step=global_step)
                
                if (i + 1) % 100 == 0:
                    clean_memory()
            
            avg_train_loss = total_loss / len(train_loader)
            print(f"✅ Epoch {epoch+1} Summary | Avg Train Loss: {avg_train_loss:.4f}")
            
            # Use both evaluation methods for comparison
            batch_evaluator = BatchEvaluator()
            corpus_evaluator = CorpusEvaluator()
            
            metrics, avg_val_loss = batch_evaluator.evaluate(model, val_loader, device, config)
            full_corpus_metrics = corpus_evaluator.evaluate(model, datasets['validation'], tokenizer, device)
            
            print(f"📊 Batch-wise Validation | Avg Val Loss: {avg_val_loss:.4f} | Metrics: {metrics}")
            print(f"🎯 Full-corpus Validation | Metrics: {full_corpus_metrics}")

            # Log epoch-level metrics to W&B
            log_data = {
                "epoch": epoch + 1,
                "avg_train_loss": avg_train_loss,
                "avg_val_loss": avg_val_loss,
            }
            # Add batch-wise metrics with prefix
            for k, v in metrics.items():
                log_data[f"batch_{k}"] = v
            # Add full-corpus metrics with prefix  
            for k, v in full_corpus_metrics.items():
                log_data[f"corpus_{k}"] = v
            wandb.log(log_data, step=global_step)

            clean_memory()

        print(f"\n🎉 Training finished in {(time.time() - start_time)/60:.2f} minutes.")

        # --- SAVE ARTIFACTS ---
        output_dir = Path(f"artifacts/{wandb.run.name}")
        save_inference_artifacts(output_dir, model, config, tokenizer, datasets)
        
        wandb.finish()

    # --- TEST EVALUATION ---
    if datasets.get('test'):
        test_evaluator = TestEvaluator()
        test_evaluator.evaluate(model, datasets['test'], tokenizer, device)
    else:
        print("\nNo test data found. Skipping test evaluation.")

if __name__ == "__main__":
    main() 