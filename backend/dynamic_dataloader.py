import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import random


class DynamicTripletDataset(Dataset):
    """
    Dataset that creates triplets dynamically during training.
    Each query gets positive passages (score > 0) and random negative passages.
    """
    
    def __init__(self, parquet_path, triplets_per_query=10):
        """
        Args:
            parquet_path: Path to parquet file with columns ['query', 'passages.passage_text', 'passages.is_selected']
            triplets_per_query: Number of triplets to generate per query
        """
        print(f"Loading dataset from {parquet_path}...")
        self.df = pd.read_parquet(parquet_path, engine='fastparquet')
        
        # Filter valid data
        print(f"Original dataset size: {len(self.df)}")
        print(f"Columns: {list(self.df.columns)}")
        
        valid_mask = (
            self.df['query'].notna() & 
            self.df['passages.is_selected'].notna() &
            self.df['passages.is_selected'].apply(lambda x: any(x) if isinstance(x, list) else (x if isinstance(x, bool) else False))
        )
        
        # Add query_id check only if column exists
        if 'query_id' in self.df.columns:
            valid_mask = valid_mask & self.df['query_id'].notna()
            
        self.df = self.df[valid_mask].reset_index(drop=True)
        print(f"Filtered dataset size: {len(self.df)}")
        
        # Create a corpus of all passages for negative sampling
        self.all_passages = self.df['passages.passage_text'].explode().tolist()
        
        print(f"Loaded {len(self.df)} queries with {len(self.all_passages)} total passages")
        
        self.triplets_per_query = triplets_per_query
        
    def __len__(self):
        return len(self.df) * self.triplets_per_query
    
    def __getitem__(self, idx):
        # Get which query this triplet belongs to
        query_idx = idx // self.triplets_per_query
        row = self.df.iloc[query_idx]
        
        query = row['query']
        passages = row['passages.passage_text']
        scores = row['passages.is_selected']
        
        # Get positive passages (score > 0 or True)
        positive_passages = [p for p, s in zip(passages, scores) if s]
        
        # Sample a positive passage
        if positive_passages:
            positive = random.choice(positive_passages)
        else:
            # Fallback - shouldn't happen with filtering
            positive = random.choice(passages)
        
        # Sample a random negative passage from entire corpus
        negative = random.choice(self.all_passages)
        
        return {
            'query': query,
            'positive': positive,
            'negative': negative
        }

def create_dataloader(parquet_path, batch_size=32, triplets_per_query=10, shuffle=True, num_workers=0):
    """
    Create a DataLoader for dynamic triplet generation.
    
    Args:
        parquet_path: Path to parquet file
        batch_size: Batch size for training
        triplets_per_query: Number of triplets per query
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes
    
    Returns:
        DataLoader: PyTorch DataLoader
    """
    dataset = DynamicTripletDataset(parquet_path, triplets_per_query)
    
    def collate_fn(batch):
        """Custom collate function to handle batch of triplets"""
        queries = [item['query'] for item in batch]
        positives = [item['positive'] for item in batch]
        negatives = [item['negative'] for item in batch]
        
        return {
            'queries': queries,
            'positives': positives,
            'negatives': negatives
        }
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )