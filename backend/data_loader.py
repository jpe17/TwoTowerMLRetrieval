import pandas as pd
import fastparquet
import random
import numpy as np
from typing import List, Tuple, Dict, Optional


class DataLoader:
    """Handles loading and preprocessing of MS MARCO parquet files."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.num_triplets_per_query = config.get('NUM_TRIPLETS_PER_QUERY', 1)
    
    def load_and_process_parquet(self, path: str, subsample_ratio: Optional[float] = None) -> List[Tuple[str, str, str]]:
        """
        Load a parquet file and create triplets with correct positive/negative logic.
        
        Args:
            path: Path to the parquet file
            subsample_ratio: Optional ratio to subsample the data (0.0 < ratio <= 1.0)
            
        Returns:
            List of triplets (query, positive_doc, negative_doc)
        """
        print(f"\nProcessing {path}...")
        df = pd.read_parquet(path, engine='fastparquet')
        
        # Apply subsampling if specified
        if subsample_ratio is not None and 0 < subsample_ratio < 1.0:
            original_size = len(df)
            df = df.sample(frac=subsample_ratio, random_state=42).reset_index(drop=True)
            print(f"  Subsampled from {original_size:,} to {len(df):,} queries (ratio: {subsample_ratio})")

        # Filter for valid rows: keep queries that have at least one passage
        valid_mask = (
            df['query'].notna() & 
            df['passages.passage_text'].notna() &
            df['passages.passage_text'].apply(lambda x: len(x) > 0 if isinstance(x, list) else False)
        )
        df = df[valid_mask].reset_index(drop=True)
        print(f"  Found {len(df):,} valid queries after filtering.")

        # Create a flat list of (query_id, passage) for negative sampling
        all_passages = []
        for idx, row in df.iterrows():
            for p in row['passages.passage_text']:
                all_passages.append((idx, p))  # tag with query index for filtering later

        triplets = []
        for idx, row in df.iterrows():
            query = row['query']
            query_passages = row['passages.passage_text']

            for _ in range(self.num_triplets_per_query): 
                positive = random.choice(query_passages)

                # Negative must be from a *different* query
                while True:
                    neg_query_id, negative = random.choice(all_passages)
                    if neg_query_id != idx:
                        break

                triplets.append((query, positive, negative))

        print(f"  Generated {len(triplets):,} triplets.")
        return triplets
    
    def load_datasets(self, subsample_ratio: Optional[float] = None) -> Dict[str, List[Tuple[str, str, str]]]:
        """
        Load all datasets (train, validation, test).
        
        Args:
            subsample_ratio: Optional ratio to subsample each dataset
            
        Returns:
            Dictionary with 'train', 'validation', 'test' keys containing triplet lists
        """
        datasets = {}
        
        dataset_paths = {
            'train': self.config['TRAIN_DATASET_PATH'],
            'validation': self.config['VAL_DATASET_PATH'],
            'test': self.config['TEST_DATASET_PATH']
        }
        
        for split, path in dataset_paths.items():
            try:
                datasets[split] = self.load_and_process_parquet(path, subsample_ratio)
            except Exception as e:
                print(f"❌ Error loading {split} dataset from {path}: {str(e)}")
                datasets[split] = []
        
        return datasets
    
    def get_dataset_stats(self, datasets: Dict[str, List[Tuple[str, str, str]]]) -> Dict[str, int]:
        """Get statistics about the loaded datasets."""
        stats = {}
        total = 0
        
        for split, data in datasets.items():
            count = len(data)
            stats[split] = count
            total += count
            
        stats['total'] = total
        return stats 