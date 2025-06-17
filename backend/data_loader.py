import pandas as pd
import fastparquet
import random
import csv
from typing import List, Tuple, Dict, Optional


class DataLoader:
    """Handles loading and preprocessing of MS MARCO parquet files."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.num_triplets_per_query = config.get('NUM_TRIPLETS_PER_QUERY', 1)
    
    def load_and_process_parquet(self, path: str, subsample_ratio: Optional[float] = None) -> List[Tuple[str, str, str]]:
        """Load parquet file and create triplets (query, positive, negative)."""
        print(f"\nProcessing {path}...")
        df = pd.read_parquet(path, engine='fastparquet')
        
        # Apply subsampling if specified
        if subsample_ratio and 0 < subsample_ratio < 1.0:
            original_size = len(df)
            df = df.sample(frac=subsample_ratio, random_state=42).reset_index(drop=True)
            print(f"  Subsampled from {original_size:,} to {len(df):,} queries")

        # Filter valid rows with non-empty passages
        valid_mask = (df['query'].notna() & 
                     df['passages.passage_text'].notna() &
                     df['passages.passage_text'].apply(lambda x: len(x) > 0 if isinstance(x, list) else False))
        df = df[valid_mask].reset_index(drop=True)
        print(f"  Found {len(df):,} valid queries after filtering.")

        # Create passage pool for negative sampling
        all_passages = [(idx, p) for idx, row in df.iterrows() 
                       for p in row['passages.passage_text']]

        # Generate triplets
        triplets = []
        rng = random.Random(42)
        for idx, row in df.iterrows():
            query = row['query']
            passages = row['passages.passage_text']
            if not passages:
                continue
                
            # Sample positive passages
            num_pos = min(self.num_triplets_per_query, len(passages))
            pos_indices = random.Random(42).sample(range(len(passages)), num_pos)
            
            for i in pos_indices:
                positive = passages[i]
                # Sample negative from different query
                while True:
                    neg_query_id, negative = rng.choice(all_passages)
                    if neg_query_id != idx:
                        break
                triplets.append((query, positive, negative))

        print(f"  Generated {len(triplets):,} triplets.")
        return triplets
    
    def load_datasets(self, subsample_ratio: Optional[float] = None) -> Dict[str, List[Tuple[str, str, str]]]:
        """Load train, validation, and test datasets."""
        datasets = {}
        paths = {
            'train': self.config['TRAIN_DATASET_PATH'],
            'validation': self.config['VAL_DATASET_PATH'],
            'test': self.config['TEST_DATASET_PATH']
        }
        
        for split, path in paths.items():
            try:
                datasets[split] = self.load_and_process_parquet(path, subsample_ratio)
                self.export_triplets(datasets[split], 'data/triplets_sample.tsv')
            except Exception as e:
                print(f"❌ Error loading {split} dataset: {str(e)}")
                datasets[split] = []
        
        return datasets
    
    def get_dataset_stats(self, datasets: Dict[str, List[Tuple[str, str, str]]]) -> Dict[str, int]:
        """Get dataset statistics."""
        stats = {split: len(data) for split, data in datasets.items()}
        stats['total'] = sum(stats.values())
        return stats
    
    def export_triplets(self, triplets: List[Tuple[str, str, str]], output_path: str):
        """Export triplets to TSV file."""
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['query', 'positive', 'negative'])
            writer.writerows(triplets)
        print(f"Exported {len(triplets)} triplets to {output_path}")
