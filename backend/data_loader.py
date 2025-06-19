import pandas as pd
import fastparquet
import random
from typing import List, Tuple, Dict, Optional


class DataLoader:
    """Handles loading and preprocessing of MS MARCO parquet files for retrieval."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.num_triplets_per_query = config.get('NUM_TRIPLETS_PER_QUERY', 1)
        self.training_mode = config.get('TRAINING_MODE', 'retrieval')  # 'retrieval' or 'ranking'
    
    def load_and_process_parquet(self, path: str, subsample_ratio: Optional[float] = None) -> List[Tuple[str, str, str]]:
        """Load parquet file and create triplets (query, positive, negative) for retrieval task."""
        print(f"\n🔍 Processing {path} for {self.training_mode} task...")
        df = pd.read_parquet(path, engine='fastparquet')
        
        if subsample_ratio and 0 < subsample_ratio < 1.0:
            original_size = len(df)
            if 'train' in path:
                seed = 42
            elif 'validation' in path:
                seed = 123
            else:
                seed = 456 # For test or others
            df = df.sample(frac=subsample_ratio, random_state=seed).reset_index(drop=True)
            print(f"  Subsampled from {original_size:,} to {len(df):,} queries (seed={seed})")

        valid_mask = (df['query'].notna() & 
                     df['passages.passage_text'].notna() &
                     df['passages.passage_text'].apply(lambda x: len(x) > 0 if isinstance(x, list) else False))
        df = df[valid_mask].reset_index(drop=True)
        print(f"  Found {len(df):,} valid queries after filtering.")

        # Collect all passages for random negatives
        all_passages = [(idx, p) for idx, row in df.iterrows() 
                       for p in row['passages.passage_text']]

        triplets = []
        if 'train' in path:
            seed = 42
        elif 'validation' in path:
            seed = 123
        else:
            seed = 456
        rng = random.Random(seed)
        
        for idx, row in df.iterrows():
            query = row['query']
            passages = row['passages.passage_text']
            
            if not passages:
                continue
            
            if self.training_mode == 'retrieval':
                # RETRIEVAL MODE: All passages are treated as positives
                # This helps the model learn general relevance
                num_pos = min(self.num_triplets_per_query, len(passages))
                pos_indices = random.Random(seed + idx).sample(range(len(passages)), num_pos)
                
                for i in pos_indices:
                    positive = passages[i]
                    # Use random negative from other queries
                    while True:
                        neg_query_id, negative = rng.choice(all_passages)
                        if neg_query_id != idx:
                            break
                    triplets.append((query, positive, negative))
                    
            else:  # ranking mode
                # RANKING MODE: Only clicked passages (is_selected=1) are positives
                # Non-clicked passages from same query become negatives
                is_selected = row.get('passages.is_selected', [])
                if not is_selected or len(passages) != len(is_selected):
                    continue
                    
                positive_indices = [i for i, selected in enumerate(is_selected) if selected == 1]
                negative_indices = [i for i, selected in enumerate(is_selected) if selected == 0]
                
                if not positive_indices:
                    continue
                    
                for pos_idx in positive_indices:
                    positive = passages[pos_idx]
                    
                    # Use negatives from same query (harder negatives for ranking)
                    if negative_indices:
                        neg_idx = rng.choice(negative_indices)
                        negative = passages[neg_idx]
                    else:
                        # Fallback to random negative
                        while True:
                            neg_query_id, negative = rng.choice(all_passages)
                            if neg_query_id != idx:
                                break
                    
                    triplets.append((query, positive, negative))

        print(f"  Generated {len(triplets):,} triplets for {self.training_mode} training.")
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
            except Exception as e:
                print(f"❌ Error loading {split} dataset: {str(e)}")
                datasets[split] = []
        
        return datasets
