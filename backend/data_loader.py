import pandas as pd
import fastparquet
import random
import csv
from typing import List, Tuple, Dict, Optional


class DataLoader:
    """Handles loading and preprocessing of MS MARCO parquet files for both retrieval and ranking tasks."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.num_triplets_per_query = config.get('NUM_TRIPLETS_PER_QUERY', 1)
        self.mode = config.get('TASK_MODE', 'retrieval')  # 'retrieval' or 'ranking'
    
    def load_and_process_parquet(self, path: str, subsample_ratio: Optional[float] = None) -> List[Tuple[str, str, str]]:
        """Load parquet file and create triplets based on the configured mode."""
        if self.mode == 'ranking':
            return self._load_for_ranking(path, subsample_ratio)
        else:
            return self._load_for_retrieval(path, subsample_ratio)
    
    def _load_for_retrieval(self, path: str, subsample_ratio: Optional[float] = None) -> List[Tuple[str, str, str]]:
        """Load parquet file and create triplets (query, positive, negative) for retrieval task."""
        print(f"\n🔍 Processing {path} for retrieval task...")
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
    
    def _load_for_ranking(self, path: str, subsample_ratio: Optional[float] = None) -> List[Tuple[str, str, str]]:
        """Load parquet file and create ranking triplets (query, selected_passage, non_selected_passage)."""
        print(f"\n🎯 Processing {path} for ranking task...")
        df = pd.read_parquet(path, engine='fastparquet')
        
        # Apply subsampling if specified
        if subsample_ratio and 0 < subsample_ratio < 1.0:
            original_size = len(df)
            df = df.sample(frac=subsample_ratio, random_state=42).reset_index(drop=True)
            print(f"  Subsampled from {original_size:,} to {len(df):,} queries")

        # Filter valid rows with passages and is_selected
        valid_mask = (df['query'].notna() & 
                     df['passages.passage_text'].notna() &
                     df['passages.is_selected'].notna() &
                     df['passages.passage_text'].apply(lambda x: len(x) > 0 if isinstance(x, list) else False) &
                     df['passages.is_selected'].apply(lambda x: len(x) > 0 if isinstance(x, list) else False))
        df = df[valid_mask].reset_index(drop=True)
        print(f"  Found {len(df):,} valid queries after filtering.")

        # Generate ranking triplets
        triplets = []
        rng = random.Random(42)
        skipped_queries = 0
        
        for idx, row in df.iterrows():
            query = row['query']
            passages = row['passages.passage_text']
            is_selected = row['passages.is_selected']
            
            if not passages or not is_selected or len(passages) != len(is_selected):
                skipped_queries += 1
                continue
            
            # Find selected passages (where is_selected = 1)
            selected_indices = [i for i, sel in enumerate(is_selected) if sel == 1]
            non_selected_indices = [i for i, sel in enumerate(is_selected) if sel == 0]
            
            if not selected_indices or not non_selected_indices:
                skipped_queries += 1
                continue
            
            # Create triplets: each selected passage vs random non-selected passages
            for selected_idx in selected_indices:
                positive_passage = passages[selected_idx]
                
                # Sample negative passages from non-selected
                num_negatives = min(self.num_triplets_per_query, len(non_selected_indices))
                negative_indices = rng.sample(non_selected_indices, num_negatives)
                
                for neg_idx in negative_indices:
                    negative_passage = passages[neg_idx]
                    triplets.append((query, positive_passage, negative_passage))

        print(f"  Generated {len(triplets):,} ranking triplets.")
        print(f"  Skipped {skipped_queries:,} queries (no selected or no non-selected passages).")
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
                # Export sample for inspection
                if datasets[split]:
                    self.export_triplets(datasets[split][:100], f'data/{self.mode}_triplets_{split}_sample.tsv')
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
        headers = {
            'retrieval': ['query', 'positive', 'negative'],
            'ranking': ['query', 'selected_passage', 'non_selected_passage']
        }
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(headers[self.mode])
            writer.writerows(triplets)
        print(f"📁 Exported {len(triplets)} {self.mode} triplets to {output_path}")
