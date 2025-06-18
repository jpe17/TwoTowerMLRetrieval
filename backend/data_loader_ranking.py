import pandas as pd
import fastparquet
import random
import csv
from typing import List, Tuple, Dict, Optional


class RankingDataLoader:
    """Handles loading and preprocessing for ranking task using is_selected."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.num_triplets_per_query = config.get('NUM_TRIPLETS_PER_QUERY', 1)
    
    def load_and_process_parquet_ranking(self, path: str, subsample_ratio: Optional[float] = None) -> List[Tuple[str, str, str]]:
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
    
    def load_ranking_datasets(self, subsample_ratio: Optional[float] = None) -> Dict[str, List[Tuple[str, str, str]]]:
        """Load train, validation, and test datasets for ranking."""
        datasets = {}
        paths = {
            'train': self.config['TRAIN_DATASET_PATH'],
            'validation': self.config['VAL_DATASET_PATH'],
            'test': self.config['TEST_DATASET_PATH']
        }
        
        for split, path in paths.items():
            try:
                datasets[split] = self.load_and_process_parquet_ranking(path, subsample_ratio)
                # Export sample for inspection
                if datasets[split]:
                    self.export_triplets(datasets[split][:100], f'data/ranking_triplets_{split}_sample.tsv')
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
            writer.writerow(['query', 'selected_passage', 'non_selected_passage'])
            writer.writerows(triplets)
        print(f"📁 Exported {len(triplets)} ranking triplets to {output_path}")


# Backwards compatibility - keep original DataLoader unchanged
from data_loader import DataLoader  # Import original for retrieval tasks 