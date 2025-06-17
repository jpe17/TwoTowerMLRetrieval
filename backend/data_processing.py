import pandas as pd
import json
import numpy as np
import os

def filter_valid_data(df):
    """Filter out rows with missing required data"""
    valid_mask = (
        df['query'].notna() & 
        df['query_id'].notna() & 
        df['query_type'].notna() & 
        df['passages.is_selected'].notna() &
        df['passages.is_selected'].apply(lambda x: any(x) if isinstance(x, list) else False)
    )
    return df[valid_mask].copy()

def flatten_data_fast(df):
    """Convert nested passage data to flat rows - optimized version"""
    # Pre-allocate lists for better performance
    queries = []
    query_ids = []
    query_types = []
    documents = []
    passage_sign_ces = []
    
    # Vectorized processing
    for idx in df.index:
        row = df.loc[idx]
        passage_texts = row['passages.passage_text']
        is_selected = row['passages.is_selected']
        
        n_passages = len(passage_texts)
        
        # Extend lists in batch
        queries.extend([row['query']] * n_passages)
        query_ids.extend([row['query_id']] * n_passages)
        query_types.extend([row['query_type']] * n_passages)
        documents.extend(passage_texts)
        passage_sign_ces.extend(is_selected)
    
    # Create DataFrame in one go
    return pd.DataFrame({
        'query': queries,
        'query_id': query_ids,
        'query_type': query_types,
        'document': documents,
        'passage_sign_de': [1] * len(documents),
        'passage_sign_ce': passage_sign_ces,
    })

def add_negative_samples_vectorized(df):
    """Ultra-fast negative sampling with vectorized operations"""
    # Get unique queries and their data
    query_groups = df.groupby('query_id')
    all_docs = df['document'].values
    n_docs = len(all_docs)
    
    # Pre-generate all random indices at once
    n_queries = len(query_groups)
    neg_indices = np.random.choice(n_docs, size=(n_queries, 10), replace=True)
    
    result_dfs = []
    
    for input_id, (query_id, group) in enumerate(query_groups):
        # Positive samples (first 10)
        pos_samples = group.head(10).copy()
        pos_samples['input_id'] = input_id
        result_dfs.append(pos_samples)
        
        # Negative samples - vectorized creation
        first_row = pos_samples.iloc[0]
        neg_docs = all_docs[neg_indices[input_id]]
        
        # Create negative samples DataFrame more efficiently
        neg_data = {
            'query': np.full(10, first_row['query']),
            'query_id': np.full(10, query_id),
            'query_type': np.full(10, first_row['query_type']),
            'document': neg_docs,
            'passage_sign_de': np.zeros(10, dtype=int),
            'passage_sign_ce': [None] * 10,
            'input_id': np.full(10, input_id)
        }
        result_dfs.append(pd.DataFrame(neg_data))
    
    return pd.concat(result_dfs, ignore_index=True)

def to_triplets_fast(df, triplets_per_query=10):
    """
    Ultra-fast conversion to triplet format using vectorized operations.
    """
    # Group by query_id and get indices for positive/negative samples
    grouped = df.groupby('query_id')
    
    # Pre-allocate arrays for better performance
    total_triplets = len(grouped) * triplets_per_query
    queries = np.empty(total_triplets, dtype=object)
    positives = np.empty(total_triplets, dtype=object)
    negatives = np.empty(total_triplets, dtype=object)
    
    triplet_idx = 0
    
    for query_id, group in grouped:
        query_text = group['query'].iloc[0]
        
        # Get positive and negative documents as arrays
        pos_mask = group['passage_sign_de'] == 1
        neg_mask = group['passage_sign_de'] == 0
        
        pos_docs = group.loc[pos_mask, 'document'].values
        neg_docs = group.loc[neg_mask, 'document'].values
        
        if len(pos_docs) == 0 or len(neg_docs) == 0:
            continue
        
        # Vectorized random sampling
        pos_indices = np.random.choice(len(pos_docs), size=triplets_per_query, replace=True)
        neg_indices = np.random.choice(len(neg_docs), size=triplets_per_query, replace=True)
        
        # Fill arrays
        end_idx = triplet_idx + triplets_per_query
        queries[triplet_idx:end_idx] = query_text
        positives[triplet_idx:end_idx] = pos_docs[pos_indices]
        negatives[triplet_idx:end_idx] = neg_docs[neg_indices]
        
        triplet_idx = end_idx
    
    # Trim arrays to actual size
    queries = queries[:triplet_idx]
    positives = positives[:triplet_idx]
    negatives = negatives[:triplet_idx]
    
    return pd.DataFrame({
        'query': queries,
        'positive_example': positives,
        'negative_example': negatives
    })

def convert_to_triplets_fast(df, max_triplets_per_query=None):
    """
    Fast conversion to triplet format with optional limit per query.
    """
    if max_triplets_per_query is None:
        # Use balanced approach with reasonable default
        return to_triplets_fast(df, triplets_per_query=100)
    else:
        return to_triplets_fast(df, triplets_per_query=max_triplets_per_query)

def convert_to_triplets_balanced(df, triplets_per_query=10):
    """
    Convert to triplets with balanced sampling - optimized version.
    """
    return to_triplets_fast(df, triplets_per_query=triplets_per_query)

def process_single_dataset_fast(input_path, output_path, dataset_name):
    """Process a single dataset with optimized functions"""
    print(f"Processing {dataset_name}...")
    print(f"Loading: {input_path}")
    
    # Load data
    df = pd.read_parquet(input_path, engine='fastparquet')
    print(f"Loaded: {len(df):,} samples")
    
    # Fast filtering
    valid_mask = (
        df['query'].notna() & 
        df['query_id'].notna() & 
        df['query_type'].notna() & 
        df['passages.is_selected'].notna() &
        df['passages.is_selected'].apply(lambda x: any(x) if isinstance(x, list) else False)
    )
    df_filtered = df[valid_mask].copy()
    print(f"After filtering: {len(df_filtered):,} samples")
    
    # Process with optimized functions
    flattened = flatten_data_fast(df_filtered)
    final_df = add_negative_samples_vectorized(flattened)
    
    # Save
    print(f"Saving: {output_path}")
    final_df.to_parquet(output_path, engine='fastparquet', index=False)
    print(f"Final dataset: {len(final_df):,} samples")
    print("-" * 50)
    
    return final_df

# Keep original functions for backward compatibility
def flatten_data(df):
    """Convert nested passage data to flat rows"""
    return flatten_data_fast(df)

def add_negative_samples_fast(df):
    """Fast negative sampling with vectorized operations"""
    return add_negative_samples_vectorized(df)

def convert_to_triplets(df, max_triplets_per_query=None):
    """Convert the processed dataset to triplet format for two-tower model training."""
    return convert_to_triplets_fast(df, max_triplets_per_query)

def process_single_dataset(input_path, output_path, dataset_name):
    """Process a single dataset efficiently"""
    return process_single_dataset_fast(input_path, output_path, dataset_name)

def to_triplets(df, triplets_per_query=10):
    """Simple conversion to triplet format for two-tower training."""
    return to_triplets_fast(df, triplets_per_query)

# Convert processed results to training format
def convert_to_training_format(triplets_df, max_samples=None):
    """
    Convert DataFrame with columns ['query', 'positive_example', 'negative_example'] 
    to list of tuples format: [(query, positive, negative), ...] - optimized version
    """
    if max_samples:
        triplets_df = triplets_df.head(max_samples)
    
    # Vectorized conversion using .values for speed
    queries = triplets_df['query'].values
    positives = triplets_df['positive_example'].values
    negatives = triplets_df['negative_example'].values
    
    # Create list of tuples in one go
    training_data = list(zip(queries, positives, negatives))
    
    return training_data

def process_and_export_to_csv(input_parquet_path, output_csv_path, dataset_name="dataset"):
    """
    Simple function to process data and export pre-triplets dataset to CSV for inspection.
    
    Args:
        input_parquet_path: Path to input parquet file
        output_csv_path: Path for output CSV file  
        dataset_name: Name for progress messages
    
    Returns:
        DataFrame: The processed dataset
    """
    print(f"Processing {dataset_name}...")
    print(f"Loading: {input_parquet_path}")
    
    # Load data
    df = pd.read_parquet(input_parquet_path, engine='fastparquet')
    print(f"Loaded: {len(df):,} samples")
    
    # Fast filtering
    valid_mask = (
        df['query'].notna() & 
        df['query_id'].notna() & 
        df['query_type'].notna() & 
        df['passages.is_selected'].notna() &
        df['passages.is_selected'].apply(lambda x: any(x) if isinstance(x, list) else False)
    )
    df_filtered = df[valid_mask].copy()
    print(f"After filtering: {len(df_filtered):,} samples")
    
    # Process with optimized functions
    flattened = flatten_data_fast(df_filtered)
    final_df = add_negative_samples_vectorized(flattened)
    
    print(f"Final processed dataset: {len(final_df):,} samples")
    
    # Export to CSV
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    final_df.to_csv(output_csv_path, index=False)
    print(f"Exported to CSV: {output_csv_path}")
    
    return final_df