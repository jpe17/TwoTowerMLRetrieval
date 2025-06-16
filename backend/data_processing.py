import pandas as pd
import json
import numpy as np

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

def flatten_data(df):
    """Convert nested passage data to flat rows"""
    rows = []
    for _, row in df.iterrows():
        for i, passage_text in enumerate(row['passages.passage_text']):
            rows.append({
                'query': row['query'],
                'query_id': row['query_id'], 
                'query_type': row['query_type'],
                'document': passage_text,
                'passage_sign_de': 1,
                'passage_sign_ce': row['passages.is_selected'][i],
            })
    return pd.DataFrame(rows)

def add_negative_samples_fast(df):
    """Fast negative sampling with vectorized operations"""
    result_rows = []
    input_id = 0
    
    # Pre-sample all documents for faster negative sampling
    all_docs = df['document'].values
    n_docs = len(all_docs)
    
    for query_id, group in df.groupby('query_id'):
        # Positive samples (first 10)
        pos_samples = group.head(10).copy()
        pos_samples['input_id'] = input_id
        result_rows.append(pos_samples)
        
        # Fast negative sampling - sample 10 at once
        first_row = pos_samples.iloc[0]
        neg_indices = np.random.choice(n_docs, size=10, replace=True)
        neg_docs = all_docs[neg_indices]
        
        # Create negative samples dataframe directly
        neg_df = pd.DataFrame({
            'query': [first_row['query']] * 10,
            'query_id': [query_id] * 10,
            'query_type': [first_row['query_type']] * 10,
            'document': neg_docs,
            'passage_sign_de': [0] * 10,
            'passage_sign_ce': [None] * 10,
            'input_id': [input_id] * 10
        })
        result_rows.append(neg_df)
        input_id += 1
    
    return pd.concat(result_rows, ignore_index=True)

def convert_to_triplets(df, max_triplets_per_query=None):
    """
    Convert the processed dataset to triplet format for two-tower model training.
    
    Args:
        df: DataFrame with columns ['query', 'query_id', 'document', 'passage_sign_de']
        max_triplets_per_query: Maximum number of triplets to generate per query (None for all combinations)
    
    Returns:
        DataFrame with columns ['query', 'positive_example', 'negative_example']
    """
    triplets = []
    
    for query_id, group in df.groupby('query_id'):
        # Get query text (should be same for all rows in group)
        query_text = group['query'].iloc[0]
        
        # Separate positive and negative documents
        positive_docs = group[group['passage_sign_de'] == 1]['document'].tolist()
        negative_docs = group[group['passage_sign_de'] == 0]['document'].tolist()
        
        # Skip if no positives or negatives
        if not positive_docs or not negative_docs:
            continue
        
        # Create all combinations of positive-negative pairs
        combinations = []
        for pos_doc in positive_docs:
            for neg_doc in negative_docs:
                combinations.append({
                    'query': query_text,
                    'positive_example': pos_doc,
                    'negative_example': neg_doc
                })
        
        # Limit triplets per query if specified
        if max_triplets_per_query and len(combinations) > max_triplets_per_query:
            combinations = np.random.choice(combinations, size=max_triplets_per_query, replace=False).tolist()
        
        triplets.extend(combinations)
    
    return pd.DataFrame(triplets)

def convert_to_triplets_balanced(df, triplets_per_query=10):
    """
    Convert to triplets with balanced sampling - equal number of triplets per query.
    
    Args:
        df: DataFrame with columns ['query', 'query_id', 'document', 'passage_sign_de']
        triplets_per_query: Number of triplets to generate per query
    
    Returns:
        DataFrame with columns ['query', 'positive_example', 'negative_example']
    """
    triplets = []
    
    for query_id, group in df.groupby('query_id'):
        query_text = group['query'].iloc[0]
        
        positive_docs = group[group['passage_sign_de'] == 1]['document'].tolist()
        negative_docs = group[group['passage_sign_de'] == 0]['document'].tolist()
        
        if not positive_docs or not negative_docs:
            continue
        
        # Sample triplets_per_query combinations
        for _ in range(triplets_per_query):
            pos_doc = np.random.choice(positive_docs)
            neg_doc = np.random.choice(negative_docs)
            
            triplets.append({
                'query': query_text,
                'positive_example': pos_doc,
                'negative_example': neg_doc
            })
    
    return pd.DataFrame(triplets)

def process_single_dataset(input_path, output_path, dataset_name):
    """Process a single dataset efficiently"""
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
    
    # Process
    flattened = flatten_data(df_filtered)
    final_df = add_negative_samples_fast(flattened)
    
    # Save
    print(f"Saving: {output_path}")
    final_df.to_parquet(output_path, engine='fastparquet', index=False)
    print(f"Final dataset: {len(final_df):,} samples")
    print("-" * 50)
    
    return final_df

def to_triplets(df, triplets_per_query=10):
    """
    Simple conversion to triplet format for two-tower training.
    
    Args:
        df: DataFrame with current format (query, document, passage_sign_de columns)
        triplets_per_query: Number of triplets to generate per query
    
    Returns:
        DataFrame with columns: query, positive_example, negative_example
    """
    triplets = []
    
    for query_id, group in df.groupby('query_id'):
        query_text = group['query'].iloc[0]
        
        # Get positive and negative documents
        pos_docs = group[group['passage_sign_de'] == 1]['document'].tolist()
        neg_docs = group[group['passage_sign_de'] == 0]['document'].tolist()
        
        # Skip if missing positives or negatives
        if not pos_docs or not neg_docs:
            continue
            
        # Create triplets by randomly pairing pos and neg documents
        for _ in range(triplets_per_query):
            triplets.append({
                'query': query_text,
                'positive_example': np.random.choice(pos_docs),
                'negative_example': np.random.choice(neg_docs)
            })
    
    return pd.DataFrame(triplets)

# Convert processed results to training format
def convert_to_training_format(triplets_df, max_samples=None):
    """
    Convert DataFrame with columns ['query', 'positive_example', 'negative_example'] 
    to list of tuples format: [(query, positive, negative), ...]
    """
    if max_samples:
        triplets_df = triplets_df.head(max_samples)
    
    training_data = []
    for _, row in triplets_df.iterrows():
        training_data.append((
            row['query'],
            row['positive_example'], 
            row['negative_example']
        ))
    
    return training_data