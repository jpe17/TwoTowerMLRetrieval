import pandas as pd
import numpy as np
import os

def print_dataframe_sample(df, title, n_rows=3):
    """Print a sample of the dataframe with title"""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Sample data (first {n_rows} rows):")
    print(df.head(n_rows).to_string())
    print('='*60)

def export_to_csv(df, output_path):
    """Export dataframe to CSV file"""
    print(f"💾 Exporting to: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Exported {len(df):,} rows to {output_path}")

def load_and_filter_data(input_path, target_samples=None):
    """Load data and filter out invalid rows"""
    print(f"📂 Loading data from: {input_path}")
    
    # Load data with fix for nested data structures
    try:
        # First try regular pandas read
        df = pd.read_parquet(input_path)
    except Exception as e:
        print(f"  ⚠️ Standard pandas read failed: {str(e)}")
        print("  🔧 Trying alternative approach for nested data...")
        
        # Alternative 1: Use fastparquet engine
        try:
            df = pd.read_parquet(input_path, engine='fastparquet')
            print("    ✅ Successfully loaded using fastparquet engine")
        except Exception as e2:
            print(f"    ❌ Fastparquet failed: {str(e2)}")
            
            # Alternative 2: Use PyArrow directly then convert
            try:
                import pyarrow.parquet as pq
                table = pq.read_table(input_path)
                df = table.to_pandas(split_blocks=True, self_destruct=True)
                print("    ✅ Successfully loaded using PyArrow direct conversion")
            except Exception as e3:
                print(f"    ❌ PyArrow direct failed: {str(e3)}")
                
                # Alternative 3: Try with use_threads=False
                try:
                    df = pd.read_parquet(input_path, use_nullable_dtypes=False)
                    print("    ✅ Successfully loaded with nullable_dtypes=False")
                except Exception as e4:
                    print(f"    ❌ All methods failed. Last error: {str(e4)}")
                    raise e4
    
    print(f"  Loaded: {len(df):,} samples")
    
    # Sample down if target_samples specified
    if target_samples and len(df) > target_samples:
        df = df.sample(n=target_samples, random_state=42)
        print(f"  ✂️ Sampled down to: {len(df):,} samples")
    
    print_dataframe_sample(df, "RAW DATA LOADED", n_rows=2)
    
    # Filter valid data
    print("🔍 Filtering valid data...")
    valid_mask = (
        df['query'].notna() & 
        df['query_id'].notna() & 
        df['query_type'].notna() & 
        df['passages.is_selected'].notna() &
        df['passages.is_selected'].apply(lambda x: any(x) if isinstance(x, list) else False)
    )
    
    df_filtered = df[valid_mask].copy()
    print(f"  After filtering: {len(df_filtered):,} samples")
    print_dataframe_sample(df_filtered, "FILTERED DATA", n_rows=2)
    
    return df_filtered

def flatten_passages(df):
    """Convert nested passage data to flat rows"""
    print("  🔄 Flattening data...")
    
    # Show what we're working with
    print("\n🔍 Example of nested data structure:")
    row = df.iloc[0]
    print(f"Query: {row['query']}")
    print(f"Query ID: {row['query_id']}")
    print(f"Passages: {len(row['passages.passage_text'])} passages")
    print(f"First passage: {row['passages.passage_text'][0][:100]}...")
    print(f"Is selected: {row['passages.is_selected'][:5]}...")
    
    # Flatten the data
    flattened_rows = []
    
    for idx, row in df.iterrows():
        query = row['query']
        query_id = row['query_id']
        query_type = row['query_type']
        passage_texts = row['passages.passage_text']
        is_selected = row['passages.is_selected']
        
        # Create one row for each passage
        for passage_text, selected in zip(passage_texts, is_selected):
            flattened_rows.append({
                'query': query,
                'query_id': query_id,
                'query_type': query_type,
                'document': passage_text,
                'is_relevant': 1 if selected else 0  # 1 for positive, 0 for negative
            })
    
    df_flat = pd.DataFrame(flattened_rows)
    print(f"  Flattened: {len(df_flat):,} rows")
    print_dataframe_sample(df_flat, "FLATTENED DATA")
    
    return df_flat

def add_negative_samples(df, neg_samples_per_query=5):
    """Add random negative samples for each query"""
    print(f"  ➕ Adding negative samples...")
    
    # Get all documents for random sampling
    all_documents = df['document'].tolist()
    result_rows = []
    
    # Process each query
    for query_id in df['query_id'].unique():
        query_data = df[df['query_id'] == query_id]
        
        # Keep existing positive samples
        result_rows.extend(query_data.to_dict('records'))
        
        # Add negative samples
        first_row = query_data.iloc[0]
        neg_docs = np.random.choice(all_documents, size=min(neg_samples_per_query, len(all_documents)), replace=False)
        
        for neg_doc in neg_docs:
            result_rows.append({
                'query': first_row['query'],
                'query_id': query_id,
                'query_type': first_row['query_type'],
                'document': neg_doc,
                'is_relevant': 0  # Negative sample
            })
    
    df_with_negatives = pd.DataFrame(result_rows)
    
    # Show statistics
    print(f"  With negatives: {len(df_with_negatives):,} rows")
    
    # Show positive/negative distribution
    pos_count = len(df_with_negatives[df_with_negatives['is_relevant'] == 1])
    neg_count = len(df_with_negatives[df_with_negatives['is_relevant'] == 0])
    print(f"    - Positive: {pos_count:,}")
    print(f"    - Negative: {neg_count:,}")
    
    print_dataframe_sample(df_with_negatives, "DATA WITH NEGATIVES")
    
    return df_with_negatives

def convert_to_triplets(df, triplets_per_query=10):
    """Convert to triplet format: (query, positive_example, negative_example)"""
    print(f"  🔄 Converting to triplets...")
    
    triplets = []
    
    for query_id in df['query_id'].unique():
        query_data = df[df['query_id'] == query_id]
        
        # Get positive and negative documents
        positive_docs = query_data[query_data['is_relevant'] == 1]['document'].tolist()
        negative_docs = query_data[query_data['is_relevant'] == 0]['document'].tolist()
        
        if not positive_docs or not negative_docs:
            continue
            
        query_text = query_data['query'].iloc[0]
        
        # Create triplets by randomly pairing positives and negatives
        for _ in range(min(triplets_per_query, len(positive_docs) * len(negative_docs))):
            pos_doc = np.random.choice(positive_docs)
            neg_doc = np.random.choice(negative_docs)
            
            triplets.append({
                'query': query_text,
                'positive_example': pos_doc,  # Match the expected column name
                'negative_example': neg_doc   # Match the expected column name
            })
    
    df_triplets = pd.DataFrame(triplets)
    print(f"  Generated: {len(df_triplets):,} triplets")
    print_dataframe_sample(df_triplets, "TRIPLET DATA")
    
    return df_triplets

def convert_to_training_format(df_triplets):
    """
    Convert triplet dataframe to list of tuples for training
    
    Args:
        df_triplets: DataFrame with columns ['query', 'positive_example', 'negative_example']
    
    Returns:
        List of tuples: [(query, positive_doc, negative_doc), ...]
    """
    print(f"🔄 Converting {len(df_triplets):,} triplets to training format...")
    
    training_data = []
    for _, row in df_triplets.iterrows():
        training_data.append((
            row['query'],
            row['positive_example'], 
            row['negative_example']
        ))
    
    print(f"✅ Converted to {len(training_data):,} training triplets")
    return training_data

def process_dataset_simple(input_path, target_samples=None):
    """
    Simple end-to-end processing pipeline that creates triplets
    
    Args:
        input_path: Path to input parquet file
        target_samples: Number of samples to process (optional)
    """
    print(f"📁 Processing {input_path.split('/')[-1].upper()} dataset (target: {target_samples:,} samples)...")
    
    # Step 1: Load and filter data
    df_filtered = load_and_filter_data(input_path, target_samples)
    
    # Step 2: Flatten nested data
    df_flat = flatten_passages(df_filtered)
    
    # Step 3: Add negative samples
    df_with_negatives = add_negative_samples(df_flat, neg_samples_per_query=5)
    
    # Step 4: Convert to triplets
    df_triplets = convert_to_triplets(df_with_negatives, triplets_per_query=10)
    
    return df_triplets