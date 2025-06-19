#!/usr/bin/env python3
"""
Render.com Deployment Setup Script
This script builds all necessary artifacts during deployment since 
artifacts/ and chroma_store/ are not in the Git repository.
"""

import os
import sys
import subprocess
import urllib.request
import zipfile
import shutil
from pathlib import Path

def download_and_extract_glove():
    """Download GloVe embeddings if not present"""
    print("📥 Checking for GloVe embeddings...")
    
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    glove_file = data_dir / "glove.6B.200d.txt"
    
    if glove_file.exists():
        print("✅ GloVe embeddings already present")
        return True
    
    print("🔄 Downloading GloVe embeddings (this may take a few minutes)...")
    glove_url = "https://nlp.stanford.edu/data/glove.6B.zip"
    zip_path = data_dir / "glove.6B.zip"
    
    try:
        urllib.request.urlretrieve(glove_url, zip_path)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Extract only the 200d file we need
            zip_ref.extract("glove.6B.200d.txt", data_dir)
        
        # Clean up zip file
        zip_path.unlink()
        print("✅ GloVe embeddings downloaded and extracted")
        return True
        
    except Exception as e:
        print(f"❌ Failed to download GloVe embeddings: {e}")
        return False

def setup_sample_data():
    """Create sample data if no data sources are available"""
    print("📝 Setting up sample data...")
    
    data_dir = Path("data")
    sample_file = data_dir / "sample_documents.txt"
    
    if sample_file.exists():
        return True
    
    # Create sample documents for demonstration
    sample_docs = [
        "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
        "Deep learning uses neural networks with multiple layers to learn complex patterns.",
        "Natural language processing helps computers understand and generate human language.",
        "Computer vision enables machines to interpret and analyze visual information.",
        "Reinforcement learning trains agents through interaction with an environment.",
        "Supervised learning uses labeled data to train predictive models.",
        "Unsupervised learning finds patterns in data without explicit labels.",
        "Transfer learning leverages pre-trained models for new tasks.",
        "Feature engineering involves selecting and transforming input variables.",
        "Model evaluation assesses the performance of machine learning algorithms.",
        "Cross-validation helps estimate model performance on unseen data.",
        "Overfitting occurs when a model learns training data too specifically.",
        "Regularization techniques prevent overfitting in machine learning models.",
        "Gradient descent optimizes model parameters by minimizing loss functions.",
        "Backpropagation calculates gradients for training neural networks.",
        "Convolutional neural networks excel at image recognition tasks.",
        "Recurrent neural networks process sequential data effectively.",
        "Attention mechanisms help models focus on relevant input parts.",
        "Transformers revolutionized natural language processing applications.",
        "BERT and GPT are powerful pre-trained language models."
    ]
    
    with open(sample_file, 'w', encoding='utf-8') as f:
        for doc in sample_docs:
            f.write(doc + '\n')
    
    print("✅ Sample data created")
    return True

def train_model():
    """Train the model using available data"""
    print("🔄 Training model...")
    
    try:
        # Add backend to Python path
        sys.path.insert(0, 'backend')
        
        # Run the training script
        result = subprocess.run([
            sys.executable, 'backend/main.py'
        ], capture_output=True, text=True, cwd='.')
        
        if result.returncode == 0:
            print("✅ Model training completed successfully")
            print("Training output:", result.stdout[-500:])  # Last 500 chars
            return True
        else:
            print(f"❌ Model training failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error during model training: {e}")
        return False

def create_chroma_index():
    """Create ChromaDB index from trained model"""
    print("🔄 Creating ChromaDB index...")
    
    try:
        # Run the indexing process
        sys.path.insert(0, 'frontend')
        
        # Convert and run the notebook programmatically
        result = subprocess.run([
            sys.executable, '-c', '''
import sys
sys.path.append("frontend")
sys.path.append("backend")

# Import required modules
import chromadb
from pathlib import Path
import pickle
import numpy as np
from query_inferencer import QueryInferencer

print("🔧 Building ChromaDB index...")

# Find latest artifacts
artifacts_base = Path("artifacts")
run_dirs = [d for d in artifacts_base.iterdir() if d.is_dir() and d.name.startswith("run")]
if not run_dirs:
    raise Exception("No training runs found")

latest_run = max(run_dirs, key=lambda x: x.stat().st_mtime)
print(f"Using artifacts from: {latest_run}")

# Load documents
documents_path = latest_run / "documents.pkl"
if not documents_path.exists():
    raise Exception("Documents file not found")

with open(documents_path, "rb") as f:
    documents = pickle.load(f)

print(f"Loaded {len(documents)} documents")

# Initialize inferencer
inferencer = QueryInferencer(str(latest_run))

# Create ChromaDB
chroma_dir = Path("frontend/chroma_store")
chroma_dir.mkdir(parents=True, exist_ok=True)

client = chromadb.PersistentClient(path=str(chroma_dir))
collection = client.get_or_create_collection("docs")

# Clear existing data
try:
    collection.delete(where={})
except:
    pass

# Generate embeddings and add to ChromaDB
batch_size = 100
for i in range(0, len(documents), batch_size):
    batch_docs = documents[i:i+batch_size]
    batch_ids = [f"doc_{j}" for j in range(i, min(i+batch_size, len(documents)))]
    
    # Generate embeddings
    embeddings = []
    for doc in batch_docs:
        emb = inferencer.get_document_embedding(doc)
        embeddings.append(emb.tolist())
    
    # Add to collection
    collection.add(
        embeddings=embeddings,
        documents=batch_docs,
        ids=batch_ids
    )
    
    print(f"Processed {min(i+batch_size, len(documents))}/{len(documents)} documents")

print(f"✅ ChromaDB index created with {collection.count()} documents")
'''
        ], capture_output=True, text=True, cwd='.')
        
        if result.returncode == 0:
            print("✅ ChromaDB index created successfully")
            return True
        else:
            print(f"❌ ChromaDB indexing failed: {result.stderr}")
            print(f"Output: {result.stdout}")
            return False
            
    except Exception as e:
        print(f"❌ Error during ChromaDB creation: {e}")
        return False

def main():
    """Main deployment setup for Render"""
    print("🚀 Starting Render deployment setup...")
    
    # Step 1: Download GloVe embeddings
    if not download_and_extract_glove():
        print("⚠️  Failed to download GloVe, using sample data only")
        
    # Step 2: Setup sample data if needed
    setup_sample_data()
    
    # Step 3: Train model
    if not train_model():
        print("❌ Model training failed - deployment cannot continue")
        return False
    
    # Step 4: Create ChromaDB index
    if not create_chroma_index():
        print("❌ ChromaDB creation failed - deployment cannot continue")
        return False
    
    print("🎉 Render deployment setup completed successfully!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 