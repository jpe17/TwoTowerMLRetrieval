#!/usr/bin/env python3
"""
Deployment setup script for TwoTowerMLRetrieval
This script ensures all necessary artifacts are present for deployment.
"""

import os
import sys
from pathlib import Path
import shutil

def check_artifacts():
    """Check if model artifacts exist"""
    artifacts_dir = Path('artifacts')
    if not artifacts_dir.exists():
        return False, "Artifacts directory not found"
    
    # Look for any run directory
    run_dirs = [d for d in artifacts_dir.iterdir() if d.is_dir() and d.name.startswith('run')]
    if not run_dirs:
        return False, "No training run directories found in artifacts/"
    
    # Check the most recent run (or specific run)
    latest_run = max(run_dirs, key=lambda x: x.stat().st_mtime)
    required_files = ['config.json', 'model_weights.h5', 'tokenizer.pkl', 'tfidf_artifacts.pkl', 'documents.pkl']
    
    for file in required_files:
        if not (latest_run / file).exists():
            return False, f"Missing required file: {file} in {latest_run}"
    
    return True, str(latest_run)

def check_chroma_store():
    """Check if ChromaDB store exists"""
    chroma_store = Path('frontend/chroma_store')
    if not chroma_store.exists():
        return False, "ChromaDB store not found at frontend/chroma_store"
    
    # Check if it has any data
    if not any(chroma_store.iterdir()):
        return False, "ChromaDB store is empty"
    
    return True, str(chroma_store)

def main():
    print("🔧 Validating deployment artifacts...")
    
    # Check artifacts
    artifacts_ok, artifacts_msg = check_artifacts()
    if not artifacts_ok:
        print(f"❌ Artifacts check failed: {artifacts_msg}")
        print("📝 To fix: Run 'python backend/main.py' to train the model")
        return False
    
    print(f"✅ Model artifacts found: {artifacts_msg}")
    
    # Check ChromaDB
    chroma_ok, chroma_msg = check_chroma_store()
    if not chroma_ok:
        print(f"❌ ChromaDB check failed: {chroma_msg}")
        print("📝 To fix: Run the 'frontend/1_Index_Documents.ipynb' notebook")
        return False
    
    print(f"✅ ChromaDB store found: {chroma_msg}")
    print("🚀 All deployment artifacts are ready!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 