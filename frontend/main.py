from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import sys
import numpy as np
from numpy.linalg import norm
from pathlib import Path
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# --- Project Root Setup ---
# This makes file paths robust, whether running locally or in a container.
APP_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = APP_DIR.parent
sys.path.append(str(PROJECT_DIR / "backend"))
# ---

# This now imports our custom, simplified inferencer
from query_inferencer import QueryInferencer 

import chromadb

# --- CONFIGURATION ---
# IMPORTANT: Now dynamically finds the latest artifacts directory
def find_latest_artifacts():
    artifacts_base = PROJECT_DIR / "artifacts"
    if not artifacts_base.exists():
        return None
    
    # Find all run directories
    run_dirs = [d for d in artifacts_base.iterdir() if d.is_dir() and d.name.startswith('run')]
    if not run_dirs:
        return None
    
    # Return the most recent one
    return max(run_dirs, key=lambda x: x.stat().st_mtime)

ARTIFACTS_PATH = find_latest_artifacts()
if ARTIFACTS_PATH is None:
    print("FATAL: No artifacts directory found. Please train a model first.")
    sys.exit(1)

print(f"📁 Using artifacts from: {ARTIFACTS_PATH}")

CHROMA_STORE_PATH = str(APP_DIR / "chroma_store")
COLLECTION_NAME = "docs"
# ---------------------

# --- INITIALIZATION ---
print("🚀 Initializing backend...")
# Initialize the inferencer with the path to the trained model artifacts
artifacts_path = Path(ARTIFACTS_PATH)
if not artifacts_path.exists():
    print(f"FATAL: Artifacts directory not found at {ARTIFACTS_PATH}")
    print("Please run backend/main.py to train a model and then run frontend/1_Index_Documents.ipynb to create the database.")
    sys.exit(1)

inferencer = QueryInferencer(artifacts_path=str(artifacts_path))

# Load TF-IDF artifacts and document list for mapping
tfidf_artifacts_path = artifacts_path / "tfidf_artifacts.pkl"
documents_path = artifacts_path / "documents.pkl"
if not tfidf_artifacts_path.exists() or not documents_path.exists():
    print(f"FATAL: TF-IDF or document artifacts not found in {ARTIFACTS_PATH}")
    print("Please re-run `backend/main.py` to generate the necessary files.")
    sys.exit(1)

with open(tfidf_artifacts_path, 'rb') as f:
    tfidf_data = pickle.load(f)
tfidf_vectorizer = tfidf_data['vectorizer']
doc_tfidf_matrix = tfidf_data['matrix']

with open(documents_path, 'rb') as f:
    all_documents_list = pickle.load(f)
# Create a mapping from document text to its index for quick TF-IDF lookups
doc_to_index = {doc: i for i, doc in enumerate(all_documents_list)}
print("✅ TF-IDF artifacts loaded.")


# Load persistent ChromaDB
client = chromadb.PersistentClient(path=CHROMA_STORE_PATH)
collection = client.get_or_create_collection(COLLECTION_NAME)
print(f"✅ ChromaDB collection '{COLLECTION_NAME}' loaded with {collection.count()} documents.")
print("✅ Backend ready.")
# ---------------------

class QueryInput(BaseModel):
    query: str
    alpha: float = 0.5  # Default value, but can be overridden

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the search interface HTML."""
    html_path = APP_DIR / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(), status_code=200)
    else:
        return HTMLResponse(content="<h1>Frontend not found</h1>", status_code=404)

@app.post("/search")
def search(input: QueryInput):
    """
    Simple 5-step hybrid search:
    1. Get top 50 documents via semantic similarity
    2. Compute semantic scores for those 50
    3. Compute TF-IDF scores for those 50 
    4. Combine using alpha weighting
    5. Return top 10 by final score
    """
    import time
    start = time.time()
    alpha = input.alpha
    
    # Step 1: Get top 50 documents via semantic similarity
    query_embedding = inferencer.get_query_embedding(input.query)
    semantic_results = collection.query(
        query_embeddings=[query_embedding.tolist()], 
        n_results=50
    )
    
    top_docs = semantic_results["documents"][0]
    semantic_distances = semantic_results["distances"][0]
    
    # Step 2: Compute semantic scores (0-1)
    semantic_scores = [1 - dist for dist in semantic_distances]
    
    # Step 3: Compute TF-IDF scores for those 50 documents
    query_tfidf = tfidf_vectorizer.transform([input.query])
    tfidf_scores = []
    
    for doc in top_docs:
        doc_idx = doc_to_index.get(doc)
        if doc_idx is not None:
            # Use pre-computed TF-IDF matrix
            tfidf_score = cosine_similarity(query_tfidf, doc_tfidf_matrix[doc_idx:doc_idx+1])[0][0]
        else:
            # Compute on-the-fly if not in index
            doc_tfidf = tfidf_vectorizer.transform([doc])
            tfidf_score = cosine_similarity(query_tfidf, doc_tfidf)[0][0]
        tfidf_scores.append(float(tfidf_score))  # Ensure float conversion
    
    # Debug: Print score ranges
    print(f"🔍 Semantic scores range: {min(semantic_scores):.3f} - {max(semantic_scores):.3f}")
    print(f"🔍 TF-IDF scores range: {min(tfidf_scores):.3f} - {max(tfidf_scores):.3f}")
    
    # Step 4: Calculate final scores using alpha
    results = []
    for i, doc in enumerate(top_docs):
        semantic_score = float(semantic_scores[i])
        tfidf_score = float(tfidf_scores[i])
        final_score = alpha * semantic_score + (1 - alpha) * tfidf_score
        
        results.append({
            "doc": doc,
            "score": float(final_score),
            "dense_score": semantic_score,
            "tfidf_score": tfidf_score
        })
    
    # Step 5: Sort by final score and return top 10
    results.sort(key=lambda x: x["score"], reverse=True)
    top_10 = results[:10]
    
    elapsed = (time.time() - start) * 1000
    print(f"⚡ Search completed in {elapsed:.1f}ms")
    
    return {
        "query": input.query,
        "alpha": alpha,
        "results": [
            {"rank": i+1, "id": f"result-{i+1}", **res} 
            for i, res in enumerate(top_10)
        ]
    }



    #return {"query": input.query, "results": list(zip(ids, docs))}


