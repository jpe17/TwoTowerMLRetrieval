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
# IMPORTANT: Update this with the folder name from your latest training run.
# Example: ARTIFACTS_PATH = PROJECT_DIR / "artifacts" / "run_20240101_120000"
ARTIFACTS_PATH = PROJECT_DIR / "artifacts" / "run-20250619_212044" # 👈 CHANGE THIS
HYBRID_ALPHA = 0.5 # Weight for dense search (1.0 = pure dense, 0.0 = pure TF-IDF)

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
    # 1. Encode query for dense search
    query_embedding = inferencer.get_query_embedding(input.query)
    
    # 2. Retrieve top 20 candidates from ChromaDB for re-ranking
    dense_results = collection.query(
        query_embeddings=[query_embedding.tolist()], 
        n_results=20 # Get a larger pool of candidates
    )
    
    dense_docs = dense_results.get("documents", [[]])[0]
    dense_distances = dense_results.get("distances", [[]])[0]

    if not dense_docs:
        return {"query": input.query, "results": []}

    # 3. Transform query for sparse search
    query_tfidf = tfidf_vectorizer.transform([input.query])
    
    # 4. Re-rank candidates using a hybrid score
    hybrid_results = []
    for doc_text, dist in zip(dense_docs, dense_distances):
        doc_idx = doc_to_index.get(doc_text)
        if doc_idx is None:
            continue
        
        # Get pre-computed TF-IDF vector for this doc
        doc_tfidf = doc_tfidf_matrix[doc_idx]
        
        # Calculate scores
        tfidf_score = cosine_similarity(query_tfidf, doc_tfidf)[0][0]
        dense_score = 1 - dist  # Chroma's distance is 1-sim

        # Combine scores
        combined_score = HYBRID_ALPHA * dense_score + (1 - HYBRID_ALPHA) * tfidf_score
        
        hybrid_results.append({
            "doc": doc_text,
            "score": float(combined_score)
        })

    # 5. Sort by hybrid score and return top 5
    hybrid_results.sort(key=lambda x: x["score"], reverse=True)
    top_5_results = hybrid_results[:5]

    # Format for final output
    final_results = []
    for i, res in enumerate(top_5_results, start=1):
        # Normalize score to 0-1 range for cleaner display, assuming scores are roughly in that range
        scaled_score = np.clip(res["score"], 0, 1)
        
        final_results.append({
            "rank": i,
            "id": f"hybrid-result-{i}",
            "doc": res["doc"],
            "score": float(scaled_score)
        })
        
    print(f"Query: '{input.query}' -> Found {len(final_results)} hybrid results.")

    return {
        "query": input.query,
        "results": final_results
    }



    #return {"query": input.query, "results": list(zip(ids, docs))}


