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
    Performs a simplified and robust hybrid search.
    1. Retrieves top candidates from both dense (semantic) and sparse (keyword) retrievers.
    2. Creates a unified pool of candidates.
    3. Re-ranks the entire pool using a weighted average of dense and sparse scores.
    """
    hybrid_alpha = input.alpha
    n_candidates = 20  # Number of candidates to fetch from each retriever
    top_k = 10         # Final number of results to return

    # --- Stage 1: Candidate Retrieval ---
    
    # 1a. DENSE retrieval (ChromaDB for semantic search)
    query_embedding = inferencer.get_query_embedding(input.query)
    dense_results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_candidates
    )
    dense_docs = dense_results.get("documents", [[]])[0]
    dense_distances = dense_results.get("distances", [[]])[0]
    dense_candidates = {doc: (1 - dist) for doc, dist in zip(dense_docs, dense_distances)}

    # 1b. SPARSE retrieval (TF-IDF for keyword search)
    query_tfidf = tfidf_vectorizer.transform([input.query])
    all_tfidf_scores = cosine_similarity(query_tfidf, doc_tfidf_matrix)[0]
    top_sparse_indices = np.argsort(all_tfidf_scores)[::-1][:n_candidates]
    sparse_docs = {all_documents_list[i] for i in top_sparse_indices}

    # 1c. Create a unified candidate pool from both retrievers
    all_candidate_docs = set(dense_candidates.keys()) | sparse_docs

    if not all_candidate_docs:
        return {"query": input.query, "results": []}

    # --- Stage 2: Re-ranking ---
    hybrid_results = []
    for doc_text in all_candidate_docs:
        # Get dense score (use its value or 0 if it wasn't a dense candidate)
        dense_score = dense_candidates.get(doc_text, 0.0)
        
        # Get sparse score for the document by looking it up in the pre-computed array
        doc_idx = doc_to_index.get(doc_text)
        tfidf_score = all_tfidf_scores[doc_idx] if doc_idx is not None else 0.0
        
        # Combine scores using dynamic alpha
        combined_score = hybrid_alpha * dense_score + (1 - hybrid_alpha) * tfidf_score

        hybrid_results.append({
            "doc": doc_text,
            "score": float(combined_score),
            "dense_score": float(dense_score),
            "tfidf_score": float(tfidf_score)
        })

    # CRITICAL: Sort by the new hybrid score to get the final top results
    hybrid_results.sort(key=lambda x: x["score"], reverse=True)
    top_k_results = hybrid_results[:top_k]

    # --- Stage 3: Formatting ---
    final_results = []
    for i, res in enumerate(top_k_results, start=1):
        final_results.append({
            "rank": i, "id": f"hybrid-result-{i}", "doc": res["doc"],
            "score": res["score"], "dense_score": res["dense_score"], "tfidf_score": res["tfidf_score"]
        })

    print(f"Query: '{input.query}' (α={hybrid_alpha:.2f}) -> "
          f"Found {len(final_results)} results from a pool of {len(all_candidate_docs)}.")

    return {
        "query": input.query,
        "alpha": hybrid_alpha,
        "results": final_results
    }



    #return {"query": input.query, "results": list(zip(ids, docs))}


