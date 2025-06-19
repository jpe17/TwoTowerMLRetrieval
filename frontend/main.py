from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import numpy as np
from numpy.linalg import norm

sys.path.append("../backend")
from query_inferencer import QueryInferencer

import chromadb
# Load persistent ChromaDB
client = chromadb.PersistentClient(path="./chroma_store")
collection = client.get_or_create_collection("docs")
print(f"Chormadb collection count: {collection.count()}")    
inferencer = QueryInferencer(config_path="fe_config.json", model_path="../data/full_model.pth")

class QueryInput(BaseModel):
    query: str

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify domains like ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/search")
def search(input: QueryInput):
    # Encode query
    embedding = inferencer.get_query_embedding(input.query)
    query_vec = np.array(embedding)

    print(f"Query: '{input.query}'")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding norm: {np.linalg.norm(embedding):.4f}") 
    print(type(embedding), type(embedding.tolist()[0]))
    print(f"Embedding: {embedding[:20]}")  # Print first 10 elements for brevity
    
    # Retrieve top 5 results
    results = collection.query(query_embeddings=[embedding], n_results=5)
    docs = results.get("documents", [[]])[0]
    ids = results.get("ids", [[]])[0]

    # Fetch document embeddings
    retrieved = collection.get(ids=ids, include=["embeddings"])
    doc_embeddings = retrieved["embeddings"]

    # Compute and store results with similarity
    scored_results = []
    for i, (doc_id, doc, doc_vec) in enumerate(zip(ids, docs, doc_embeddings), start=1):
        doc_vec = np.array(doc_vec)
        similarity = np.dot(query_vec, doc_vec) / (norm(query_vec) * norm(doc_vec) + 1e-8)
        scored_results.append({
            "rank": i,
            "id": doc_id,
            "doc": doc,
            "score": float(similarity)
        })

    # Sort by similarity descending
    scored_results.sort(key=lambda x: x["score"], reverse=True)

    return {
        "query": input.query,
        "results": scored_results
    }



    #return {"query": input.query, "results": list(zip(ids, docs))}


