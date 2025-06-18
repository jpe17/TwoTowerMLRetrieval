from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
import chromadb

model = SentenceTransformer('sentence-transformers/msmarco-MiniLM-L-6-v3')
# Load persistent ChromaDB
client = chromadb.PersistentClient(path="./chroma_store")
collection = client.get_or_create_collection("docs")

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
    query_embedding = model.encode([input.query])[0].tolist()

    # Retrieve top 5 results
    results = collection.query(query_embeddings=[query_embedding], n_results=5)
    docs = results.get("documents", [[]])[0]
    ids = results.get("ids", [[]])[0]

    return {"query": input.query, "results": list(zip(ids, docs))}


