import torch
import numpy as np
from model import ModelFactory
from tokenizer import PretrainedTokenizer
from data_loader import DataLoader
import json
import os
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

def build_document_embeddings():
    """Build and save document embeddings for evaluation."""
    
    # Load config and model
    with open('backend/config.json', 'r') as f:
        config = json.load(f)
    
    device = torch.device(config.get('DEVICE', 'cpu'))
    model = torch.load('data/full_model.pth', map_location=device)
    model.eval()
    
    # Load tokenizer
    tokenizer = PretrainedTokenizer(config['WORD_TO_IDX_PATH'])
    
    # Load all documents from test set
    data_loader = DataLoader(config)
    datasets = data_loader.load_datasets()
    test_triplets = datasets['test']
    
    # Extract unique documents
    all_documents = set()
    for query, pos_doc, neg_doc in test_triplets:
        all_documents.add(pos_doc)
        all_documents.add(neg_doc)
    
    documents = list(all_documents)
    print(f"Found {len(documents):,} unique documents")
    
    # Create document-to-index mapping
    doc_to_idx = {doc: idx for idx, doc in enumerate(documents)}
    
    # Encode documents in batches
    batch_size = 64
    doc_embeddings = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(documents), batch_size), desc="Encoding documents"):
            batch_docs = documents[i:i + batch_size]
            
            # Tokenize batch
            batch_tokens = [
                torch.tensor(tokenizer.encode(doc), dtype=torch.long)
                for doc in batch_docs
            ]
            
            # Pad and encode
            batch_tensor = pad_sequence(batch_tokens, batch_first=True).to(device)
            embeddings = model.encode_document(batch_tensor)
            
            doc_embeddings.append(embeddings.detach().cpu().numpy())

    
    # Combine all embeddings
    all_embeddings = np.vstack(doc_embeddings)
    
    # Save embeddings and mapping
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(os.path.join(output_dir, 'test_document_embeddings.npy'), all_embeddings)
    
    import pickle
    with open(os.path.join(output_dir, 'test_doc_to_idx.pkl'), 'wb') as f:
        pickle.dump(doc_to_idx, f)
    
    # Save document texts for reference
    with open(os.path.join(output_dir, 'test_documents.txt'), 'w', encoding='utf-8') as f:
        for doc in documents:
            f.write(doc + '\n')
    
    print(f"✅ Saved {len(documents):,} document embeddings")
    print(f"   Embeddings: {all_embeddings.shape}")
    print(f"   Files: test_document_embeddings.npy, test_doc_to_idx.pkl, test_documents.txt")

if __name__ == "__main__":
    build_document_embeddings()