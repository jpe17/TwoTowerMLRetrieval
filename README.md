Step 1: Create VENV
cd TwoTowerMLRetrieval
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

Step 2: 
Create folder data and artifacts
Notebooks/download_dataset.ipynb to download the MS Marco Dataset
Gemma Embeddings v1.0
GemmaEmbed is a dense-vector embedding model, trained especially for retrieval.

Step 3:

backend/config.json tweak for training

Explain:

{
    "TRAIN_DATASET_PATH": "data/ms_marco_train.parquet",
    "VAL_DATASET_PATH": "data/ms_marco_validation.parquet",
    "TEST_DATASET_PATH": "data/ms_marco_test.parquet",
    "EMBEDDINGS_PATH": "data/embeddings.npy",
    "WORD_TO_IDX_PATH": "data/word_to_idx.pkl",
    
    "SUBSAMPLE_RATIO": 1.0,
    "NUM_TRIPLETS_PER_QUERY": 1,
    "TRAINING_MODE": "retrieval",
    
    "HIDDEN_DIM": 256,
    "RNN_TYPE": "GRU",
    "NUM_LAYERS": 2,
    "BIDIRECTIONAL": true,
    "DROPOUT": 0.2,
    
    "BATCH_SIZE": 128,
    "EPOCHS": 10,
    "LR": 0.00005,
    
    "MARGIN": 0.3,
    "NORMALIZE_OUTPUT": true
} 

Step 4: Run python backend/main.py 

Step 5: Need to create a wandb account to view the metrics, asides from Terminal

Step 6: Now model params have been saved on artifacts:

Now we have trained a Two Tower model to encode queries as well as tfidf and saved the docmument embeddings, model and tfid artifacts

Once done, we want to have an app that we can send an untrained query to and infer what documents using a combination of tfidf and encoder. 

