# --- Tokenizer and Vocab ---
import pickle
import numpy as np
from collections import defaultdict
from itertools import chain

class PretrainedTokenizer:
    def __init__(self, word_to_idx_path):
        # Load pretrained word_to_idx mapping
        with open(word_to_idx_path, 'rb') as f:
            self.word2idx = pickle.load(f)
        
        print(f"Loaded vocabulary with {len(self.word2idx):,} tokens")

    def encode(self, sentence):
        # Only include words that exist in vocabulary, skip unknown words
        return [self.word2idx[word.lower()] for word in sentence.split() if word.lower() in self.word2idx]

    def vocab_size(self):
        return len(self.word2idx)
