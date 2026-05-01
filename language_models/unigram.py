import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
import random

# Set random seed for reproducibility
torch.manual_seed(42)
random.seed(42)

# ==========================================
# 1-Gram (Unigram) Model
# ==========================================
class UnigramModel:
    def __init__(self, corpus_tokens):
        self.counts = Counter(corpus_tokens)
        self.total_tokens = len(corpus_tokens)
        
    def predict_next(self, context=None):   # context is ignored – unigram has no memory
        # Returns the most frequent word in the training corpus
        most_common = self.counts.most_common(1)[0]
        return most_common[0]

print("--- Unigram Language Model from Your Own Text ---")
print("The model will learn word frequencies from the sentence you provide.\n")

# 1. User inputs the training corpus (replaces the hardcoded 'text')
user_corpus = input("Enter your training corpus (e.g., a sentence): ").strip()
if not user_corpus:
    user_corpus = "i like to eat apples and i like to eat bananas"  # fallback
    print(f"Using default corpus: '{user_corpus}'")

tokens = user_corpus.split()
print(f"\nCorpus tokens: {tokens}\n")

# 2. Build the unigram model
unigram = UnigramModel(tokens)

# 3. (Optional) Get context from user – unigram ignores it
context = input("Enter any context phrase (will be ignored by unigram): ").strip()
if not context:
    context = "i like to eat"

# 4. Prediction – always the most frequent word
prediction = unigram.predict_next(context)
print(f"\nUnigram prediction (most frequent word in your corpus): '{prediction}'")