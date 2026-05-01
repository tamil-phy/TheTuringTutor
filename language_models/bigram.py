import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict, Counter
import random

# Set random seed for reproducibility
torch.manual_seed(42)
random.seed(42)

# ==========================================
# Bigram Model (Context = 1)
# Predicts next word based ONLY on the last word seen.
# ==========================================
class BigramModel:
    def __init__(self, corpus_tokens):
        self.bigram_counts = defaultdict(Counter)
        for i in range(len(corpus_tokens) - 1):
            current_word = corpus_tokens[i]
            next_word = corpus_tokens[i+1]
            self.bigram_counts[current_word][next_word] += 1

    def predict_next(self, current_word):
        # If current_word never appeared in training, we cannot predict
        if current_word not in self.bigram_counts:
            return "<UNKNOWN>"
        # Return the most frequent next word for this context
        return self.bigram_counts[current_word].most_common(1)[0][0]

print("--- Bigram Language Model from Your Own Text ---")
print("The model learns which word follows which word from your training sentence.\n")

# 1. User inputs the training corpus
user_corpus = input("Enter your training corpus (e.g., 'i like to eat apples'): ").strip()
if not user_corpus:
    user_corpus = "i like to eat apples and i like to eat bananas"
    print(f"Using default corpus: '{user_corpus}'")

tokens = user_corpus.split()
print(f"\nCorpus tokens: {tokens}")

# 2. Build the bigram model
bigram = BigramModel(tokens)

# 3. User inputs the context (last word)
context_word = input("\nEnter the last word you want to condition on (context): ").strip()
if not context_word:
    context_word = "eat"
    print(f"Using default context: '{context_word}'")

# 4. Prediction
prediction = bigram.predict_next(context_word)
print(f"\nPrediction after '{context_word}': '{prediction}'")