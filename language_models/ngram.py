import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict, Counter
import random

# Set random seed for reproducibility
torch.manual_seed(42)
random.seed(42)

# ==========================================
# N-Gram Model (Context = N-1 words)
# Predicts next word based on exact last (N-1) word sequence.
# ==========================================
class NgramModel:
    def __init__(self, corpus_tokens, n):
        self.n = n
        self.context_counts = defaultdict(Counter)
        
        context_len = n - 1
        for i in range(len(corpus_tokens) - context_len):
            context = tuple(corpus_tokens[i : i + context_len])
            next_word = corpus_tokens[i + context_len]
            self.context_counts[context][next_word] += 1

    def predict_next(self, context_list):
        # Use only the last (n-1) words from the given context
        context_tuple = tuple(context_list[-(self.n - 1):])
        if context_tuple not in self.context_counts:
            return "<UNKNOWN CONTEXT>"
        return self.context_counts[context_tuple].most_common(1)[0][0]

print("--- N‑Gram Language Model from Your Own Text ---")
print("Learns which word follows a specific sequence of N‑1 words.\n")

# 1. Training corpus
user_corpus = input("Enter your training corpus (sentence): ").strip()
if not user_corpus:
    user_corpus = "i like to eat apples and i like to eat bananas but i do not like to eat grapes"
    print(f"Using default corpus: '{user_corpus}'")

tokens = user_corpus.split()
print(f"\nCorpus tokens: {tokens}\n")

# 2. Choose N
while True:
    try:
        n = int(input("Enter N (gram size, e.g., 2 for bigram, 3 for trigram): ").strip())
        if n < 2:
            print("N must be at least 2. Using N=2.")
            n = 2
        break
    except ValueError:
        print("Please enter an integer.")

# 3. Build the model
ngram = NgramModel(tokens, n)

# 4. Query context (as a phrase)
context_phrase = input(f"\nEnter a context phrase (at least {n-1} words): ").strip()
if not context_phrase:
    # Provide a default example based on the corpus
    if len(tokens) >= n:
        default_context = " ".join(tokens[:n-1])
    else:
        default_context = " ".join(tokens)
    context_phrase = default_context
    print(f"Using default context: '{context_phrase}'")

context_words = context_phrase.split()
if len(context_words) < n - 1:
    print(f"Warning: context has only {len(context_words)} word(s), but {n-1} are needed. Padding with '<UNK>'? I'll still use what you gave.")
    # The model will check exact tuple; if too short, it will likely be unknown.

# 5. Prediction
prediction = ngram.predict_next(context_words)
print(f"\nPrediction after '{context_phrase}': '{prediction}'")

# Optional: show why it might fail
if prediction == "<UNKNOWN CONTEXT>":
    print("\n⚠️ The exact last {}‑word sequence '{}' never appeared in your training corpus.".format(
        n-1, " ".join(context_words[-(n-1):])))