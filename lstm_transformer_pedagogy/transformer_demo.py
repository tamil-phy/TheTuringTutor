import torch
import torch.nn as nn
import torch.nn.functional as F

print("==========================================")
print("The Transformer Revolution: Self-Attention")
print("==========================================\n")

print("RNNs and LSTMs process words sequentially. To connect word 1 to word 10,")
print("information must pass through 9 intermediate steps (O(N) distance).\n")

print("Transformers use 'Self-Attention'. Every word looks directly at EVERY OTHER word")
print("simultaneously (O(1) distance). No sequential bottleneck!\n")

# A simple sentence where "bank" means different things based on context
sentence = "the bank of the river"
words = sentence.split()
seq_len = len(words)

print(f"Sentence: '{sentence}'")
print("-" * 50)

# Simulate word embeddings (normally learned, we use random here for demonstration)
# Embedding dimension = 4
embed_dim = 4
torch.manual_seed(42) # For reproducible random values
embeddings = torch.randn(seq_len, embed_dim)

# In a real Transformer, we project embeddings into Query, Key, and Value vectors
# using learned linear layers.
W_q = nn.Linear(embed_dim, embed_dim, bias=False)
W_k = nn.Linear(embed_dim, embed_dim, bias=False)
W_v = nn.Linear(embed_dim, embed_dim, bias=False)

# Get Q, K, V
Q = W_q(embeddings)
K = W_k(embeddings)
V = W_v(embeddings)

# Calculate Attention Scores
# Formula: Softmax( (Q * K^T) / sqrt(d_k) )
attention_scores = torch.matmul(Q, K.transpose(0, 1))
attention_scores = attention_scores / (embed_dim ** 0.5)

# Apply softmax to get probabilities (weights)
attention_weights = F.softmax(attention_scores, dim=-1)

print("\n--- The Attention Matrix ---")
print("How much does each word 'pay attention' to the other words?\n")

# Print the attention weights in a readable grid
header = "      " + "".join([f"{w:>8}" for w in words])
print(header)

for i, word in enumerate(words):
    row_str = f"{word:<6}"
    for j in range(seq_len):
        weight = attention_weights[i, j].item()
        row_str += f"{weight:8.2f}"
    print(row_str)

print("\nNotice:")
print("1. Every word has a direct mathematical connection to every other word.")
print("2. 'bank' can look directly at 'river' without passing through 'of' and 'the'.")
print("3. All of this is calculated as a single matrix multiplication (Parallel!),")
print("   unlike RNNs which require a for-loop over time (Sequential!).")
