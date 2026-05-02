import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter

# ------------------------------------------------------------
print("=" * 60)
print("LSTM LANGUAGE MODEL - Standalone Version")
print("=" * 60)
print("Learns to predict the next word given a 3‑word context.")
print("LSTM uses a memory cell to avoid vanishing gradients and handle longer dependencies.\n")

# ------------------------------------------------------------
# 1. GET TRAINING CORPUS FROM USER
# ------------------------------------------------------------
user_corpus = input("Enter your training sentence (corpus): ").strip()
if not user_corpus:
    user_corpus = "i like to eat apples and i like to eat bananas but i do not like to eat grapes"
    print(f"Using default corpus: '{user_corpus}'")

tokens = user_corpus.split()
print(f"\nCorpus tokens: {tokens}\n")

# ------------------------------------------------------------
# 2. BUILD VOCABULARY
# ------------------------------------------------------------
vocab = sorted(list(set(tokens)))
word2idx = {word: i for i, word in enumerate(vocab)}
idx2word = {i: word for i, word in enumerate(vocab)}
vocab_size = len(vocab)
print(f"Vocabulary ({vocab_size} unique words): {vocab}\n")

# ------------------------------------------------------------
# 3. CREATE TRAINING SAMPLES (context size = 3)
# ------------------------------------------------------------
context_size = 3
X_train = []
y_train = []
for i in range(len(tokens) - context_size):
    context = [word2idx[w] for w in tokens[i:i+context_size]]
    target = word2idx[tokens[i+context_size]]
    X_train.append(context)
    y_train.append(target)

X_tensor = torch.tensor(X_train, dtype=torch.long)
y_tensor = torch.tensor(y_train, dtype=torch.long)
print(f"Created {len(X_train)} training examples.")

if len(X_train) > 0:
    print(f"Example: context {[idx2word[i] for i in X_train[0]]} -> next word '{idx2word[y_train[0]]}'\n")
else:
    print("Error: Not enough words to create training examples. Please provide a corpus with at least 4 words.\n")
    import sys
    sys.exit(1)

# ------------------------------------------------------------
# 4. DEFINE LSTM MODEL
# ------------------------------------------------------------
class LSTMLM(nn.Module):
    def __init__(self, vocab_size, embed_dim=10, hidden_dim=16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)                 # (batch, seq_len, embed_dim)
        out, (hidden, cell) = self.lstm(embedded)    # out: (batch, seq_len, hidden_dim)
        # Use only the last time step's output
        return self.fc(out[:, -1, :])                # (batch, vocab_size)

# ------------------------------------------------------------
# 5. TRAIN THE LSTM
# ------------------------------------------------------------
model = LSTMLM(vocab_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

print("Training LSTM Language Model...")
epochs = 150
for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(X_tensor)
    loss = criterion(output, y_tensor)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

print("Training complete!\n")

# ------------------------------------------------------------
# 6. TEST WITH YOUR OWN 3‑WORD CONTEXT
# ------------------------------------------------------------
print("--- Test the LSTM ---")
print("Enter a 3‑word context. Unknown words will be mapped to the first word in vocabulary.")
test_input = input("Context (3 words): ").strip()
if not test_input:
    test_input = "and i do"
    print(f"Using default: '{test_input}'")

test_words = test_input.split()
if len(test_words) != 3:
    print(f"Warning: You gave {len(test_words)} word(s). Using last 3 words or padding.")
    # Pad or truncate to exactly 3 words
    if len(test_words) < 3:
        test_words = [vocab[0]] * (3 - len(test_words)) + test_words
    else:
        test_words = test_words[-3:]

print(f"\nContext: {' '.join(test_words)}")

# Convert to indices (unknown -> 0)
indices = [word2idx.get(w, 0) for w in test_words]
tensor_input = torch.tensor([indices], dtype=torch.long)

with torch.no_grad():
    pred_idx = model(tensor_input).argmax().item()
    predicted_word = idx2word[pred_idx]

print(f"LSTM prediction for next word: '{predicted_word}'")
print("\nNote: The LSTM generalizes to unseen contexts and handles longer dependencies better than a simple RNN.")