import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter, defaultdict
import random

# ------------------------------------------------------------
print("=" * 60)
print("4. The Neural Solution: RNN & LSTM Language Models")
print("=" * 60)
print("Instead of counting exact matches, Neural Networks learn")
print("to compress the entire past context into a 'Hidden State' vector.\n")

# ------------------------------------------------------------
# 1. Get training corpus from user
# ------------------------------------------------------------
user_corpus = input("Enter your training sentence (corpus): ").strip()
if not user_corpus:
    user_corpus = "i like to eat apples and i like to eat bananas but i do not like to eat grapes"
    print(f"Using default corpus: '{user_corpus}'")

tokens = user_corpus.split()
print(f"\nCorpus tokens: {tokens}\n")

# ------------------------------------------------------------
# 2. Build vocabulary and training data (context size = 3)
# ------------------------------------------------------------
vocab = sorted(list(set(tokens)))
word2idx = {word: i for i, word in enumerate(vocab)}
idx2word = {i: word for i, word in enumerate(vocab)}
vocab_size = len(vocab)
print(f"Vocabulary ({vocab_size} words): {vocab}\n")

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
print(f"Example: context {[idx2word[i] for i in X_train[0]]} -> next word '{idx2word[y_train[0]]}'\n")

# ------------------------------------------------------------
# 3. Define neural models
# ------------------------------------------------------------
class RNNLM(nn.Module):
    def __init__(self, vocab_size, embed_dim=10, hidden_dim=16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        out, _ = self.rnn(embedded)
        return self.fc(out[:, -1, :])

class LSTMLM(nn.Module):
    def __init__(self, vocab_size, embed_dim=10, hidden_dim=16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        out, _ = self.lstm(embedded)
        return self.fc(out[:, -1, :])

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, embed_dim=16, hidden_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = nn.Parameter(torch.zeros(1, context_size, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=2,
                                                   dim_feedforward=hidden_dim,
                                                   batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x) + self.pos_encoder[:, :x.size(1), :]
        out = self.transformer(embedded)
        return self.fc(out[:, -1, :])

# ------------------------------------------------------------
# 4. Initialize and train models
# ------------------------------------------------------------
rnn_model = RNNLM(vocab_size)
lstm_model = LSTMLM(vocab_size)
transformer_model = TransformerLM(vocab_size)

criterion = nn.CrossEntropyLoss()
optimizer_rnn = optim.Adam(rnn_model.parameters(), lr=0.01)
optimizer_lstm = optim.Adam(lstm_model.parameters(), lr=0.01)
optimizer_transformer = optim.Adam(transformer_model.parameters(), lr=0.01)

print("Training RNN, LSTM, and Transformer Language Models...")
epochs = 150
for epoch in range(epochs):
    optimizer_rnn.zero_grad()
    rnn_out = rnn_model(X_tensor)
    rnn_loss = criterion(rnn_out, y_tensor)
    rnn_loss.backward()
    optimizer_rnn.step()

    optimizer_lstm.zero_grad()
    lstm_out = lstm_model(X_tensor)
    lstm_loss = criterion(lstm_out, y_tensor)
    lstm_loss.backward()
    optimizer_lstm.step()

    optimizer_transformer.zero_grad()
    transformer_out = transformer_model(X_tensor)
    transformer_loss = criterion(transformer_out, y_tensor)
    transformer_loss.backward()
    optimizer_transformer.step()

print("Training Complete!\n")

# ------------------------------------------------------------
# 5. Test with your own context
# ------------------------------------------------------------
print("--- Testing the N-Gram Wall Sequence (or your own) ---")
test_context = input("Enter a 3-word context (or press Enter for default 'and i do'): ").strip()
if not test_context:
    test_context = "and i do"
test_words = test_context.split()

if len(test_words) != 3:
    print(f"Warning: You entered {len(test_words)} words, but the model expects exactly 3. Using last 3 words if available.")
    # Take last 3 words, or pad with first vocabulary word if too short
    if len(test_words) < 3:
        test_words = [vocab[0]] * (3 - len(test_words)) + test_words
    else:
        test_words = test_words[-3:]

print(f"Context: '{' '.join(test_words)}'")

# Convert to indices, unknown words become 0 (first word in vocab)
test_indices = [word2idx.get(w, 0) for w in test_words]
test_tensor = torch.tensor([test_indices], dtype=torch.long)

with torch.no_grad():
    rnn_pred_idx = rnn_model(test_tensor).argmax().item()
    lstm_pred_idx = lstm_model(test_tensor).argmax().item()
    trans_pred_idx = transformer_model(test_tensor).argmax().item()

print(f"\nRNN Model Prediction:      '{idx2word[rnn_pred_idx]}'")
print(f"LSTM Model Prediction:     '{idx2word[lstm_pred_idx]}'")
print(f"Transformer Prediction:    '{idx2word[trans_pred_idx]}'")
print("\nNotice how Neural Networks still make a prediction!")
print("They don't rely on exact matches – they generalise context through hidden states.")