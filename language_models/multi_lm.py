import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter

print("=" * 70)
print("TRAIN ALL THREE MODELS (RNN, LSTM, Transformer) + GENERATE FROM EACH")
print("=" * 70)

# ------------------------------------------------------------------
# 1. Get training corpus from user
# ------------------------------------------------------------------
user_corpus = input("\nEnter your training sentence (corpus): ").strip()
if not user_corpus:
    user_corpus = "i like to eat apples and i like to eat bananas but i do not like to eat grapes"
    print(f"Using default corpus: '{user_corpus}'")

tokens = user_corpus.split()
print(f"\nCorpus tokens: {tokens}\n")

# ------------------------------------------------------------------
# 2. Build vocabulary and training data (context size = 3)
# ------------------------------------------------------------------
vocab = sorted(list(set(tokens)))
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for i, w in enumerate(vocab)}
vocab_size = len(vocab)
print(f"Vocabulary ({vocab_size} unique words): {vocab}\n")

context_size = 3
X_train, y_train = [], []
for i in range(len(tokens) - context_size):
    ctx = [word2idx[w] for w in tokens[i:i+context_size]]
    tgt = word2idx[tokens[i+context_size]]
    X_train.append(ctx)
    y_train.append(tgt)

X_tensor = torch.tensor(X_train, dtype=torch.long)
y_tensor = torch.tensor(y_train, dtype=torch.long)
print(f"Training samples: {len(X_train)}")
if len(X_train) > 0:
    print(f"Example: {[idx2word[i] for i in X_train[0]]} -> '{idx2word[y_train[0]]}'\n")
else:
    print("Corpus too short for training (need at least 4 words). Exiting.")
    exit()

# ------------------------------------------------------------------
# 3. Define the three models
# ------------------------------------------------------------------
class RNNLM(nn.Module):
    def __init__(self, vocab_size, embed_dim=10, hidden_dim=16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.rnn(emb)
        return self.fc(out[:, -1, :])

class LSTMLM(nn.Module):
    def __init__(self, vocab_size, embed_dim=10, hidden_dim=16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.lstm(emb)
        return self.fc(out[:, -1, :])

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, embed_dim=16, nhead=2, hidden_dim=32, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = nn.Parameter(torch.zeros(1, context_size, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=hidden_dim,
            batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, vocab_size)
    def forward(self, x):
        emb = self.embedding(x) + self.pos_encoder[:, :x.size(1), :]
        out = self.transformer(emb)
        return self.fc(out[:, -1, :])

# ------------------------------------------------------------------
# 4. Training function
# ------------------------------------------------------------------
def train_model(model, name, epochs=150, lr=0.01):
    print(f"Training {name}...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(X_tensor)
        loss = criterion(out, y_tensor)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 50 == 0:
            print(f"  {name} Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    return model

# ------------------------------------------------------------------
# 5. Train all three models
# ------------------------------------------------------------------
rnn_model = train_model(RNNLM(vocab_size), "RNN")
lstm_model = train_model(LSTMLM(vocab_size), "LSTM")
transformer_model = train_model(TransformerLM(vocab_size), "Transformer")
print("\nAll three models trained successfully!\n")

# ------------------------------------------------------------------
# 6. Generate text from all models (autoregressive)
# ------------------------------------------------------------------
def generate_text(model, start_words, num_words=8):
    current = start_words.copy()
    for _ in range(num_words):
        window = current[-context_size:] if len(current) >= context_size else current
        indices = [word2idx.get(w, 0) for w in window]
        while len(indices) < context_size:
            indices.insert(0, 0)
        inp = torch.tensor([indices], dtype=torch.long)
        with torch.no_grad():
            pred_idx = model(inp).argmax().item()
            pred_word = idx2word[pred_idx]
            current.append(pred_word)
    return " ".join(current)

print("--- TEXT GENERATION ---")
seed_phrase = input("Enter seed phrase (at least 1 word): ").strip()
if not seed_phrase:
    seed_phrase = "i like"
    print(f"Using default: '{seed_phrase}'")
seed_words = seed_phrase.split()

try:
    num_words = int(input("Number of words to generate: ").strip())
except:
    num_words = 8

print("\n" + "=" * 70)
print("GENERATED TEXT FROM EACH MODEL:")
print("=" * 70)

rnn_out = generate_text(rnn_model, seed_words, num_words)
lstm_out = generate_text(lstm_model, seed_words, num_words)
trans_out = generate_text(transformer_model, seed_words, num_words)

print(f"RNN        : '{rnn_out}'")
print(f"LSTM       : '{lstm_out}'")
print(f"Transformer: '{trans_out}'")
print("\nNote: Each model learns differently – compare their 'hallucinations'!")