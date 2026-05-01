import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter

print("=" * 70)
print("RNN vs LSTM vs TRANSFORMER on a 250‑word story")
print("=" * 70)

# ------------------------------------------------------------------
# Large corpus: a short story with long-range dependencies
# ------------------------------------------------------------------
corpus = ("the old man walked to the river every morning he carried a wooden bucket the river was wide and deep the man would fill the bucket with water then walk back to his small hut one day a young boy followed the man to the river the boy asked why do you carry water every day the man smiled and said because the village needs water the boy looked at the river and said the river is so big why not build a pipe the man laughed and said that is a good idea but we have no tools many years later that boy grew up and became an engineer he returned to the village and built a long pipe from the river to the village the old man was very happy and the village never carried buckets again the engineer said the hardest part was not building the pipe it was remembering why we needed it the old man nodded and said that is wisdom the engineer then asked the old man what will you do now the old man replied i will sit by the river and watch the water flow the engineer smiled and said that sounds peaceful can i join you the old man said of course you can you are always welcome the two sat by the river for a long time watching the water and the clouds the engineer thought about his childhood and how he had asked the same question many years ago he realized that the old man had taught him something more important than building pipes he had taught him patience and kindness the engineer decided to stay in the village and build more things a school a bridge a small hospital the villagers were grateful and the old man became a legend the story ofthe man andthe boy was told to every new generationthe moral ofthe story is that small acts of kindness can lead to great changes but only if we remember why we startedthe journeythe boy who became an engineer never forgotthe old man's smile on that first morning bythe river that memory guided him through every difficult project years later whenthe engineer was old himself he toldthe same story to a young girl who asked why do you build so many things he answered because someone once carried water for me and i want to carry water for others the girl thought for a moment then said that is_the best reason i have ever heard")


print("\nTraining corpus (250 words):\n")
print(corpus)
print("\n" + "-" * 70)

tokens = corpus.split()
print(f"\nTotal tokens: {len(tokens)}")
print(f"Unique words: {len(set(tokens))}\n")

# ------------------------------------------------------------------
# Build vocabulary and training data (context size = 4 for better context)
# ------------------------------------------------------------------
context_size = 4  # Slightly larger context for richer patterns
vocab = sorted(list(set(tokens)))
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for i, w in enumerate(vocab)}
vocab_size = len(vocab)

X_train, y_train = [], []
for i in range(len(tokens) - context_size):
    ctx = [word2idx[w] for w in tokens[i:i+context_size]]
    tgt = word2idx[tokens[i+context_size]]
    X_train.append(ctx)
    y_train.append(tgt)

X_tensor = torch.tensor(X_train, dtype=torch.long)
y_tensor = torch.tensor(y_train, dtype=torch.long)
print(f"Training samples: {len(X_train)}")
if X_train:
    print(f"Example: {[idx2word[i] for i in X_train[0]]} -> '{idx2word[y_train[0]]}'\n")

# ------------------------------------------------------------------
# Model definitions (same as before, but with embed_dim=16 for richer representation)
# ------------------------------------------------------------------
class RNNLM(nn.Module):
    def __init__(self, vocab_size, embed_dim=16, hidden_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.rnn(emb)
        return self.fc(out[:, -1, :])

class LSTMLM(nn.Module):
    def __init__(self, vocab_size, embed_dim=16, hidden_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.lstm(emb)
        return self.fc(out[:, -1, :])

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, embed_dim=16, nhead=2, hidden_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = nn.Parameter(torch.zeros(1, context_size, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=hidden_dim,
            batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)  # 2 layers for better long-range
        self.fc = nn.Linear(embed_dim, vocab_size)
    def forward(self, x):
        emb = self.embedding(x) + self.pos_encoder[:, :x.size(1), :]
        out = self.transformer(emb)
        return self.fc(out[:, -1, :])

# ------------------------------------------------------------------
# Train all three models
# ------------------------------------------------------------------
def train_model(model, name, epochs=200, lr=0.005):
    print(f"Training {name}...")
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(X_tensor)
        loss = criterion(out, y_tensor)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 50 == 0:
            print(f"  {name} Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    return model

rnn = train_model(RNNLM(vocab_size), "RNN")
lstm = train_model(LSTMLM(vocab_size), "LSTM")
trans = train_model(TransformerLM(vocab_size), "Transformer")
print("\nAll three models trained successfully.\n")

# ------------------------------------------------------------------
# Autoregressive generation function
# ------------------------------------------------------------------
def generate(model, start_words, num_words=20):
    current = start_words.copy()
    for _ in range(num_words):
        window = current[-context_size:] if len(current) >= context_size else current
        indices = [word2idx.get(w, 0) for w in window]
        while len(indices) < context_size:
            indices.insert(0, 0)
        inp = torch.tensor([indices], dtype=torch.long)
        with torch.no_grad():
            pred_idx = model(inp).argmax().item()
            current.append(idx2word[pred_idx])
    return " ".join(current)

# ------------------------------------------------------------------
# Interactive generation
# ------------------------------------------------------------------
print("--- TEXT GENERATION FROM EACH MODEL ---")
seed = input("Enter seed phrase (e.g., 'the old man'): ").strip()
if not seed:
    seed = "the old man"
try:
    num = int(input("Number of words to generate: ").strip())
except:
    num = 20

print("\n" + "=" * 70)
print(f"Seed: '{seed}'\n")
print(f"RNN        : {generate(rnn, seed.split(), num)}")
print(f"LSTM       : {generate(lstm, seed.split(), num)}")
print(f"Transformer: {generate(trans, seed.split(), num)}")
print("\n" + "=" * 70)
print("Note: The Transformer often produces the most coherent continuation,")
print("while RNN might repeat or lose the story thread. LSTM sits in between.")