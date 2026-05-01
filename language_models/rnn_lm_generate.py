import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter

# ------------------------------------------------------------
print("=" * 60)
print("RNN LANGUAGE MODEL - WITH TEXT GENERATION")
print("=" * 60)
print("After training, the model can generate new sentences autoregressively.\n")

# ------------------------------------------------------------
# 1. GET TRAINING CORPUS
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
# 3. CREATE TRAINING SAMPLES (context_size = 3)
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
print(f"Created {len(X_train)} training examples.\n")

# ------------------------------------------------------------
# 4. DEFINE RNN MODEL
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# 5. TRAIN THE MODEL
# ------------------------------------------------------------
model = RNNLM(vocab_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

print("Training RNN Language Model...")
epochs = 150
for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(X_tensor)
    loss = criterion(output, y_tensor)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 50 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
print("Training complete!\n")

# ------------------------------------------------------------
# 6. GENERATION FUNCTION (autoregressive)
# ------------------------------------------------------------
def generate_text(model, start_words, num_words_to_generate=8):
    current_context = start_words.copy()
    for _ in range(num_words_to_generate):
        # Take last 'context_size' words
        window = current_context[-context_size:] if len(current_context) >= context_size else current_context
        # Convert to indices, pad if needed
        indices = [word2idx.get(w, 0) for w in window]
        while len(indices) < context_size:
            indices.insert(0, 0)
        tensor_input = torch.tensor([indices], dtype=torch.long)
        with torch.no_grad():
            pred_idx = model(tensor_input).argmax().item()
            pred_word = idx2word[pred_idx]
            current_context.append(pred_word)
    return " ".join(current_context)

# ------------------------------------------------------------
# 7. USER INPUT FOR GENERATION
# ------------------------------------------------------------
print("--- Text Generation ---")
seed_phrase = input("Enter seed phrase (at least 1 word): ").strip()
if not seed_phrase:
    seed_phrase = "i like"
    print(f"Using default: '{seed_phrase}'")
seed_words = seed_phrase.split()

try:
    num_words = int(input("Number of words to generate: ").strip())
except:
    num_words = 8

generated = generate_text(model, seed_words, num_words)
print(f"\nGenerated text: '{generated}'")
print("\nNote: The RNN generates by feeding its own predictions back as context.")