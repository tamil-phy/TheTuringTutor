import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict, Counter
import random

# ------------------------------------------------------------------
# 1. GET TRAINING CORPUS
# ------------------------------------------------------------------
print("=" * 70)
print("ALL-IN-ONE LANGUAGE MODEL DEMO (Unigram, Bigram, N-gram, RNN, LSTM, Transformer)")
print("=" * 70)

corpus = input("\nEnter your training corpus (sentence / text): ").strip()
if not corpus:
    corpus = "i like to eat apples and i like to eat bananas but i do not like to eat grapes"
    print(f"Using default corpus: '{corpus}'")

tokens = corpus.split()
print(f"\nCorpus tokens: {tokens}\n")

# ------------------------------------------------------------------
# 2. BUILD VOCABULARY (for neural models)
# ------------------------------------------------------------------
vocab = sorted(list(set(tokens)))
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for i, w in enumerate(vocab)}
vocab_size = len(vocab)
print(f"Vocabulary ({vocab_size} unique words): {vocab}\n")

# Prepare training data for neural models (context size = 3)
context_size = 3
X_train, y_train = [], []
for i in range(len(tokens) - context_size):
    ctx = [word2idx[w] for w in tokens[i:i+context_size]]
    tgt = word2idx[tokens[i+context_size]]
    X_train.append(ctx)
    y_train.append(tgt)

X_tensor = torch.tensor(X_train, dtype=torch.long)
y_tensor = torch.tensor(y_train, dtype=torch.long)
print(f"Neural training samples: {len(X_train)} (context size = {context_size})\n")

# ------------------------------------------------------------------
# 3. TRADITIONAL N-GRAM MODELS
# ------------------------------------------------------------------
class UnigramModel:
    def __init__(self, tokens):
        self.counter = Counter(tokens)
    def predict(self, context=None):
        return self.counter.most_common(1)[0][0]

class BigramModel:
    def __init__(self, tokens):
        self.counts = defaultdict(Counter)
        for i in range(len(tokens)-1):
            self.counts[tokens[i]][tokens[i+1]] += 1
    def predict(self, last_word):
        if last_word not in self.counts:
            return "<UNKNOWN>"
        return self.counts[last_word].most_common(1)[0][0]

class NgramModel:
    def __init__(self, tokens, n):
        self.n = n
        self.counts = defaultdict(Counter)
        ctx_len = n-1
        for i in range(len(tokens)-ctx_len):
            ctx = tuple(tokens[i:i+ctx_len])
            nxt = tokens[i+ctx_len]
            self.counts[ctx][nxt] += 1
    def predict(self, context_list):
        ctx = tuple(context_list[-(self.n-1):])
        if ctx not in self.counts:
            return "<UNKNOWN CONTEXT>"
        return self.counts[ctx].most_common(1)[0][0]

# ------------------------------------------------------------------
# 4. NEURAL MODELS (RNN, LSTM, Transformer)
# ------------------------------------------------------------------
class RNNLM(nn.Module):
    def __init__(self, vocab_size, embed_dim=10, hidden_dim=16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    def forward(self, x):
        emb = self.embed(x)
        out, _ = self.rnn(emb)
        return self.fc(out[:, -1, :])

class LSTMLM(nn.Module):
    def __init__(self, vocab_size, embed_dim=10, hidden_dim=16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    def forward(self, x):
        emb = self.embed(x)
        out, _ = self.lstm(emb)
        return self.fc(out[:, -1, :])

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, embed_dim=16, nhead=2, hidden_dim=32, num_layers=1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_enc = nn.Parameter(torch.zeros(1, context_size, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead,
                                                   dim_feedforward=hidden_dim,
                                                   batch_first=True, dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, vocab_size)
    def forward(self, x):
        emb = self.embed(x) + self.pos_enc[:, :x.size(1), :]
        out = self.transformer(emb)
        return self.fc(out[:, -1, :])

def train_neural_model(model_class, model_name, epochs=150, lr=0.01):
    model = model_class(vocab_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(X_tensor)
        loss = criterion(out, y_tensor)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 50 == 0:
            print(f"  {model_name} Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    return model

# ------------------------------------------------------------------
# 5. USER INTERACTION – CHOOSE MODEL & MODE
# ------------------------------------------------------------------
print("\n--- AVAILABLE MODELS ---")
print("1. Unigram (always most frequent word)")
print("2. Bigram (based on last word only)")
print("3. N-gram (user-defined N, based on last N-1 words)")
print("4. RNN (neural, context=3 words)")
print("5. LSTM (neural, context=3 words)")
print("6. Transformer (neural, context=3 words)")
print("7. Compare all models on a single prediction")

choice = input("\nSelect model (1-7): ").strip()

# Build traditional models (always cheap)
unigram = UnigramModel(tokens)
bigram = BigramModel(tokens)

# For N-gram, ask for N
ngram_n = 3
if choice in ['3', '7']:
    try:
        ngram_n = int(input("Enter N for N-gram model (>=2): ").strip())
        if ngram_n < 2:
            ngram_n = 2
    except:
        ngram_n = 3
ngram = NgramModel(tokens, ngram_n)

# Neural models: train only if needed
rnn_model = None
lstm_model = None
transformer_model = None
neural_trained = False

if choice in ['4', '5', '6', '7']:
    print("\nTraining neural models (this may take ~10-20 seconds)...")
    if choice in ['4', '7']:
        rnn_model = train_neural_model(RNNLM, "RNN")
    if choice in ['5', '7']:
        lstm_model = train_neural_model(LSTMLM, "LSTM")
    if choice in ['6', '7']:
        transformer_model = train_neural_model(TransformerLM, "Transformer")
    neural_trained = True
    print("Neural training complete.\n")

# ------------------------------------------------------------------
# 6. PREDICTION MODE (single next word)
# ------------------------------------------------------------------
def get_user_context():
    phrase = input("Enter context phrase: ").strip()
    if not phrase:
        phrase = "i like to"
        print(f"Using default: '{phrase}'")
    return phrase.split()

def predict_unigram():
    return unigram.predict()

def predict_bigram(words):
    if not words:
        return "<need last word>"
    return bigram.predict(words[-1])

def predict_ngram(words):
    if len(words) < ngram_n-1:
        return f"<need at least {ngram_n-1} words>"
    return ngram.predict(words)

def predict_neural(model, words, model_name):
    if len(words) < context_size:
        # pad left with first vocab word (index 0)
        padded = [vocab[0]] * (context_size - len(words)) + words[-context_size:]
    else:
        padded = words[-context_size:]
    indices = [word2idx.get(w, 0) for w in padded]
    tensor_input = torch.tensor([indices], dtype=torch.long)
    with torch.no_grad():
        pred_idx = model(tensor_input).argmax().item()
    return idx2word[pred_idx]

if choice in ['1','2','3','4','5','6']:
    # Single prediction
    context_words = get_user_context()
    print(f"\nContext: '{' '.join(context_words)}'")
    if choice == '1':
        print(f"Unigram prediction: '{predict_unigram()}'")
    elif choice == '2':
        last = context_words[-1] if context_words else ""
        print(f"Bigram prediction (last word='{last}'): '{predict_bigram(context_words)}'")
    elif choice == '3':
        print(f"{ngram_n}-gram prediction: '{predict_ngram(context_words)}'")
    elif choice == '4' and rnn_model:
        print(f"RNN prediction: '{predict_neural(rnn_model, context_words, 'RNN')}'")
    elif choice == '5' and lstm_model:
        print(f"LSTM prediction: '{predict_neural(lstm_model, context_words, 'LSTM')}'")
    elif choice == '6' and transformer_model:
        print(f"Transformer prediction: '{predict_neural(transformer_model, context_words, 'Transformer')}'")
    else:
        print("Model not available. Train it first.")

elif choice == '7':
    # Compare all models
    context_words = get_user_context()
    print(f"\nContext: '{' '.join(context_words)}'")
    print("-" * 50)
    print(f"Unigram:                    '{predict_unigram()}'")
    print(f"Bigram (last word only):    '{predict_bigram(context_words)}'")
    print(f"{ngram_n}-gram:              '{predict_ngram(context_words)}'")
    if rnn_model:
        print(f"RNN:                        '{predict_neural(rnn_model, context_words, 'RNN')}'")
    if lstm_model:
        print(f"LSTM:                       '{predict_neural(lstm_model, context_words, 'LSTM')}'")
    if transformer_model:
        print(f"Transformer:                '{predict_neural(transformer_model, context_words, 'Transformer')}'")
    if not (rnn_model or lstm_model or transformer_model):
        print("(No neural models were trained.)")
    print("-" * 50)
    print("\nNeural networks generalize even when the exact word sequence never appeared.\n")

# ------------------------------------------------------------------
# 7. AUTOREGRESSIVE GENERATION (only for neural models if available)
# ------------------------------------------------------------------
if neural_trained and choice in ['4','5','6','7']:
    print("\n--- TEXT GENERATION (autoregressive) ---")
    gen_choice = input("Do you want to generate a continuation? (y/n): ").strip().lower()
    if gen_choice == 'y':
        seed = input("Enter seed phrase (at least 1 word): ").strip()
        if not seed:
            seed = "i like"
        try:
            gen_len = int(input("How many words to generate? (default 8): ").strip())
        except:
            gen_len = 8

        seed_words = seed.split()
        # Pick which neural model to use for generation
        print("\nWhich model to generate with?")
        if rnn_model: print("  rnn")
        if lstm_model: print("  lstm")
        if transformer_model: print("  transformer")
        gen_model_name = input("Enter choice: ").strip().lower()

        gen_model = None
        if gen_model_name == 'rnn' and rnn_model:
            gen_model = rnn_model
        elif gen_model_name == 'lstm' and lstm_model:
            gen_model = lstm_model
        elif gen_model_name == 'transformer' and transformer_model:
            gen_model = transformer_model
        else:
            # fallback to first available
            gen_model = rnn_model or lstm_model or transformer_model
            if gen_model:
                print(f"Using {gen_model.__class__.__name__}")

        if gen_model:
            def generate(model, start_words, length):
                current = start_words.copy()
                for _ in range(length):
                    # take last context_size words
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

            generated = generate(gen_model, seed_words, gen_len)
            print(f"\nGenerated text: '{generated}'")
        else:
            print("No neural model available for generation.")

print("\nThank you for using the All-in-One Language Model demo!")