import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict, Counter
import random

# Set random seed for deterministic training so the outputs match the slides exactly
torch.manual_seed(42)
random.seed(42)

# நமது Language Models-க்கான எளிய ஆங்கில வாக்கியம் (Corpus)
text = "i like to eat apples and i like to eat bananas but i do not like to eat grapes"
tokens = text.split()

print("--- The Problem: Language Modeling ---")
print("Goal: Predict the next word given a sequence of previous words.\n")
print(f"Corpus: '{text}'\n")

# ==========================================
# 1-Gram (Unigram) Model
# Context length: 0 வார்த்தைகள்
# வார்த்தை எவ்வளவு முறை வந்துள்ளது என்பதை வைத்து மட்டும் கணிக்கிறது
# ==========================================
class UnigramModel:
    def __init__(self, corpus_tokens):
        self.counts = Counter(corpus_tokens)
        self.total_tokens = len(corpus_tokens)
        
    def predict_next(self):
        # ஒரு 1-gram model-இல் கணிப்பு என்பது ஒட்டுமொத்தமாக அதிகம் பயன்படுத்தப்பட்ட வார்த்தையைத் தேர்ந்தெடுப்பதே ஆகும்.
        most_common = self.counts.most_common(1)[0]
        return most_common[0]

unigram = UnigramModel(tokens)
print("1. Unigram Model (Context = 0)")
print("Input: 'i like to eat'")
# கவனிக்கவும், predict function-க்கு நாம் input-ஐ அனுப்பவே இல்லை, ஏனெனில் இது context-ஐப் பார்ப்பதில்லை!
print(f"Prediction: '{unigram.predict_next()}' (It just guesses the most frequent word in the corpus)")
print("-" * 40)


# ==========================================
# 2-Gram (Bigram) Model
# Context length: 1 வார்த்தை
# கடைசியாக வந்த வார்த்தைக்குப் பின் எது வந்தது என்பதை வைத்து கணிக்கிறது
# ==========================================
class BigramModel:
    def __init__(self, corpus_tokens):
        self.bigram_counts = defaultdict(Counter)
        for i in range(len(corpus_tokens) - 1):
            current_word = corpus_tokens[i]
            next_word = corpus_tokens[i+1]
            self.bigram_counts[current_word][next_word] += 1
            
    def predict_next(self, current_word):
        if current_word not in self.bigram_counts:
            return "<UNKNOWN>"
        # current_word-க்கு அடுத்து அதிகம் வரும் வார்த்தையைத் திருப்பியனுப்புகிறது
        return self.bigram_counts[current_word].most_common(1)[0][0]

bigram = BigramModel(tokens)
print("2. Bigram Model (Context = 1)")
print("Input: 'i like to eat'")
# நாம் "eat" என்ற கடைசி வார்த்தையை மட்டுமே அனுப்புகிறோம்
last_word = "eat"
print(f"Prediction after '{last_word}': '{bigram.predict_next(last_word)}'")
print("-" * 40)


# ==========================================
# N-Gram Model
# Context length: N-1 வார்த்தைகள்
# N-1 வார்த்தைகளின் சரியான வரிசைக்குப் பின் எது வந்தது என்பதை வைத்து கணிக்கிறது
# ==========================================
class NgramModel:
    def __init__(self, corpus_tokens, n):
        self.n = n
        self.context_counts = defaultdict(Counter)
        
        # நமக்கு Context-ஆக N-1 வார்த்தைகள் தேவை
        context_len = n - 1
        for i in range(len(corpus_tokens) - context_len):
            context = tuple(corpus_tokens[i : i + context_len])
            next_word = corpus_tokens[i + context_len]
            self.context_counts[context][next_word] += 1
            
    def predict_next(self, context_list):
        # கடைசி N-1 வார்த்தைகளை மட்டும் எடுத்துக்கொள்கிறது
        context_tuple = tuple(context_list[-(self.n - 1):])
        if context_tuple not in self.context_counts:
            return "<UNKNOWN CONTEXT>"
        return self.context_counts[context_tuple].most_common(1)[0][0]

# 4-gram model-ஐ (Context = 3) சோதித்துப் பார்ப்போம்
four_gram = NgramModel(tokens, n=4)
print("3. N-gram Model (N=4, Context = 3)")
input_sequence = ["i", "like", "to"]
print(f"Input: '{' '.join(input_sequence)}'")
print(f"Prediction: '{four_gram.predict_next(input_sequence)}'")
print("-" * 40)

print("THE N-GRAM WALL:")
# இந்தச் சரியான வார்த்தை வரிசை corpus-இல் இல்லை
bad_input = ["and", "i", "do"]
print(f"Input: '{' '.join(bad_input)}'")
print(f"Prediction: '{four_gram.predict_next(bad_input)}'")
print("Why did it fail? Because the EXACT 3-word sequence ('and', 'i', 'do') never appeared in the corpus!")
print("This is the Curse of Dimensionality. We need models that can generalize context (RNNs) instead of just counting exact matches.")

# ==========================================
# Neural தீர்வு: RNN & LSTM
# ==========================================
print("\n" + "=" * 40)
print("4. The Neural Solution: RNN & LSTM Language Models")
print("=" * 40)
print("Instead of counting exact matches, Neural Networks learn to compress the entire past context into a 'Hidden State' vector.\n")

# Vocabulary-ஐ உருவாக்குதல்
vocab = list(set(tokens))
word2idx = {word: i for i, word in enumerate(vocab)}
idx2word = {i: word for i, word in enumerate(vocab)}
vocab_size = len(vocab)

# Training data-வை தயார் செய்தல்
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

class RNNLM(nn.Module):
    def __init__(self, vocab_size, embed_dim=10, hidden_dim=16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x):
        embedded = self.embedding(x)
        out, hidden = self.rnn(embedded)
        # Context-இல் உள்ள கடைசி வார்த்தையின் output-ஐ வைத்து கணிக்கிறது
        prediction = self.fc(out[:, -1, :])
        return prediction

class LSTMLM(nn.Module):
    def __init__(self, vocab_size, embed_dim=10, hidden_dim=16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x):
        embedded = self.embedding(x)
        out, (hidden, cell) = self.lstm(embedded)
        prediction = self.fc(out[:, -1, :])
        return prediction

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, embed_dim=16, hidden_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # Simple learnable positional encoding
        self.pos_encoder = nn.Parameter(torch.zeros(1, context_size, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=2, dim_feedforward=hidden_dim, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.fc = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, x):
        embedded = self.embedding(x) + self.pos_encoder[:, :x.size(1), :]
        out = self.transformer(embedded)
        prediction = self.fc(out[:, -1, :])
        return prediction

rnn_model = RNNLM(vocab_size)
lstm_model = LSTMLM(vocab_size)
transformer_model = TransformerLM(vocab_size)

criterion = nn.CrossEntropyLoss()
optimizer_rnn = optim.Adam(rnn_model.parameters(), lr=0.01)
optimizer_lstm = optim.Adam(lstm_model.parameters(), lr=0.01)
optimizer_transformer = optim.Adam(transformer_model.parameters(), lr=0.01)

print("Training RNN, LSTM, and Transformer Language Models...")
# 150 epochs-க்கு train செய்தல்
for epoch in range(150):
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

print("--- Testing the N-Gram Wall Sequence ---")
test_context = ["and", "i", "do"]
print(f"Context: '{' '.join(test_context)}'")

# வார்த்தைகளை index எண்களாக மாற்றுதல், தெரியாத வார்த்தைகளுக்கு 0-ஐப் பயன்படுத்துதல்
test_tensor = torch.tensor([[word2idx.get(w, 0) for w in test_context]], dtype=torch.long)

with torch.no_grad():
    rnn_pred_idx = rnn_model(test_tensor).argmax().item()
    lstm_pred_idx = lstm_model(test_tensor).argmax().item()

print(f"N-Gram Model Prediction: '{four_gram.predict_next(test_context)}'")
print(f"RNN Model Prediction:    '{idx2word[rnn_pred_idx]}'")
print(f"LSTM Model Prediction:   '{idx2word[lstm_pred_idx]}'")
print("\nNotice how the Neural Networks still make a prediction! They don't rely on exact matches, they generalize the context through their hidden states.")

# ==========================================
# 5. The Magic of Generation (Autoregressive)
# ==========================================
print("\n" + "=" * 40)
print("5. The Magic of Generation (Autoregressive)")
print("=" * 40)
print("By feeding the predicted word back into the model as the next input context,")
print("we can generate entirely new sentences. This is how ChatGPT works!\n")

def generate_text(model, start_words, num_words_to_generate=5):
    # நமது தற்போதைய Context-ஐ copy செய்து கொள்கிறோம்
    current_context = start_words.copy()
    
    for _ in range(num_words_to_generate):
        # கடைசியாக உள்ள 'context_size' வார்த்தைகளை மட்டும் எடுக்கிறோம்
        window = current_context[-context_size:] if len(current_context) >= context_size else current_context
        
        # Context size-ஐ விட குறைவாக இருந்தால், 0-ஐ (padding) சேர்த்து நிரப்புகிறோம்
        padded_window = [word2idx.get(w, 0) for w in window]
        while len(padded_window) < context_size:
            padded_window.insert(0, 0)
            
        tensor_input = torch.tensor([padded_window], dtype=torch.long)
        with torch.no_grad():
            prediction_idx = model(tensor_input).argmax().item()
            predicted_word = idx2word[prediction_idx]
            # கணித்த வார்த்தையை மீண்டும் input context-ல் சேர்க்கிறோம்!
            current_context.append(predicted_word)
            
    return " ".join(current_context)

seed_text = ["i", "like"]
print(f"Seed Context: '{' '.join(seed_text)}'")
print(f"RNN Generates:         '{generate_text(rnn_model, seed_text, 8)}'")
print(f"LSTM Generates:        '{generate_text(lstm_model, seed_text, 8)}'")
print(f"Transformer Generates: '{generate_text(transformer_model, seed_text, 8)}'")
print("\nThe model 'hallucinates' or creates new paths based on what it learned!")
