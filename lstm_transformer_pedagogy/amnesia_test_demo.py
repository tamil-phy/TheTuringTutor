import torch
import torch.nn as nn
import torch.optim as optim
import string
import random
import sys
import time

print("=" * 65)
print("THE AMNESIA TEST: LSTM vs Transformer (Context Memory)")
print("=" * 65)

# --- 1. The Transparent Corpus ---
print("--- 1. The Training Corpus ---")
print("We are building a transparent corpus of 5,000 text strings.")
print("Each string is exactly 20 characters long.")
print("The Goal: Predict the VERY FIRST letter of the string.")

alphabet = list(string.ascii_uppercase)
vocab_size = len(alphabet)
char_to_idx = {char: idx for idx, char in enumerate(alphabet)}
idx_to_char = {idx: char for idx, char in enumerate(alphabet)}

SEQ_LEN = 20
NUM_TRAIN = 5000

def generate_corpus(num_samples):
    X_str = []
    y_str = []
    X_tensor = torch.zeros(num_samples, SEQ_LEN, dtype=torch.long)
    y_tensor = torch.zeros(num_samples, dtype=torch.long)
    
    for i in range(num_samples):
        # Generate 20 random uppercase letters
        seq = [random.choice(alphabet) for _ in range(SEQ_LEN)]
        target = seq[0] # The target is always the first letter
        
        X_str.append(" ".join(seq))
        y_str.append(target)
        
        X_tensor[i] = torch.tensor([char_to_idx[char] for char in seq])
        y_tensor[i] = char_to_idx[target]
        
    return X_str, y_str, X_tensor, y_tensor

X_str_train, y_str_train, X_train, y_train = generate_corpus(NUM_TRAIN)

print(f"\nExample 1: '{X_str_train[0]}' -> Target: '{y_str_train[0]}'")
print(f"Example 2: '{X_str_train[1]}' -> Target: '{y_str_train[1]}'")
print(f"Example 3: '{X_str_train[1200]}' -> Target: '{y_str_train[1200]}'\n")

# --- 2. PyTorch Architectures ---
class LSTM_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 16)
        self.lstm = nn.LSTM(16, 32, batch_first=True)
        self.fc = nn.Linear(32, vocab_size)
        
    def forward(self, x):
        emb = self.embedding(x)
        out, (hidden, cell) = self.lstm(emb)
        return self.fc(hidden.squeeze(0)) # Use final hidden state

class Transformer_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 16)
        self.pos_embedding = nn.Embedding(100, 16)
        
        # We use MultiheadAttention directly so we can extract the Attention Weights for the demo!
        self.attention = nn.MultiheadAttention(embed_dim=16, num_heads=1, batch_first=True)
        self.fc = nn.Linear(16, vocab_size)
        
    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(x.size(0), -1)
        emb = self.embedding(x) + self.pos_embedding(positions)
        
        attn_output, attn_weights = self.attention(emb, emb, emb)
        
        # Return the prediction for the LAST token, AND the attention weights
        return self.fc(attn_output[:, -1, :]), attn_weights

import warnings
warnings.filterwarnings("ignore") # Clean presentation

# --- 3. Live Training ---
print("--- 2. Live Training (No Messy Math) ---")
lstm_model = LSTM_Model()
trans_model = Transformer_Model()

criterion = nn.CrossEntropyLoss()
opt_lstm = optim.Adam(lstm_model.parameters(), lr=0.01)
opt_trans = optim.Adam(trans_model.parameters(), lr=0.01)

batch_size = 200
epochs = 30

def train_lstm():
    sys.stdout.write(f"Training LSTM on corpus... [")
    sys.stdout.flush()
    for epoch in range(epochs):
        for i in range(0, NUM_TRAIN, batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            
            opt_lstm.zero_grad()
            pred = lstm_model(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            opt_lstm.step()
        
        if (epoch + 1) % max(1, epochs // 20) == 0:
            sys.stdout.write("=")
            sys.stdout.flush()
    sys.stdout.write("] 100% Done\n")

def train_transformer_visually():
    print("\nTraining Transformer... (Watch the Attention Shortcut form!)")
    
    # Check untrained state BEFORE training starts
    _, attn_weights = trans_model(X_train[0:1])
    weight = attn_weights[0, -1, 0].item() * 100
    print(f"  [Untrained] Last Token looks at First Token: {weight:5.1f}% (Random guessing...)")
    
    for epoch in range(epochs):
        for i in range(0, NUM_TRAIN, batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            
            opt_trans.zero_grad()
            pred, _ = trans_model(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            opt_trans.step()
        
        # Visually extract and print the attention weight at specific epochs!
        if epoch == 14:
            _, attn_weights = trans_model(X_train[0:1])
            weight = attn_weights[0, -1, 0].item() * 100
            print(f"  [Epoch 15]  Last Token looks at First Token: {weight:5.1f}% (The math shortcut is forming!)")
        elif epoch == 29:
            _, attn_weights = trans_model(X_train[0:1])
            weight = attn_weights[0, -1, 0].item() * 100
            print(f"  [Epoch 30]  Last Token looks at First Token: {weight:5.1f}% (The O(1) Shortcut is built!)")

train_lstm()
train_transformer_visually()
print()

# --- 4. The Live Test ---
print("--- 3. The Live Test ---")
test_X_str, test_y_str, test_X, test_y = generate_corpus(1)

print(f"Unseen Test String: '{test_X_str[0]}'")
print(f"Target Answer:      '{test_y_str[0]}'")
print("-" * 30)

with torch.no_grad():
    lstm_pred_idx = lstm_model(test_X).argmax(dim=1).item()
    trans_pred_logits, _ = trans_model(test_X)
    trans_pred_idx = trans_pred_logits.argmax(dim=1).item()

lstm_pred_char = idx_to_char[lstm_pred_idx]
trans_pred_char = idx_to_char[trans_pred_idx]

print(f"LSTM Predicted:        '{lstm_pred_char}' ", end="")
if lstm_pred_char == test_y_str[0]:
    print("(Surprising Success!)")
else:
    print("(Failed! The Vanishing Gradient destroyed the memory of the first letter)")

print(f"Transformer Predicted: '{trans_pred_char}' ", end="")
if trans_pred_char == test_y_str[0]:
    print("(Success! Attention instantly retrieved the first letter)")
else:
    print("(Failed!)")
print()
