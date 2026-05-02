import torch
import torch.nn as nn
import torch.optim as optim
import random
import sys
import time

# --- ANSI Color Codes ---
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    ENDC = '\033[0m'

def print_header(text):
    print(f"\n{Colors.HEADER}{Colors.BOLD}╔{'═'*78}╗{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}║ {text.center(76)} ║{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}╚{'═'*78}╝{Colors.ENDC}\n")

def pause_for_audience():
    input(f"\n{Colors.BOLD}[Press Enter to continue...]{Colors.ENDC}")
    print("\033[A\033[K\r", end="")

print_header("THE INFERENCE TEST: Long-Term Context Dependency")

# --- 1. The Inference Corpus ---
print("--- 1. The Inference Corpus ---")
print("Unlike the 'Amnesia Test' (which was a pure copy task), this is an INFERENCE task.")
print("The model must deduce the missing word at the end of a sentence based on a clue")
print("hidden at the very beginning of the sentence.\n")

# Vocabulary Data
country_language_pairs = {
    "France": "French",
    "Japan": "Japanese",
    "Spain": "Spanish",
    "Germany": "German",
    "Italy": "Italian",
    "China": "Chinese",
    "Brazil": "Portuguese",
    "Tamil Nadu": "Tamil",
    "Russia": "Russian",
    "Greece": "Greek"
}

fillers = [
    "weather", "food", "travel", "music", "art", "history", "culture", "people", 
    "cities", "nature", "architecture", "sports", "fashion", "technology", "literature",
    "is", "very", "beautiful", "and", "the", "amazing", "so", "I", "decided", "to", "learn"
]

# Build Vocab
vocab = list(country_language_pairs.keys()) + list(country_language_pairs.values()) + fillers
vocab_size = len(vocab)
word_to_idx = {word: idx for idx, word in enumerate(vocab)}
idx_to_word = {idx: word for idx, word in enumerate(vocab)}

SEQ_LEN = 75 # 1 Country + 74 Fillers
NUM_TRAIN = 2000

def generate_corpus(num_samples):
    X_str = []
    y_str = []
    X_tensor = torch.zeros(num_samples, SEQ_LEN, dtype=torch.long)
    y_tensor = torch.zeros(num_samples, dtype=torch.long)
    
    countries = list(country_language_pairs.keys())
    
    for i in range(num_samples):
        country = random.choice(countries)
        language = country_language_pairs[country]
        
        # Sequence: [Country] + [random filler words]
        seq = [country] + [random.choice(fillers) for _ in range(SEQ_LEN - 1)]
        
        X_str.append(" ".join(seq))
        y_str.append(language)
        
        X_tensor[i] = torch.tensor([word_to_idx[word] for word in seq])
        y_tensor[i] = word_to_idx[language]
        
    return X_str, y_str, X_tensor, y_tensor

X_str_train, y_str_train, X_train, y_train = generate_corpus(NUM_TRAIN)

print(f"{Colors.CYAN}Example 1:{Colors.ENDC} '{X_str_train[0]} [MASK]' -> Target: {Colors.GREEN}{y_str_train[0]}{Colors.ENDC}")
print(f"{Colors.CYAN}Example 2:{Colors.ENDC} '{X_str_train[1]} [MASK]' -> Target: {Colors.GREEN}{y_str_train[1]}{Colors.ENDC}")
print(f"{Colors.CYAN}Example 3:{Colors.ENDC} '{X_str_train[2]} [MASK]' -> Target: {Colors.GREEN}{y_str_train[2]}{Colors.ENDC}\n")

print("The model CANNOT just copy the first word. It must understand the relationship:")
print(f"Country -> Language. But the clue is {SEQ_LEN - 1} words away from the prediction point!")

#pause_for_audience()

# --- 2. PyTorch Architectures ---
class LSTM_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 32)
        self.lstm = nn.LSTM(32, 32, batch_first=True)
        self.fc = nn.Linear(32, vocab_size)
        
    def forward(self, x):
        emb = self.embedding(x)
        out, (hidden, cell) = self.lstm(emb)
        return self.fc(hidden.squeeze(0))

class Transformer_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 64)
        self.pos_embedding = nn.Embedding(200, 64)
        
        self.attention = nn.MultiheadAttention(embed_dim=64, num_heads=2, batch_first=True)
        self.fc = nn.Linear(64, vocab_size)
        
    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(x.size(0), -1)
        emb = self.embedding(x) + self.pos_embedding(positions)
        
        attn_output, attn_weights = self.attention(emb, emb, emb, average_attn_weights=False)
        return self.fc(attn_output[:, -1, :]), attn_weights

lstm_model = LSTM_Model()
trans_model = Transformer_Model()

criterion = nn.CrossEntropyLoss()
opt_lstm = optim.Adam(lstm_model.parameters(), lr=0.01)
opt_trans = optim.Adam(trans_model.parameters(), lr=0.01)

batch_size = 500
epochs = 40

# --- 3. Live Test Setup ---
test_X_str, test_y_str, test_X, test_y = generate_corpus(1)
test_country = next(c for c in country_language_pairs.keys() if test_X_str[0].startswith(c))

def evaluate_models(title, is_trained=False):
    print(f"\n{Colors.YELLOW}{title}{Colors.ENDC}")
    print(f"Unseen Context: '{test_X_str[0]} {Colors.BOLD}[MASK]{Colors.ENDC}'")
    print(f"Target Answer:  '{Colors.GREEN}{test_y_str[0]}{Colors.ENDC}' (Inferred from '{test_country}')")
    print("-" * 60)

    lstm_model.eval()
    trans_model.eval()

    with torch.no_grad():
        lstm_pred_idx = lstm_model(test_X).argmax(dim=1).item()
        trans_pred_logits, attn_weights = trans_model(test_X)
        trans_pred_idx = trans_pred_logits.argmax(dim=1).item()
        
        # attn_weights is (batch, num_heads, tgt_len, src_len) because average_attn_weights=False
        # We take the max attention paid by ANY head from the last token to the first token
        clue_attention = attn_weights[0, :, -1, 0].max().item() * 100

    lstm_pred_word = idx_to_word[lstm_pred_idx]
    trans_pred_word = idx_to_word[trans_pred_idx]

    # Evaluate LSTM
    print(f"{Colors.BOLD}LSTM Predicted:{Colors.ENDC}        '{lstm_pred_word}' ", end="")
    if lstm_pred_word == test_y_str[0]:
        if not is_trained:
            print(f"({Colors.GREEN}Lucky Guess! 1/10 chance{Colors.ENDC})")
        else:
            print(f"({Colors.GREEN}Success!{Colors.ENDC})")
    else:
        if not is_trained:
            print(f"({Colors.RED}Failed! Model has random weights{Colors.ENDC})")
        else:
            print(f"({Colors.RED}Failed! It forgot the clue due to Vanishing Gradients{Colors.ENDC})")

    # Evaluate Transformer
    print(f"{Colors.BOLD}Transformer Predicted:{Colors.ENDC} '{trans_pred_word}' ", end="")
    if trans_pred_word == test_y_str[0]:
        if not is_trained:
            print(f"({Colors.GREEN}Lucky Guess! 1/10 chance{Colors.ENDC})")
        else:
            print(f"({Colors.GREEN}Success!{Colors.ENDC})")
            print(f"   {Colors.CYAN}-> One of its attention heads paid {clue_attention:.1f}% of its focus directly to '{test_country}'!{Colors.ENDC}")
    else:
        if not is_trained:
            print(f"({Colors.RED}Failed! Model has random weights{Colors.ENDC})")
        else:
            print(f"({Colors.RED}Failed!{Colors.ENDC})")
    print()
    
    lstm_model.train()
    trans_model.train()

# --- Run Before Training Test ---
evaluate_models("BEFORE TRAINING: The Inference Test", is_trained=False)
#pause_for_audience()

# --- 4. Live Training ---
print(f"{Colors.YELLOW}--- Live Training ---{Colors.ENDC}")

def train_lstm():
    sys.stdout.write(f"Training LSTM on inference logic... [")
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
    print("\nTraining Transformer...")
    
    for epoch in range(epochs):
        for i in range(0, NUM_TRAIN, batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            
            opt_trans.zero_grad()
            pred, _ = trans_model(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            opt_trans.step()
        
        if epoch == 14:
            trans_model.eval()
            with torch.no_grad():
                _, attn_weights = trans_model(test_X)
            weight = attn_weights[0, :, -1, 0].max().item() * 100
            print(f"  [Epoch 15] Attention on clues forming... ({weight:5.1f}%)")
            trans_model.train()
        elif epoch == 39:
            trans_model.eval()
            with torch.no_grad():
                _, attn_weights = trans_model(test_X)
            weight = attn_weights[0, :, -1, 0].max().item() * 100
            print(f"  [Epoch 40] Attention firmly locked on the Country! ({weight:5.1f}%)")
            trans_model.train()

train_lstm()
train_transformer_visually()

#pause_for_audience()

# --- Run After Training Test ---
evaluate_models("AFTER TRAINING: The Inference Test", is_trained=True)
