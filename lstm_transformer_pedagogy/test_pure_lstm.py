import torch
import torch.nn as nn
import time
from lstm_vs_transformer_race import TinyTransformer

class PurePythonLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size=18):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        # LSTM weights
        self.W_ii = nn.Linear(hidden_size, hidden_size)
        self.W_hi = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_if = nn.Linear(hidden_size, hidden_size)
        self.W_hf = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_ig = nn.Linear(hidden_size, hidden_size)
        self.W_hg = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_io = nn.Linear(hidden_size, hidden_size)
        self.W_ho = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.vocab_layer = nn.Linear(hidden_size, vocab_size)
        self.hidden_size = hidden_size

    def forward(self, x):
        batch_size, seq_len = x.shape
        emb = self.embedding(x)
        
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        c = torch.zeros(batch_size, self.hidden_size, device=x.device)
        
        outputs = []
        for t in range(seq_len):
            x_t = emb[:, t, :]
            i_t = torch.sigmoid(self.W_ii(x_t) + self.W_hi(h))
            f_t = torch.sigmoid(self.W_if(x_t) + self.W_hf(h))
            g_t = torch.tanh(self.W_ig(x_t) + self.W_hg(h))
            o_t = torch.sigmoid(self.W_io(x_t) + self.W_ho(h))
            c = f_t * c + i_t * g_t
            h = o_t * torch.tanh(c)
            outputs.append(h.unsqueeze(1))
            
        out = torch.cat(outputs, dim=1)
        return self.vocab_layer(out)

vocab_size = 287
model_trans = TinyTransformer(vocab_size, seq_len=100)
model_lstm = PurePythonLSTM(vocab_size)

X = torch.randint(0, vocab_size, (64, 100))

start = time.time()
lstm_out = model_lstm(X)
lstm_loss = lstm_out.sum()
lstm_loss.backward()
lstm_time = time.time() - start

start = time.time()
trans_out = model_trans(X)
trans_loss = trans_out.sum()
trans_loss.backward()
trans_time = time.time() - start

print(f"Pure Python LSTM Time: {lstm_time:.4f}s")
print(f"Pure Python Transformer Time: {trans_time:.4f}s")
