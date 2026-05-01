import torch
import time
from lstm_vs_transformer_race import PurePythonLSTM, TinyTransformer, vocab
import math
import torch.nn.functional as F

batch_size = 64
seq_len = 1000
d_model = 256

massive_batch = torch.randn(batch_size, seq_len, d_model)

pure_lstm = PurePythonLSTM(d_model=d_model)

start = time.time()
pure_lstm(massive_batch)
print("LSTM:", time.time()-start)

start = time.time()
W_q = torch.nn.Linear(d_model, d_model, bias=False)
W_k = torch.nn.Linear(d_model, d_model, bias=False)
W_v = torch.nn.Linear(d_model, d_model, bias=False)

Q = W_q(massive_batch)
K = W_k(massive_batch)
V = W_v(massive_batch)
scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_model)
attn_weights = F.softmax(scores, dim=-1)
context = torch.matmul(attn_weights, V)
print("Transformer:", time.time()-start)
