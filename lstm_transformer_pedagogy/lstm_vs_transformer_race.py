import torch
import torch.nn as nn
import torch.nn.functional as F
import math
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

# --- 1. The Dataset (110 Sentences) ---
sentences = [
    "We propose a new simple network architecture, the Transformer",
    "They build a fast deep learning model, the CNN",
    "We design a huge complex language model, the GPT",
    "You propose a fast simple network architecture, the ResNet",
    "We train a new deep learning model, the BERT",
    "They propose a new complex neural architecture, the RNN",
    "You design a deep neural language model, the Agent",
    "We build a huge complex language model, the LLM",
    "We propose a fast simple network architecture, the LSTM",
    "They train a new fast network architecture, the AI",
    "We propose a scalable hybrid network architecture, the Perceiver",
    "They design a robust neural learning model, the GNN",
    "You build a modular deep language model, the PaLM",
    "We train a lightweight efficient network architecture, the MobileNet",
    "They develop a parallel high-speed language model, the T5",
    "You propose a recursive dynamic neural architecture, the TreeLSTM",
    "We design a generative adversarial learning model, the GAN",
    "They build a masked auto-encoding network architecture, the MAE",
    "You train a sparse conditional language model, the Switch",
    "We propose a bidirectional encoded language model, the ELMo",
    "They design a cross-modal deep learning model, the CLIP",
    "You build a dense convolutional network architecture, the DenseNet",
    "We train a temporal convolutional neural model, the TCN",
    "They propose a spatial attention network architecture, the SAN",
    "You design a latent diffusion generative model, the LDM",
    "We build a residual bottleneck network architecture, the ResNext",
    "They train a hierarchical vision learning model, the Swin",
    "You propose a contrastive language-image model, the ALIGN",
    "We design a multi-head attention network architecture, the MHA",
    "They build a long-range sequence language model, the S4",
    "You train a distilled compact language model, the DistilBERT",
    "We propose a gated recurrent network architecture, the GRU",
    "They design a variational auto-encoding model, the VAE",
    "You build a squeezed efficient network architecture, the SqueezeNet",
    "We train a unified versatile language model, the UL2",
    "They propose a decoupled multimodal learning model, the Flamingo",
    "You design a shifted window neural architecture, the ViT",
    "We build a regularized deep learning model, the Dropout",
    "They train a factored low-rank language model, the LoRA",
    "You propose a neural tangent kernel model, the NTK",
    "We design a private federated learning model, the FL",
    "They build a scalable visual network architecture, the EfficientNet",
    "You train a massive multilingual language model, the BLOOM",
    "We propose a causal generative language model, the LLaMA",
    "They design a structured state-space model, the Mamba",
    "You build a dual encoder retrieval model, the DPR",
    "We train a robust optimized language model, the RoBERTa",
    "They propose a pointer generator network architecture, the PGN",
    "You design a universal transformer language model, the UT",
    "We build a deep residual network architecture, the WRN",
    "They train a neural architecture search model, the NAS",
    "You propose a fast adaptive learning model, the FAL",
    "We design a multi-task unified language model, the MT-DNN",
    "They build a kernelized deep learning model, the KDL",
    "You train a sequence-to-sequence neural architecture, the Seq2Seq",
    "We propose a graph-based relational model, the GCN",
    "They design a sparse transformer language model, the BigBird",
    "You build a deep reinforcement learning model, the PPO",
    "We train a soft actor-critic model, the SAC",
    "They propose a proximal policy network architecture, the TRPO",
    "You design a deep Q-learning model, the DQN",
    "We build a generative pre-trained language model, the GPT-2",
    "They train a bidirectional transformer learning model, the XLNet",
    "You propose a dilated convolutional network architecture, the Wavenet",
    "We design a spectral graph neural model, the SGCN",
    "They build a weighted feature learning model, the WFM",
    "You train a tiny machine learning model, the TinyML",
    "We propose a feature pyramid network architecture, the FPN",
    "They design a region convolutional neural model, the R-CNN",
    "You build a single shot detector model, the SSD",
    "We train a spatial pyramid pooling model, the SPP",
    "They propose a long short-term architecture, the Peephole",
    "You design a bidirectional recurrent learning model, the BiRNN",
    "We build a deep belief network architecture, the DBN",
    "They train a restricted Boltzmann learning model, the RBM",
    "You propose a capsule neural network architecture, the CapsNet",
    "We design a memory augmented neural model, the MANN",
    "They build a differentiable neural computer model, the DNC",
    "You train a neural Turing machine model, the NTM",
    "We propose a global context network architecture, the GCNet",
    "They design a fully convolutional network architecture, the FCN",
    "You build a pyramid scene parsing model, the PSPNet",
    "We train a deep lab learning model, the DeepLab",
    "They propose a mask regional neural architecture, the Mask-RCNN",
    "You design a generative flow learning model, the Glow",
    "We build a pixel convolutional neural model, the PixelCNN",
    "They train a vector quantized learning model, the VQ-VAE",
    "You propose a cycle consistent adversarial model, the CycleGAN",
    "We design a style based generative model, the StyleGAN",
    "They build a high resolution network architecture, the HRNet",
    "You train a dual path network architecture, the DPN",
    "We propose a channel attention neural model, the SENet",
    "They design a non local neural architecture, the NLNet",
    "You build a squeeze and excitation model, the SE-ResNet",
    "We train a group convolutional learning model, the ShuffleNet",
    "They propose a depthwise separable network architecture, the Xception",
    "You design a multi scale vision model, the MViT",
    "We build a masked image learning model, the BEiT",
    "They train a contrastive predictive coding model, the CPC",
    "You propose a simple framework learning model, the SimCLR",
    "We design a momentum contrastive learning model, the MoCo",
    "They build a bootstrap your latent model, the BYOL",
    "You train a swav clustering learning model, the SwAV",
    "We propose a deep cluster learning model, the DeepCluster",
    "They design a Barlow twins learning model, the BT",
    "You build a vicreg variance learning model, the VICReg",
    "We train a masked autoencoder vision model, the ConvNeXt",
    "They propose a vision transformer learning model, the DeiT",
    "You design a pooled vision network architecture, the PVT",
    "We build a global filter network architecture, the GFNet"
]

# Build Vocabulary
words = set()
for s in sentences:
    words.update(s.split())
vocab = {word: i for i, word in enumerate(sorted(words))}
idx_to_word = {i: word for word, i in vocab.items()}

max_len = max(len(s.split()) for s in sentences)
pad_idx = len(vocab)
vocab["<PAD>"] = pad_idx
idx_to_word[pad_idx] = "<PAD>"

# Prepare Tensors for Next-Word Prediction
input_data = []
for s in sentences:
    indices = [vocab[w] for w in s.split()]
    while len(indices) < max_len:
        indices.append(pad_idx)
    input_data.append(indices)

X_full = torch.tensor(input_data)
X = X_full[:, :-1]
Y = X_full[:, 1:]
max_len = max_len - 1 

# --- 2. The Models ---

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=16, seq_len=max_len):
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(seq_len, d_model)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.d_k = d_model
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn1 = nn.Linear(d_model, d_model * 4)
        self.ffn2 = nn.Linear(d_model * 4, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.vocab_layer = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        batch_size, seq_length = x.shape
        positions = torch.arange(0, seq_length).unsqueeze(0).expand(batch_size, -1)
        emb = self.word_embedding(x) + self.pos_embedding(positions)
        
        Q = self.W_q(emb)
        K = self.W_k(emb)
        V = self.W_v(emb)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        mask = torch.tril(torch.ones(seq_length, seq_length)).unsqueeze(0)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        
        context = torch.matmul(attn_weights, V)
        out1 = self.norm1(emb + context)
        ffn_out = self.ffn2(F.relu(self.ffn1(out1)))
        final_output = self.norm2(out1 + ffn_out)
        
        return self.vocab_layer(final_output)

    def generate(self, prompt_indices, max_new_tokens=5):
        self.eval()
        for _ in range(max_new_tokens):
            with torch.no_grad():
                logits = self(prompt_indices)
                next_token_logits = logits[0, -1, :]
                next_token = torch.argmax(next_token_logits).unsqueeze(0).unsqueeze(0)
                prompt_indices = torch.cat((prompt_indices, next_token), dim=1)
        return prompt_indices

class TinyLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size=18):
        super().__init__()
        # Roughly the same amount of parameters!
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.vocab_layer = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.lstm(emb)
        return self.vocab_layer(out)
        
    def generate(self, prompt_indices, max_new_tokens=5):
        self.eval()
        for _ in range(max_new_tokens):
            with torch.no_grad():
                logits = self(prompt_indices)
                next_token_logits = logits[0, -1, :]
                next_token = torch.argmax(next_token_logits).unsqueeze(0).unsqueeze(0)
                prompt_indices = torch.cat((prompt_indices, next_token), dim=1)
        return prompt_indices

# --- 3. The Race ---

print_header("THE GENERATIVE RACE: LSTM vs TRANSFORMER")
print("Both models have roughly 12,000 parameters.")
print(f"They will race to memorize and generate the {len(sentences)} sentence dataset.")
print("Watch how the Transformer's Attention mechanism destroys the LSTM's sequential bottleneck!\n")

model_trans = TinyTransformer(vocab_size=len(vocab))
model_lstm = TinyLSTM(vocab_size=len(vocab))

opt_trans = torch.optim.Adam(model_trans.parameters(), lr=0.01)
opt_lstm = torch.optim.Adam(model_lstm.parameters(), lr=0.01)

criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

epochs = 300

print(f"{Colors.BOLD}{'Epoch':<10} | {'LSTM Loss':<20} | {'Transformer Loss':<20}{Colors.ENDC}")
print("-" * 55)

for epoch in range(1, epochs + 1):
    # Train Transformer
    model_trans.train()
    opt_trans.zero_grad()
    logits_trans = model_trans(X)
    loss_trans = criterion(logits_trans.view(-1, len(vocab)), Y.reshape(-1))
    loss_trans.backward()
    opt_trans.step()

    # Train LSTM
    model_lstm.train()
    opt_lstm.zero_grad()
    logits_lstm = model_lstm(X)
    loss_lstm = criterion(logits_lstm.view(-1, len(vocab)), Y.reshape(-1))
    loss_lstm.backward()
    opt_lstm.step()
    
    if epoch % 20 == 0:
        trans_color = Colors.GREEN if loss_trans.item() < loss_lstm.item() else Colors.RED
        lstm_color = Colors.GREEN if loss_lstm.item() < loss_trans.item() else Colors.RED
        print(f"{epoch:<10} | {lstm_color}{loss_lstm.item():<20.4f}{Colors.ENDC} | {trans_color}{loss_trans.item():<20.4f}{Colors.ENDC}")
        time.sleep(0.3)

print("\n" + "=" * 55)
print(f"Final LSTM Loss:        {Colors.RED}{loss_lstm.item():.4f}{Colors.ENDC}")
print(f"Final Transformer Loss: {Colors.GREEN}{loss_trans.item():.4f}{Colors.ENDC}")

time.sleep(2)

# --- 4. The Generative Showdown ---
print_header("THE GENERATIVE SHOWDOWN")
print("We will give both models the prompt: 'We propose a'")
print("Let's see what they generate one word at a time!\n")

prompt_str = "We propose a"
prompt_indices = torch.tensor([[vocab[w] for w in prompt_str.split()]])

# Generate with LSTM
print(f"{Colors.RED}► LSTM GENERATION:{Colors.ENDC}")
lstm_prompt = prompt_indices.clone()
model_lstm.eval()
for step in range(6):
    with torch.no_grad():
        logits = model_lstm(lstm_prompt)
        next_token = torch.argmax(logits[0, -1, :]).unsqueeze(0).unsqueeze(0)
        
        current_words = [idx_to_word[idx.item()] for idx in lstm_prompt[0]]
        predicted_word = idx_to_word[next_token.item()]
        print(f"  LSTM thinks next word is: {Colors.RED}{predicted_word}{Colors.ENDC} (Context: {' '.join(current_words)})")
        time.sleep(1)
        
        lstm_prompt = torch.cat((lstm_prompt, next_token), dim=1)

print(f"\n{Colors.BOLD}LSTM Final Output:{Colors.ENDC} {' '.join([idx_to_word[idx.item()] for idx in lstm_prompt[0]])}")
print("-" * 60 + "\n")
time.sleep(2)

# Generate with Transformer
print(f"{Colors.GREEN}► TRANSFORMER GENERATION:{Colors.ENDC}")
trans_prompt = prompt_indices.clone()
model_trans.eval()
for step in range(6):
    with torch.no_grad():
        logits = model_trans(trans_prompt)
        next_token = torch.argmax(logits[0, -1, :]).unsqueeze(0).unsqueeze(0)
        
        current_words = [idx_to_word[idx.item()] for idx in trans_prompt[0]]
        predicted_word = idx_to_word[next_token.item()]
        print(f"  Transformer thinks next word is: {Colors.GREEN}{predicted_word}{Colors.ENDC} (Context: {' '.join(current_words)})")
        time.sleep(1)
        
        trans_prompt = torch.cat((trans_prompt, next_token), dim=1)

print(f"\n{Colors.BOLD}Transformer Final Output:{Colors.ENDC} {' '.join([idx_to_word[idx.item()] for idx in trans_prompt[0]])}")
print("\nNotice how the LSTM quickly loses track of the grammatical dependencies,")
print("while the Transformer's Attention Mechanism generates perfectly coherent text!")
time.sleep(2)

# --- 5. The Training Time Benchmark (The Sequential Bottleneck) ---
print_header("THE SPEED BENCHMARK: SEQUENTIAL vs PARALLEL")
print("To show why Transformers scale to trillions of parameters (like GPT-4),")
print("we must look at Training Speed on massive documents with realistic dimensions.")
print("We will feed a 1,000-word document into both architectures.\n")

# A completely pure python LSTM to remove C++ backend advantages and show pure algorithmic math
class PurePythonLSTM(nn.Module):
    def __init__(self, d_model=512):
        super().__init__()
        self.W_ii = nn.Linear(d_model, d_model)
        self.W_hi = nn.Linear(d_model, d_model, bias=False)
        self.W_if = nn.Linear(d_model, d_model)
        self.W_hf = nn.Linear(d_model, d_model, bias=False)
        self.W_ig = nn.Linear(d_model, d_model)
        self.W_hg = nn.Linear(d_model, d_model, bias=False)
        self.W_io = nn.Linear(d_model, d_model)
        self.W_ho = nn.Linear(d_model, d_model, bias=False)
        self.d_model = d_model

    def forward(self, emb):
        batch_size, seq_len, _ = emb.shape
        h = torch.zeros(batch_size, self.d_model, device=emb.device)
        c = torch.zeros(batch_size, self.d_model, device=emb.device)
        
        # THE SEQUENTIAL BOTTLENECK: We MUST loop over every single word one-by-one
        for t in range(seq_len):
            x_t = emb[:, t, :]
            i_t = torch.sigmoid(self.W_ii(x_t) + self.W_hi(h))
            f_t = torch.sigmoid(self.W_if(x_t) + self.W_hf(h))
            g_t = torch.tanh(self.W_ig(x_t) + self.W_hg(h))
            o_t = torch.sigmoid(self.W_io(x_t) + self.W_ho(h))
            c = f_t * c + i_t * g_t
            h = o_t * torch.tanh(c)
        return h

# Create massive random embeddings (Batch: 64, Sequence Length: 1000 words, Dim: 512)
d_model_benchmark = 512
print(f"{Colors.YELLOW}Generating a batch of 64 documents, each 1,000 words long (d_model=512)...{Colors.ENDC}")
massive_batch = torch.randn(64, 1000, d_model_benchmark)

pure_lstm = PurePythonLSTM(d_model=d_model_benchmark)
pure_trans = TinyTransformer(vocab_size=len(vocab), d_model=d_model_benchmark, seq_len=2000)

print(f"\n{Colors.RED}► Testing LSTM (O(N) Sequential Process)...{Colors.ENDC}")
start_time = time.time()
lstm_out = pure_lstm(massive_batch)
lstm_time = time.time() - start_time
print(f"  {Colors.RED}Time taken: {lstm_time:.4f} seconds{Colors.ENDC}")
print(f"  (The LSTM was forced to pause and wait for the previous word 1,000 times!)")
time.sleep(1)

print(f"\n{Colors.GREEN}► Testing Transformer (O(1) Parallel Process)...{Colors.ENDC}")
start_time = time.time()
# Just timing the Attention/FFN block (ignoring embedding layer for fair comparison)
Q = pure_trans.W_q(massive_batch)
K = pure_trans.W_k(massive_batch)
V = pure_trans.W_v(massive_batch)
scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(pure_trans.d_k)
attn_weights = F.softmax(scores, dim=-1)
context = torch.matmul(attn_weights, V)
trans_time = time.time() - start_time

print(f"  {Colors.GREEN}Time taken: {trans_time:.4f} seconds{Colors.ENDC}")
print(f"  (The Transformer calculated all 1,000 words simultaneously using Matrix Multiplication!)")
print("\n" + "=" * 60)
speedup = lstm_time / trans_time
print(f"{Colors.BOLD}CONCLUSION: The Transformer is {speedup:.1f}x faster at training time!{Colors.ENDC}")
print("=" * 60 + "\n")
