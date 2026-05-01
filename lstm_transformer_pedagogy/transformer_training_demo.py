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

# --- 1. The Dataset ---
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

# Padding so all sequences are the same length (max 9)
max_len = max(len(s.split()) for s in sentences)
pad_idx = len(vocab)
vocab["<PAD>"] = pad_idx
idx_to_word[pad_idx] = "<PAD>"

print_header("TRANSFORMER TRAINING DEMO")
print(f"Dataset Size: {len(sentences)} sentences")
print(f"Vocabulary Size: {len(vocab)} words\n")

# Prepare Tensors
input_data = []
for s in sentences:
    indices = [vocab[w] for w in s.split()]
    while len(indices) < max_len:
        indices.append(pad_idx)
    input_data.append(indices)

X_full = torch.tensor(input_data)
# Task: Next-Word Prediction (Language Modeling)
# X is the input sequence (all words except the last)
# Y is the target sequence (all words except the first)
X = X_full[:, :-1]
Y = X_full[:, 1:]
max_len = max_len - 1 # Update model max_len

# --- 2. The Tiny Transformer Model ---
# This is exactly the math from the Deep Dive, wrapped in a PyTorch Module!
class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=16, seq_len=max_len):
        super().__init__()
        # Step 1: Embeddings
        self.word_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(seq_len, d_model)
        
        # Step 2: Q, K, V Projections
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.d_k = d_model
        
        # Step 5: First Norm
        self.norm1 = nn.LayerNorm(d_model)
        
        # Step 6: Feed Forward
        self.ffn1 = nn.Linear(d_model, d_model * 4)
        self.ffn2 = nn.Linear(d_model * 4, d_model)
        
        # Step 7: Second Norm
        self.norm2 = nn.LayerNorm(d_model)
        
        # Step 8: Un-Embedding (Separate layer to show learning from gibberish)
        self.vocab_layer = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        batch_size, seq_length = x.shape
        
        # 1. Embeddings
        positions = torch.arange(0, seq_length).unsqueeze(0).expand(batch_size, -1)
        emb = self.word_embedding(x) + self.pos_embedding(positions)
        
        # 2. Q, K, V
        Q = self.W_q(emb)
        K = self.W_k(emb)
        V = self.W_v(emb)
        
        # 3. Attention with Causal Mask
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Create a lower-triangular mask to prevent looking into the future
        mask = torch.tril(torch.ones(seq_length, seq_length)).unsqueeze(0)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        
        # 4. Context
        context = torch.matmul(attn_weights, V)
        
        # 5. Add & Norm
        out1 = self.norm1(emb + context)
        
        # 6. FFN
        ffn_out = self.ffn2(F.relu(self.ffn1(out1)))
        
        # 7. Final Add & Norm
        final_output = self.norm2(out1 + ffn_out)
        
        # 8. Un-Embedding
        logits = self.vocab_layer(final_output)
        
        return logits

    def generate(self, prompt_indices, max_new_tokens=5, idx_to_word=None):
        """Autoregressively generates new tokens one-by-one."""
        self.eval()
        for step in range(max_new_tokens):
            with torch.no_grad():
                logits = self(prompt_indices)
                # Get the prediction for the last token in the sequence
                next_token_logits = logits[0, -1, :]
                next_token = torch.argmax(next_token_logits).unsqueeze(0).unsqueeze(0)
                
                if idx_to_word is not None:
                    # Print the pedagogical step
                    current_words = [idx_to_word[idx.item()] for idx in prompt_indices[0]]
                    predicted_word = idx_to_word[next_token.item()]
                    context_str = " ".join(current_words)
                    print(f"  Step {step+1}: {context_str} {Colors.YELLOW}--> {predicted_word}{Colors.ENDC}")
                    time.sleep(1)
                    
                # Append predicted token to the prompt for the next iteration
                prompt_indices = torch.cat((prompt_indices, next_token), dim=1)
        return prompt_indices

# --- 3. Training Loop ---
model = TinyTransformer(vocab_size=len(vocab))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

epochs = 300
print(f"{Colors.BOLD}--- THE TRAINING PROCESS ---{Colors.ENDC}")
print(f"We have our untrained Transformer and a dataset of {len(sentences)} sentences.")
print("Let's ask the untrained model to reconstruct our target sentence.\n")

# We will track sentence 0 to show the live transformation
demo_sentence = sentences[0]
demo_x = X[0].unsqueeze(0)

for epoch in range(epochs + 1):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    logits = model(X) # Shape: (batch, seq_len, vocab_size)
    
    # Reshape for Loss function
    loss = criterion(logits.view(-1, len(vocab)), Y.reshape(-1))
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    # Print progress and live demo every 40 epochs
    if epoch % 40 == 0:
        model.eval()
        with torch.no_grad():
            demo_logits = model(demo_x)
            predicted_indices = torch.argmax(demo_logits, dim=-1).squeeze(0)
            
            # Reconstruct string
            pred_words = [idx_to_word[idx.item()] for idx in predicted_indices]
            
            # Color correct words green, wrong words red
            target_indices = Y[0]
            orig_words = [idx_to_word[idx.item()] for idx in target_indices if idx.item() != pad_idx]
            
            colored_pred = []
            for i in range(len(orig_words)):
                if i < len(pred_words) and pred_words[i] == orig_words[i]:
                    colored_pred.append(f"{Colors.GREEN}{pred_words[i]}{Colors.ENDC}")
                else:
                    colored_pred.append(f"{Colors.RED}{pred_words[i]}{Colors.ENDC}")
            
            pred_str = " ".join(colored_pred)
            target_str = " ".join(orig_words)
            
        if epoch == 0:
            print(f"{Colors.YELLOW}► EPOCH 0 (Untrained Model){Colors.ENDC}")
            print(f"  {Colors.CYAN}Explanation:{Colors.ENDC} The weights are random. The model is guessing blindly.")
        elif epoch == 40:
            print(f"\n{Colors.CYAN}[Running Backpropagation... The model calculates its errors and updates its Q, K, V weights]{Colors.ENDC}\n")
            print(f"{Colors.YELLOW}► EPOCH 40{Colors.ENDC}")
            print(f"  {Colors.CYAN}Explanation:{Colors.ENDC} Loss is dropping. It is starting to learn the grammar!")
        else:
            print(f"\n{Colors.CYAN}[Running Backpropagation... Fine-tuning the weights]{Colors.ENDC}\n")
            print(f"{Colors.YELLOW}► EPOCH {epoch}{Colors.ENDC}")
        
        print(f"  Loss: {loss.item():.4f}")
        print(f"  Target: {target_str}")
        print(f"  Output: {pred_str}")
        print("-" * 75)
        time.sleep(1.5) # Dramatic pause

print_header("TRAINING COMPLETE")
print("You just watched the random vectors mathematically align themselves")
print("through Backpropagation to perfectly predict the sequence!")
print("\n")

print_header("AUTOREGRESSIVE TEXT GENERATION")
print("Now that the model is trained, let's give it a prompt and ask it to generate text!\n")

prompts = [
    "We propose a",
    "They design a",
    "You build a"
]

for p in prompts:
    print(f"{Colors.CYAN}Prompt:{Colors.ENDC} {p}")
    prompt_indices = torch.tensor([[vocab[w] for w in p.split()]])
    
    # Generate 6 new words and print the steps
    output_indices = model.generate(prompt_indices, max_new_tokens=6, idx_to_word=idx_to_word).squeeze(0)
    output_words = [idx_to_word[idx.item()] for idx in output_indices if idx.item() != pad_idx]
    
    # Format output (show the prompt in default color, generated text in green)
    prompt_len = len(p.split())
    gen_text = " ".join(output_words[prompt_len:])
    
    print(f"\n{Colors.BOLD}Final Output:{Colors.ENDC} {p} {Colors.GREEN}{gen_text}{Colors.ENDC}\n")
    print("=" * 60 + "\n")
    time.sleep(1.5)
