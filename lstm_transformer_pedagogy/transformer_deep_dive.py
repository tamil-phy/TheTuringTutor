
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import time

# --- ANSI Color Codes & UI Utilities ---
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    ENDC = '\033[0m'

def print_header(text):
    print(f"\n{Colors.HEADER}{Colors.BOLD}╔{'═'*78}╗{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}║ {text.center(76)} ║{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}╚{'═'*78}╝{Colors.ENDC}\n")

def print_step(title, desc):
    print(f"\n{Colors.YELLOW}{Colors.BOLD}► {title}{Colors.ENDC}")
    print(f"{Colors.CYAN}{desc}{Colors.ENDC}\n")

def pause_for_audience():
    input(f"\n{Colors.BOLD}[Press Enter to continue to the next step...]{Colors.ENDC}")
    print("\033[A\033[K\r", end="") # Clear the input line

def get_color_bg(val, min_val, max_val, is_prob=False):
    """Returns ANSI escape string for a background color based on value magnitude."""
    # Normalize value between 0 and 1
    if is_prob:
        norm = max(0.0, min(1.0, val))
        # Probability colormap: Black (0) to Bright Blue (1)
        r, g, b = int(0), int(100 * norm), int(255 * norm)
    else:
        # Diverging colormap: Red (-), Black (0), Green (+)
        max_mag = max(abs(min_val), abs(max_val), 1e-5)
        norm = val / max_mag # -1 to 1
        if norm < 0:
            intensity = int(abs(norm) * 255)
            r, g, b = intensity, 0, 0
        else:
            intensity = int(norm * 255)
            r, g, b = 0, intensity, 0

    # Choose text color (white or black) based on background brightness
    brightness = (r * 299 + g * 587 + b * 114) / 1000
    text_color = "30" if brightness > 128 else "37" # 30=black, 37=white
    return f"\033[48;2;{r};{g};{b}m\033[{text_color}m"

def print_tensor_heatmap(tensor, row_labels=None, col_labels=None, is_prob=False, title=""):
    """Prints a 2D tensor as a colorful heatmap."""
    if tensor.dim() == 3:
        tensor = tensor.squeeze(0) # Remove batch dim if present
    
    if tensor.dim() != 2:
        print(tensor)
        return

    rows, cols = tensor.shape
    min_val, max_val = tensor.min().item(), tensor.max().item()

    if title:
        print(f"{Colors.BOLD}{title}{Colors.ENDC}")

    # Print Column Headers
    if col_labels:
        header_row = " " * 15
        for c in range(cols):
            label = str(col_labels[c])[:8] if c < len(col_labels) else f"C{c}"
            header_row += f"{label:>8} "
        print(f"{Colors.CYAN}{header_row}{Colors.ENDC}")

    # Print Rows
    for r in range(rows):
        row_str = ""
        if row_labels:
            label = str(row_labels[r])[:12] if r < len(row_labels) else f"R{r}"
            row_str += f"{Colors.CYAN}{label:>14} {Colors.ENDC}"
        else:
            row_str += f"{Colors.CYAN}Row {r:>2}: {Colors.ENDC}"

        for c in range(cols):
            val = tensor[r, c].item()
            color_code = get_color_bg(val, min_val, max_val, is_prob)
            formatted_val = f"{val: .2f}"
            row_str += f"{color_code}{formatted_val:>7}{Colors.ENDC} "
        
        # If probability, print row sum
        if is_prob:
            row_sum = tensor[r].sum().item()
            row_str += f" | {Colors.BOLD}Σ={row_sum:.2f}{Colors.ENDC}"
            
        print(row_str)
    print()

# --- Setup ---
torch.manual_seed(42) # For reproducible random values
print_header("TRANSFORMER DEEP DIVE: The Mathematical Mechanics")
print(f"We will build a Transformer block from scratch using the text you provided:")
print(f"{Colors.GREEN}'We propose a new simple network architecture, the Transformer'{Colors.ENDC}\n")

sentence = ["We", "propose", "a", "new", "simple", "network", "architecture,", "the", "Transformer"]
vocab = {word: i for i, word in enumerate(set(sentence))} # Simple vocab mapping
seq_len = len(sentence)
d_model = 6 # Small dimension to fit on screen beautifully
d_k = 6 # Dimension of Q and K
d_v = 6 # Dimension of V
d_ff = 12 # Feed forward dimension

print(f"{Colors.BOLD}Sequence Length:{Colors.ENDC} {seq_len} words")
print(f"{Colors.BOLD}Model Dimension (d_model):{Colors.ENDC} {d_model}")
pause_for_audience()

input_indices = torch.tensor([[vocab[word] for word in sentence]])

# --- Step 1: Embeddings & Positional Encoding ---
print_step("Step 1: Input Embeddings & Positional Encoding", 
           "Converting discrete words into dense mathematical vectors (Embeddings), and injecting Position Signals (since there is no RNN).")

word_embedding_layer = nn.Embedding(len(vocab), d_model)
pos_embedding_layer = nn.Embedding(seq_len, d_model)

with torch.no_grad():
    word_embeddings = word_embedding_layer(input_indices).squeeze(0)
    positions = torch.arange(0, seq_len)
    pos_embeddings = pos_embedding_layer(positions)
    x = word_embeddings + pos_embeddings # The combined input

print_tensor_heatmap(word_embeddings, row_labels=sentence, title="Word Embeddings Matrix (Learned Semantics)")
pause_for_audience()

print_tensor_heatmap(pos_embeddings, row_labels=[f"Pos {i}" for i in range(seq_len)], title="Positional Encodings (Where am I?)")
pause_for_audience()

print_tensor_heatmap(x, row_labels=sentence, title="Final Input Matrix (X) = Embeddings + Positional Encodings")
pause_for_audience()


# --- Step 2: The Q, K, V Projections ---
print_step("Step 2: Generating Query (Q), Key (K), and Value (V) Matrices", 
           "Every word generates three vectors:\n"
           "  - Query (Q): What context am I looking for?\n"
           "  - Key (K): What information do I contain?\n"
           "  - Value (V): If you match with me, this is my actual meaning.")

W_q = nn.Linear(d_model, d_k, bias=False)
W_k = nn.Linear(d_model, d_k, bias=False)
W_v = nn.Linear(d_model, d_v, bias=False)

with torch.no_grad():
    Q = W_q(x)
    K = W_k(x)
    V = W_v(x)

print_tensor_heatmap(Q, row_labels=sentence, title="Queries (Q): What each word is looking for")
print_tensor_heatmap(K, row_labels=sentence, title="Keys (K): What each word represents")
print_tensor_heatmap(V, row_labels=sentence, title="Values (V): The payload of each word")
pause_for_audience()


# --- Step 3: Scaled Dot-Product Attention ---
print_step("Step 3: Scaled Dot-Product Attention", 
           "The core mathematical engine: Softmax((Q @ K^T) / sqrt(d_k))\n"
           "We take the dot product of every Query with every Key to find out how much words should 'pay attention' to each other.")

with torch.no_grad():
    # 1. Dot Product (Scores)
    scores = torch.matmul(Q, K.transpose(-2, -1))
    
    print_tensor_heatmap(scores, row_labels=sentence, col_labels=sentence, 
                         title="Raw Match Scores Matrix (Q @ K^T)")
    pause_for_audience()

    # 2. Scale & Softmax
    scaled_scores = scores / math.sqrt(d_k)
    attention_weights = F.softmax(scaled_scores, dim=-1)

    print_tensor_heatmap(attention_weights, row_labels=sentence, col_labels=sentence, is_prob=True,
                         title="Attention Weights (Softmax applied to scaled scores)\nObserve how every row perfectly sums to 1.0 (100% distribution of attention)")
pause_for_audience()


# --- Step 4: Context Aggregation ---
print_step("Step 4: Extracting the New Context", 
           "Now we multiply the Attention Weights by the Values (V).\n"
           "If 'Transformer' pays 90% attention to 'architecture', it absorbs 90% of 'architecture's' Value vector.")

with torch.no_grad():
    attention_output = torch.matmul(attention_weights, V)

print_tensor_heatmap(attention_output, row_labels=sentence, 
                     title="Context-Aware Representations (Attention_Weights @ V)")
pause_for_audience()


# --- Step 5: Add & Norm (Residual Connection 1) ---
print_step("Step 5: Add & Norm (Residual Connection)", 
           "We add the original input (X) back to the attention output (Residual Connection).\n"
           "This provides an 'amnesia shortcut' to prevent the vanishing gradient problem, followed by Layer Normalization.")

layer_norm_1 = nn.LayerNorm(d_model)

with torch.no_grad():
    residual_1 = x + attention_output
    out_1 = layer_norm_1(residual_1)

print_tensor_heatmap(out_1, row_labels=sentence, title="Output Matrix after Add & Norm")
pause_for_audience()


# --- Step 6: Feed-Forward Network ---
print_step("Step 6: The Feed-Forward Network (FFN)", 
           "Attention gathers information. The FFN processes it.\n"
           "We expand the dimensions temporarily to 'think' about the gathered context, then compress it back down.")

ffn_linear_1 = nn.Linear(d_model, d_ff)
ffn_linear_2 = nn.Linear(d_ff, d_model)

with torch.no_grad():
    ffn_intermediate = F.relu(ffn_linear_1(out_1))
    
    print(f"Expanding from {d_model} dimensions to {d_ff} dimensions...")
    print_tensor_heatmap(ffn_intermediate, row_labels=sentence, title=f"Intermediate FFN Matrix (after ReLU)")
    pause_for_audience()

    ffn_output = ffn_linear_2(ffn_intermediate)
    print(f"Compressing back to {d_model} dimensions...")
    print_tensor_heatmap(ffn_output, row_labels=sentence, title="FFN Output Matrix")
pause_for_audience()


# --- Step 7: Final Add & Norm ---
print_step("Step 7: Final Add & Norm", 
           "One last residual connection and normalization to finish the Transformer block!")

layer_norm_2 = nn.LayerNorm(d_model)

with torch.no_grad():
    final_output = layer_norm_2(out_1 + ffn_output)

print_tensor_heatmap(final_output, row_labels=sentence, title="Final Transformed Matrix (The Output of the Block)")

pause_for_audience()

# --- Step 8: Decoding (Un-Embedding) ---
print_step("Step 8: Un-Embedding (Converting Numbers back to Text)", 
           "We have our final mathematical vectors. To get English back, we multiply these vectors by the transposed\n"
           "Word Embeddings matrix. This finds the word in our vocabulary whose 'semantic DNA' is mathematically closest to our output!")

with torch.no_grad():
    # Weight tying: Use the transpose of the input embeddings as the output un-embeddings
    logits = torch.matmul(final_output, word_embedding_layer.weight.T)
    predicted_indices = torch.argmax(logits, dim=-1)

# Inverse vocab mapping
idx_to_word = {i: word for word, i in vocab.items()}

print(f"{Colors.BOLD}Final Output Conversion:{Colors.ENDC}")
print(f"   {'Input Word':>14}  ->  {'Closest Word in Vector Space':<20}")
print("  " + "-"*48)
for i, orig_word in enumerate(sentence):
    pred_word = idx_to_word[predicted_indices[i].item()]
    print(f"   {Colors.CYAN}{orig_word:>14}{Colors.ENDC}  ->  {Colors.GREEN}{Colors.BOLD}{pred_word}{Colors.ENDC}")

print("\n(Note: Since this model is untrained, the predictions are random, but this is exactly how ChatGPT predicts the next word!)")

print_header("TRANSFORMER BLOCK COMPLETE")
print("You have just traced the exact sequence of mathematical operations that powers")
print("models like GPT-4, Gemini, and the architecture proposed in 'Attention Is All You Need'!")
print("\n")
