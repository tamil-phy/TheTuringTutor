import torch
import torch.nn as nn

print("=" * 50)
print("Handling Variable Sequence Lengths in NLP")
print("=" * 50)
print("Language tasks often involve variable sequence length inputs and/or outputs.")
print("Depending on the task, we structure our models differently.\n")

vocab_size = 100
embed_dim = 16
hidden_dim = 32
num_classes = 2 # Positive / Negative
num_tags = 5 # POS tags

# 1. Sequence-to-Label (Many-to-One)
# E.g., Sentiment Classification
# Input: Variable length sequence
# Output: Single label
print("--- 1. Sequence-to-Label (Sentiment Classification) ---")
print("Input: Variable length sequence (e.g., 'the movie was great')")
print("Output: Single label (e.g., Positive)")

class SentimentClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        embedded = self.embedding(x)
        out, hidden = self.rnn(embedded)
        # We only care about the LAST hidden state for the final classification
        final_state = out[:, -1, :] 
        return self.fc(final_state)

model_seq2label = SentimentClassifier()
dummy_input = torch.randint(0, vocab_size, (1, 7)) # Sequence of length 7
output = model_seq2label(dummy_input)
print(f"Input shape: {dummy_input.shape} -> Output shape: {output.shape} (1 label predicted)\n")


# 2. Sequence-to-Sequence (Aligned / Same Length)
# E.g., Part-of-Speech (POS) Tagging
# Input: Variable length sequence
# Output: Sequence of the SAME length
print("--- 2. Sequence-to-Sequence Aligned (POS Tagging) ---")
print("Input: Variable length sequence (e.g., 'i love apples')")
print("Output: Same length sequence (e.g., 'PRON VERB NOUN')")

class POSTagger(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_tags)
        
    def forward(self, x):
        embedded = self.embedding(x)
        out, hidden = self.rnn(embedded)
        # We want an output for EVERY step in the sequence
        return self.fc(out)

model_aligned = POSTagger()
dummy_input = torch.randint(0, vocab_size, (1, 7)) # Sequence of length 7
output = model_aligned(dummy_input)
print(f"Input shape: {dummy_input.shape} -> Output shape: {output.shape} (7 tags predicted)\n")


# 3. Sequence-to-Sequence (Unaligned / Different Length)
# E.g., Machine Translation
# Input: Variable length sequence
# Output: Variable length sequence (often different length)
print("--- 3. Sequence-to-Sequence Unaligned (Translation) ---")
print("Input: Sequence of length N (e.g., 'how are you')")
print("Output: Sequence of length M (e.g., 'எப்படி இருக்கிறீர்கள்')")
print("Requires an Encoder-Decoder architecture.\n")

class EncoderDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_emb = nn.Embedding(vocab_size, embed_dim)
        self.encoder_rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        
        self.decoder_emb = nn.Embedding(vocab_size, embed_dim)
        self.decoder_rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, src, tgt):
        # 1. Encode source
        enc_emb = self.encoder_emb(src)
        _, hidden = self.encoder_rnn(enc_emb) # Capture the "memory"
        
        # 2. Decode using encoder's hidden state as context
        dec_emb = self.decoder_emb(tgt)
        out, _ = self.decoder_rnn(dec_emb, hidden)
        return self.fc(out)

model_encdec = EncoderDecoder()
src_input = torch.randint(0, vocab_size, (1, 5)) # English: 5 words
tgt_input = torch.randint(0, vocab_size, (1, 3)) # Tamil: 3 words
output = model_encdec(src_input, tgt_input)
print(f"Source shape: {src_input.shape}, Target shape: {tgt_input.shape} -> Output shape: {output.shape}")
print("Encoder compresses 5 words into memory, Decoder generates 3 words from it.\n")
