# TheTuringTutor

This repo contains my tutoring materials for teaching NLP and language models. I use these for 1-on-1 sessions, but they're also useful if you're learning on your own.

The main idea is simple: you don't really understand something until you've built it yourself. So instead of just reading papers or using libraries, we implement things from scratch. We start with basic statistical models (unigrams, bigrams) and work our way up to transformers and modern LLMs.

## What's the approach?

I find that most people learn better when they can see the whole picture—from the math to the actual running code. So everything here includes:

- The underlying math and intuition
- Implementations from scratch (minimal dependencies)
- Comparisons between different approaches
- Lots of comments explaining why things work the way they do

This isn't about memorizing APIs or copying code. It's about understanding the core ideas well enough that you could rebuild them if you needed to.

## What's in here?

### machine_learning/

Before diving into neural networks, this covers fundamental machine learning concepts with hands-on Jupyter notebooks:

- `Hands_on_linear_regression.ipynb` - linear regression from scratch
- `Linear_vs_Polynomial.ipynb` - comparing linear and polynomial fits
- `real_world.ipynb` - applying ML to real-world datasets
- `toronto_session_2_supervised_learning.ipynb` - supervised learning fundamentals

Start here if you're completely new to ML. These notebooks build intuition for optimization, loss functions, and model fitting before adding the complexity of neural networks.

### neural_network/

- `add_nn.ipynb` - teaches neural networks by solving a simple problem: learning to add two numbers

This might sound trivial, but it's actually a great way to understand:
- How forward and backward passes work
- Why training can fail (and how to debug it)
- What gradient descent is actually doing
- The difference between good and bad hyperparameters

The notebook shows both working and broken versions so you can see what goes wrong.

### language_models/

This is where we start from the basics. The progression goes from simple statistical models to neural networks:

**Statistical approaches:**
- `unigram.py` - simplest baseline, just word frequencies
- `bigram.py` - one step up, predicts based on the previous word
- `ngram.py` - extends to longer contexts
- `ngram_models.py` - utilities for working with n-grams

**Neural approaches:**
- `rnn_lm.py` - basic RNN language model
- `rnn_lm_generate.py` - text generation using RNNs
- `nn_models.py` - various neural architectures

**Comparisons and demos:**
- `all_language_models.py` - runs everything side-by-side (unigram through transformer)
- `demo.py` - compares RNN, LSTM, and Transformer on a storytelling task
- `multi_lm.py` - framework for comparing multiple models
- `sequence_tasks_demo.py` - different sequence modeling tasks

The idea is to start with unigrams so you can see why they don't work well, then build up to bigrams and n-grams. Once you hit the limitations of statistical models (they can't handle long-range dependencies), neural models start to make sense.

### lstm_transformer_pedagogy/

This directory goes deeper into the architectures that actually work for modern NLP:

- `lstm_lm.py` - LSTM language model
- `transformer_lm.py` - Transformer language model
- `transformer_deep_dive.py` - detailed walkthrough of how transformers work
- `transformer_demo.py` - interactive demos
- `transformer_training_demo.py` - shows the training process step by step
- `lstm_vs_transformer_race.py` - head-to-head performance comparison
- `test_pure_lstm.py` - tests and validation
- `amnesia_test_demo.py` - experiments on memory retention
- `test_speed2.py` - speed benchmarks

### rnn_name_generation/

- `RNN.ipynb` - character-level RNN for generating names

This is a classic teaching example. The model learns patterns in names (character by character) and can generate new ones. It's a good way to see how RNNs handle sequential data at the character level.

### seq2seq/

- `seq_2_seq.ipynb` - English to Tamil translation using encoder-decoder architecture

This notebook implements a character-level seq2seq model that translates English words to Tamil. It's a complete encoder-decoder system with:
- An LSTM encoder that reads the English word
- An LSTM decoder that generates the Tamil translation
- Teacher forcing during training
- Actual translation examples

Good for understanding how sequence-to-sequence models work before moving to attention mechanisms.

### தமிழ்_gpt/

- `தமிழ்_GPT.ipynb` - GPT-style model for Tamil text generation

This applies modern LLM architectures to Tamil, a morphologically rich language. Shows how the same principles (transformer architecture, autoregressive training) work across different languages and writing systems.

## Getting started

You'll need Python 3.8+ and PyTorch (and TensorFlow for some notebooks):

```bash
pip install torch tensorflow numpy matplotlib jupyter
```

The materials are split between Python scripts (`.py` files) and Jupyter notebooks (`.ipynb` files).

**For Jupyter notebooks:**
```bash
# Start a notebook server
jupyter notebook

# Then open any .ipynb file from the browser
# Good starting points:
# - machine_learning/Hands_on_linear_regression.ipynb
# - neural_network/add_nn.ipynb
```

**For Python scripts:**
```bash
# Start with the comparison of all model types
cd language_models
python all_language_models.py

# Or jump to transformers if you're already comfortable with the basics
cd lstm_transformer_pedagogy
python transformer_demo.py
```

## Video lectures

I've recorded a lecture series that walks through many of these concepts. If you prefer video explanations alongside the code, check out the playlist here:

**[Lecture Series](https://youtube.com/playlist?list=PLIez9dJD6K9YJVjckIwQbq3ebP97OSH-f)**

The videos cover the same material but with more explanation and live coding demos.

## Recommended learning path

If you're new to this, I'd suggest following this order:

**Start here: ML fundamentals** (`machine_learning/`)
- Linear regression and polynomial fitting
- Understanding loss functions and optimization
- Working with real-world datasets
- This builds the foundation before neural networks

**Then: Basic neural networks** (`neural_network/`)
- Learn how backprop actually works by teaching a network to add numbers
- See what goes wrong and how to fix it
- Understand gradient descent intuitively

**Next: Statistical language models** (`language_models/`)
- Unigrams and bigrams (see why simple frequency-based methods fail)
- N-grams (understand the context window problem)
- This gives you intuition for why we need neural approaches

**After that: RNN-based models** (`language_models/`, `rnn_name_generation/`)
- Simple RNNs (understand recurrent connections)
- LSTMs and GRUs (see how we solve the vanishing gradient problem)
- Character-level generation (names, text)
- Train a small language model and watch it struggle with long-range dependencies

**Then: Sequence-to-sequence** (`seq2seq/`)
- Encoder-decoder architecture
- Translation as a sequence problem
- Teacher forcing
- Why this is still limited (spoiler: no attention yet)

**Next: Attention and transformers** (`lstm_transformer_pedagogy/`)
- Self-attention mechanism (the core innovation)
- Multi-head attention (why multiple attention heads help)
- Full transformer architecture
- Compare LSTM vs Transformer performance directly

**Finally: Modern applications** (`தமிழ்_gpt/`)
- GPT-style models
- Multilingual applications
- Current architectures in practice

You don't have to follow this exactly, but jumping straight to transformers without understanding RNNs first makes it harder to appreciate why transformers are designed the way they are.

## How to use this

**If you're learning on your own:**
- Read the code before you run it. Everything is commented pretty heavily.
- Don't just run the demos—modify them. Change hyperparameters, break things, see what happens.
- Try implementing small variations yourself.
- The code is meant to be readable, not production-ready. Focus on understanding the concepts.

**If you're using this for teaching:**
- Each module is self-contained, so you can pick and choose what's relevant.
- The demos work well as starting points for discussions.
- Students learn more when they modify the code themselves rather than just reading it.

## What's covered (and what's coming)

**Currently available:**

*Machine learning foundations*
- Linear and polynomial regression
- Loss functions and optimization
- Supervised learning fundamentals
- Real-world dataset applications

*Neural network basics*
- Forward and backward propagation
- Gradient descent visualization
- Training diagnostics and debugging
- Hyperparameter tuning

*Statistical language models*
- Unigram, bigram, and n-gram models
- Why they fail for real language modeling
- Probability-based text prediction

*Neural sequence models*
- RNNs and why they have trouble with long sequences
- LSTMs/GRUs and the vanishing gradient problem
- Character-level generation
- RNN language models with text generation

*Sequence-to-sequence*
- Encoder-decoder architecture
- English-Tamil translation (character-level)
- Teacher forcing

*Transformers and attention*
- Self-attention mechanism
- Multi-head attention
- Full transformer architecture
- Performance comparisons (RNN vs LSTM vs Transformer)
- Memory and long-range dependency experiments

*Applications*
- Tamil GPT (multilingual LLMs)
- Name generation
- Translation systems

**Still working on:**
- Detailed positional encoding analysis
- Pretraining techniques (MLM, CLM, etc.)
- Fine-tuning strategies
- Attention mechanisms in seq2seq
- Efficient transformer variants
- RAG and retrieval methods

This is a living repo—I add new materials as I teach new topics or find better ways to explain existing ones.

## Contributing

This is mainly for my own tutoring work, but if you spot bugs or have suggestions, feel free to open an issue. Just keep in mind that the goal here is teaching, not performance—so "this could be 10x faster if..." suggestions might not be the priority. Clarity wins over optimization.

## License

Everything here is for educational purposes. Use the code however you want for learning.

---

If you have questions about the materials or want to suggest improvements, open an issue. For private tutoring inquiries, same thing—start with an issue and we can take it from there.
