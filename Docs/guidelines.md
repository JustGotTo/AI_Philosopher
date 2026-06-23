# Project Guidelines
    
This is a placeholder of the project guidelines for Junie.
Replace this text with any project-level instructions for Junie, e.g.:

## Project Structure:

The project is a decoder-only Small Language model with Segment Wise Hierarchial Masked Adaptive Attention with Contextual and
Moral bias inserted in it. The model will also be Quantised using TurboQuant to reduce computational cost at each stage.


Model's ultimate goal - a research of whether different type attention could potentially create
the roles of morals and context for AI and showcase their significance. Also it attempts to analyse the text at a
human level - with deeper focus on the semantics rather than vocabulary.

### Project Overview

This repository contains building blocks for a small language-model pipeline, including quantization and adaptive attention components, along with unit and integration tests.

Components:
- `BytePairEncoder` (in `BytePairEncoder.py`): A minimal byte‑pair encoding (BPE) tokenizer that learns merges from a prompt and encodes text into integer token IDs.
- `Backend` layers (in `Backend.py`):
  - `Embedding`: Wraps `nn.Embedding` to map token IDs to dense vectors.
  - `AddNorm`: Residual add + RMS normalization with learned scale.
  - `LinearPostAttention`: Lightweight affine transform intended after attention.
  - `SentenceFeedForward`: Two-layer MLP with GELU and AddNorm.
  - `PhraseFeedForward`: One-layer projection with AddNorm and dropout.
  - `WordFeedForward`: MLP with shrinked input size, GELU, and AddNorm.
  - `BeliefsLayer`: Multihead attention-based layer using a Top-k algorithm to calculate agent beliefs.
  - Note: `AdaptiveMultiheadMaskedAttention` and `ContextLayer` are in development and partially implemented.
- `TurboQuant` (in `PolarQuant.py`): 4-bit quantization (TurboQuant) that packs two 4-bit values into a `uint8` to reduce KV cache consumption. Uses Fast Walsh-Hadamard Transform (FWHT) for rotation.
- `Unittests.py`: Unit tests validating dimensionality, accuracy (for TurboQuant), and forward-pass timing.
- `IntegrationTests.py`: Pipeline-level tests (e.g., Tokenizer -> Embedding -> FeedForward, or FeedForward -> Quantization).

### Dependencies
- Python 3.9+
- PyTorch (tested with 2.x on CPU)
- NumPy

Install (example):
```
pip install torch numpy --index-url https://download.pytorch.org/whl/cpu
```

### How to run tests (Windows / PowerShell)
From the project root (`C:\Users\Vova\PycharmProjects\AI_Philosopher`):
```
# Run all unit tests
python -m unittest -v Unittests.py

# Run integration tests
python -m unittest -v IntegrationTests.py
```

### Test policy and performance budgets
- Every test checks both:
  1) Output dimensionality (tensor shapes or token list structure), and
  2) Forward-pass time of the layer/encoder on small inputs.
- Default CPU time budget used in tests: 0.5 seconds per forward pass for small batches (e.g., batch=4, seq=16).
- Tests run models on CPU and set deterministic seeds for reproducibility.
- `TurboQuant` tests also verify dequantization accuracy (Cosine Similarity >= 0.97).

### Code style
- Follow PEP 8 and keep style consistent with existing files.
- Prefer explicit shapes and comments for tensor dimensions.
- Use `torch.long` for embedding indices; keep models on CPU unless explicitly needed.

### Notes for contributors
- `TurboQuant` requires `hidden_size` to be a power of two due to the FWHT implementation.
- `AdaptiveMultiheadMaskedAttention` is not yet fully functional. Avoid using it in production pipelines until completed.
- Keep unit tests fast and deterministic; prefer small synthetic inputs.
