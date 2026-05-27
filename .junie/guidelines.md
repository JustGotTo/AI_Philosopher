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

This repository contains simple building blocks for a small language-model pipeline and accompanying unit tests.

Components:
- `BytePairEncoder` (in `BytePairEncoder.py`): A minimal byte‑pair encoding (BPE) tokenizer that learns merges from a prompt and encodes text into integer token IDs.
- `Backend` layers (in `Backend.py`):
  - `Embedding`: Wraps `nn.Embedding` to map token IDs to dense vectors.
  - `AddNorm`: Residual add + RMS normalization with learned scale.
  - `LinearPostAttention`: Lightweight affine transform intended after attention.
  - `SentenceFeedForward`: Two-layer MLP with GELU and AddNorm.
  - `PhraseFeedForward`: One-layer projection with AddNorm and dropout.
  - Note: `WordfeedForward` and `AdaptiveMultiheadMaskedAttention` are not finished and are intentionally excluded from tests.
- `Unittests.py`: Unit tests validating dimensionality and forward-pass timing for the encoder and finished backend layers.

### Dependencies
- Python 3.9+
- PyTorch (tested with 2.x on CPU)

Install (example):
```
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### How to run tests (Windows / PowerShell)
From the project root (`C:\Users\Vova\PycharmProjects\AI_Philosopher`):
```
python -m unittest -v Unittests.py
```

### Test policy and performance budgets
- Every test checks both:
  1) Output dimensionality (tensor shapes or token list structure), and
  2) Forward-pass time of the layer/encoder on small inputs.
- Default CPU time budget used in tests: 0.5 seconds per forward pass for small batches (e.g., batch=4, seq=16). Adjust locally if needed.
- Tests run models on CPU and set deterministic seeds for reproducibility where relevant.

### Code style
- Follow PEP 8 and keep style consistent with existing files.
- Prefer explicit shapes and comments for tensor dimensions.
- Use `torch.long` for embedding indices; keep models on CPU unless explicitly needed.

### Notes for contributors
- Avoid modifying unfinished classes (`WordfeedForward`, `AdaptiveMultiheadMaskedAttention`) unless you plan to complete them and add corresponding tests.
- Keep unit tests fast and deterministic; prefer small synthetic inputs.
