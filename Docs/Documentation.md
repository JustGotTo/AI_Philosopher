# Project Documentation

This document provides a detailed description of the core components of the AI Philosopher project, including their purpose and mathematical structure.

## 1. Byte Pair Encoder (BPE)
**File:** `BytePairEncoder.py`

### Purpose
The `BytePairEncoder` is a subword tokenization method that iteratively merges the most frequent pair of adjacent tokens into a single new token. 
This allows the model to handle rare words by breaking them into smaller, known subwords and manages vocabulary size efficiently.

### Mathematical Structure and Logic
1.  **Initialization:** Starts with individual characters as the initial vocabulary.
2.  **Pair Frequency Calculation:** For a given corpus, count the occurrences of all adjacent pairs of tokens.
    $$ \text{freq}(t_i, t_{i+1}) = \sum_{w \in \text{corpus}} \text{count}((t_i, t_{i+1}) \in w) $$
3.  **Iterative Merging:** Find the pair $(L, R)$ with the highest frequency:
    $$ (L, R) = \arg\max_{p} \text{freq}(p) $$
    Create a new token $LR$ and add it to the vocabulary. Replace all occurrences of $(L, R)$ with $LR$ in the corpus.
4.  **Encoding:** To encode a word, apply the learned merges in the order they were discovered during training.

---

## 2. Backend Layers
**File:** `Backend.py`

### Embedding
-   **Purpose:** Maps integer token IDs to dense continuous vectors.
-   **Structure:** Wraps `torch.nn.Embedding`. For an input token $i$, it returns a vector $E[i] \in \mathbb{R}^d$.

### AddNorm (RMSNorm)
-   **Purpose:** Combines residual connections with Root Mean Square Layer Normalization to stabilize training.
-   **Mathematical Structure:**
    Given input $x$ and residual $r$:
    $$ \text{output} = \text{weight} \cdot \frac{x + r}{\text{RMS}(x + r) + \epsilon} $$
    Where $\text{RMS}(z) = \sqrt{\frac{1}{n} \sum z_i^2}$.

### LinearPostAttention
-   **Purpose:** A lightweight affine transformation applied after the attention mechanism.
-   **Mathematical Structure:**
    $$ y = w \cdot x + b $$
    Initialized with $w = 0.9$ and $b = 0$.

### FeedForward Networks (Sentence, Phrase, Word)
-   **Purpose:** These layers process the context at different granularities (sentence, phrase, word).
-   **SentenceFeedForward:** Two-layer MLP with GELU and AddNorm.
    $$ y = \text{Linear}_2(\text{GELU}(\text{AddNorm}(\text{Linear}_1(x)))) $$
-   **PhraseFeedForward:** One-layer projection with GELU, AddNorm, and Dropout.
-   **WordFeedForward:** MLP designed for "shrinked" input sizes, applying GELU, Linear, and AddNorm.

### BeliefsLayer
-   **Purpose:** Calculates the "beliefs" of the agent using a Top-k selection on attention outputs.
-   **Mathematical Structure:**
    1. Standard multi-head attention: $A = \text{Attention}(x, x, x)$
    2. Top-k filter: $\text{topk}(A, k)$ keeps only the $k$ most significant neurons.
    3. Update rule:
       $$ \text{beliefs} = \alpha \cdot A - (1 - \alpha) \cdot \text{topk}(A, k) $$
       (Note: The implementation uses $\alpha = 0.99$).

### AdaptiveMultiheadMaskedAttention
-   **Purpose:** A segmented hierarchical masked attention mechanism that incorporates contextual and moral biases (beliefs).
-   **Mathematical Structure:**
    -   **Segmented Attention:** The input is split into chunks with a sliding window.
    -   **Masking:** A custom mask prevents tokens from attending to specific "chunks" of sentences.
    -   **Belief Integration:**
        $$ \text{Attention}(Q, K, V, B) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}} + B + M\right)V $$
        Where $B$ is the output from the `BeliefsLayer` and $M$ is the mask.

---

## 3. PolarQuant (TurboQuant)
**File:** `PolarQuant.py`

### Purpose
`TurboQuant` implements 4-bit quantization to reduce KV cache consumption. It uses the Fast Walsh-Hadamard Transform (FWHT) to rotate the data, spreading its energy and making it more amenable to quantization.

### Mathematical Structure
1.  **Rotation (FWHT):**
    Before quantization, the input $x$ is multiplied by random signs and transformed using FWHT:
    $$ x_{\text{rot}} = \text{FWHT}(x \cdot \text{signs}) $$
    This rotation helps in reducing outliers.
2.  **Quantization:**
    Data is scaled to a 4-bit range ([-8, 7]):
    $$ \text{scale} = \max(|x_{\text{rot}}|) / 8 $$
    $$ q = \text{clamp}(\text{round}(x_{\text{rot}} / \text{scale}), -8, 7) $$
3.  **Packing:**
    Two 4-bit values are packed into a single 8-bit `uint8` byte:
    $$ \text{packed} = (q_{\text{even}} \& 0xF) | ((q_{\text{odd}} \& 0xF) \ll 4) $$
4.  **Dequantization:**
    The process is reversed, including the inverse FWHT (which is the same as FWHT scaled by $1/\sqrt{d}$).
