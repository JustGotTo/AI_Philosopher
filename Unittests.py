import unittest
import time
import random
import numpy as np
import torch as torch

from BytePairEncoder import BytePairEncoder
from Backend import (
    Embedding,
    AddNorm,
    LinearPostAttention,
    SentenceFeedForward,
    PhraseFeedForward,
)


TIME_BUDGET_SEC = 0.5  # Per-forward pass CPU time budget for small inputs


def set_seeds(seed: int = 1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class TestBytePairEncoder(unittest.TestCase):
    def test_bpe_tokenize_structure_and_time(self):
        set_seeds()
        prompt = "hello world this is a tiny test for bpe tokenizer"
        bpe = BytePairEncoder(vocab_size=1000)

        t0 = time.perf_counter()
        tokens = bpe.forward(prompt)
        dt = time.perf_counter() - t0

        # Structure checks
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)
        for tok in tokens:
            self.assertIsInstance(tok, int)
            self.assertGreaterEqual(tok, 0)
            self.assertLess(tok, bpe.vocab_size)

        # Performance check
        self.assertLess(dt, TIME_BUDGET_SEC, f"BPE forward pass too slow: {dt:.4f}s")


class TestBackendLayers(unittest.TestCase):
    def test_embedding_shape_and_time(self):
        set_seeds()
        batch, seq_len = 4, 16
        vocab_size, embed_dim = 200, 32
        # Input indices must be in [0, vocab_size)
        x = torch.randint(low=0, high=vocab_size, size=(batch, seq_len), dtype=torch.long)
        layer = Embedding(prompt="some prompt for encoder warmup", vocab_size=vocab_size, embedding_dim=embed_dim)

        t0 = time.perf_counter()
        y = layer.forward(x)
        dt = time.perf_counter() - t0

        # Shape: (batch, seq_len, embed_dim)
        self.assertEqual(tuple(y.shape), (batch, seq_len, embed_dim))
        self.assertLess(dt, TIME_BUDGET_SEC, f"Embedding forward pass too slow: {dt:.4f}s")

    def test_addnorm_shape_and_time(self):
        set_seeds()
        batch, seq_len, hidden = 4, 16, 64
        residual = torch.randn(batch, seq_len, hidden)
        x = torch.randn(batch, seq_len, hidden)
        layer = AddNorm(hidden_size=hidden)

        t0 = time.perf_counter()
        y = layer.forward(residual, x)
        dt = time.perf_counter() - t0

        self.assertEqual(tuple(y.shape), (batch, seq_len, hidden))
        self.assertLess(dt, TIME_BUDGET_SEC, f"AddNorm forward pass too slow: {dt:.4f}s")

    def test_linear_post_attention_shape_and_time(self):
        set_seeds()
        batch, seq_len, d = 4, 16, 64
        x = torch.randn(batch, seq_len, d)
        layer = LinearPostAttention(output_size=d)

        t0 = time.perf_counter()
        y = layer.forward(x)
        dt = time.perf_counter() - t0

        self.assertEqual(tuple(y.shape), (batch, seq_len, d))
        self.assertLess(dt, TIME_BUDGET_SEC, f"LinearPostAttention forward pass too slow: {dt:.4f}s")

    def test_sentence_feed_forward_shape_and_time(self):
        set_seeds()
        batch, seq_len, hidden, out = 4, 16, 64, 48
        x = torch.randn(batch, seq_len, hidden)
        layer = SentenceFeedForward(hidden_size=hidden, output_size=out)

        t0 = time.perf_counter()
        y = layer.forward(x)
        dt = time.perf_counter() - t0

        self.assertEqual(tuple(y.shape), (batch, seq_len, out))
        self.assertLess(dt, TIME_BUDGET_SEC, f"SentenceFeedForward forward pass too slow: {dt:.4f}s")

    def test_phrase_feed_forward_shape_and_time(self):
        set_seeds()
        batch, seq_len, hidden, out, shrink = 4, 16, 64, 32, 16
        x = torch.randn(batch, seq_len, hidden)
        layer = PhraseFeedForward(hidden_size=hidden, output_size=out, shrinked_size=shrink)

        t0 = time.perf_counter()
        y = layer.forward(x)
        dt = time.perf_counter() - t0

        self.assertEqual(tuple(y.shape), (batch, seq_len, out))
        self.assertLess(dt, TIME_BUDGET_SEC, f"PhraseFeedForward forward pass too slow: {dt:.4f}s")

class TestTurboQuant(unittest.TestCase):
    def test_turboquant_shape_and_accuracy(self):
        set_seeds()
        from PolarQuant import TurboQuant
        hidden = 512 # Must be power of 2
        batch, seq_len = 2, 8
        x = torch.randn(batch, seq_len, hidden)
        tq = TurboQuant(hidden_size=hidden)

        t0 = time.perf_counter()
        packed, scale, amax = tq.quantize(x)
        x_hat = tq.dequantize()
        dt = time.perf_counter() - t0

        # Shape checks
        # Packed is (batch, seq_len, hidden // 2) because it packs two 4-bit values into uint8
        self.assertEqual(tuple(packed.shape), (batch, seq_len, hidden // 2))
        self.assertEqual(tuple(x_hat.shape), (batch, seq_len, hidden))

        # Accuracy check (around 97-98% as requested)
        cos_sim = torch.nn.functional.cosine_similarity(x.flatten(), x_hat.flatten(), dim=0)
        self.assertGreaterEqual(cos_sim.item(), 0.97, f"Cosine similarity too low: {cos_sim.item():.4f}")

        # Performance check
        self.assertLess(dt, TIME_BUDGET_SEC, f"TurboQuant cycle too slow: {dt:.4f}s")

class TestAdaptiveMultiheadMaskedAttentionMask(unittest.TestCase):
    def test_create_mask_dimensions_and_zero_counts(self):
        set_seeds()
        from Backend import AdaptiveMultiheadMaskedAttention
        from types import SimpleNamespace

        batch_size = 8
        mask_window_size = 3  # ensure < batch_size so no internal clipping

        # Avoid calling the real __init__ (which has incomplete logic).
        # Instead, build a minimal dummy carrying only the attributes used by createMask.
        dummy = SimpleNamespace(
            batch_size=batch_size,
            mask_window_size=mask_window_size,
            embedding_dim=1,  # Using 1 ensures exactly mask_window_size zeros per row
        )

        t0 = time.perf_counter()
        # Call the unbound method with our dummy instance.
        mask = AdaptiveMultiheadMaskedAttention.createMask(dummy)
        print(mask)
        dt = time.perf_counter() - t0

        # Dimension check
        self.assertIsInstance(mask, torch.Tensor)
        self.assertEqual(tuple(mask.shape), (batch_size, batch_size))

        # Values should be only 0 or 1
        unique_vals = torch.unique(mask)
        for v in unique_vals:
            self.assertIn(float(v.item()), [0.0, 1.0])

        # Each "layer" (row) should contain exactly mask_window_size zeros
        for i in range(batch_size):
            zero_count = int((mask[i] == 0).sum().item())
            self.assertEqual(
                zero_count, mask_window_size,
                f"Row {i} should have exactly {mask_window_size} zeros, got {zero_count}"
            )

        # Optional performance check
        self.assertLess(dt, TIME_BUDGET_SEC, f"createMask too slow: {dt:.4f}s")

class TestAdaptiveMultiheadMaskedAttentionSplitBatch(unittest.TestCase):
    def test_split_batch_equal_chunks_stride(self):
        set_seeds()
        from Backend import AdaptiveMultiheadMaskedAttention
        from types import SimpleNamespace

        # Prepare an input tensor whose last dimension length is an exact multiple of chunk size
        batch, seq, features = 2, 3, 12
        chunk_size = 3  # pretend mean sentence length
        x = torch.arange(batch * seq * features, dtype=torch.float32).view(batch, seq, features)

        # Build a dummy carrying only what split_batch needs
        dummy = SimpleNamespace(
            # Monkeypatch getMeanSentenceLength to return our desired integer chunk size
            getMeanSentenceLength=lambda prompt: chunk_size
        )

        prompt = "xxx."  # content irrelevant due to monkeypatch

        t0 = time.perf_counter()
        y = AdaptiveMultiheadMaskedAttention.split_batch(dummy, x, prompt)
        dt = time.perf_counter() - t0

        # Expected behavior: take every `chunk_size`-th element along the last dim, starting at 0
        expected = x[..., 0::chunk_size]

        # Shape check: last dimension shrinks by factor of chunk_size
        self.assertEqual(tuple(y.shape), (batch, seq, features // chunk_size))

        # Value check: exactly equals the simple stride slice
        self.assertTrue(torch.equal(y, expected), "split_batch should be equivalent to x[..., 0::chunk_size]")

        # "Equal chunks" interpretation: with divisible length, the resulting last dim is uniform across batch and seq
        self.assertTrue(all(d == features // chunk_size for d in y.shape[-1:]))

        # Performance check
        self.assertLess(dt, TIME_BUDGET_SEC, f"split_batch too slow: {dt:.4f}s")


if __name__ == "__main__":
    unittest.main(verbosity=2)
