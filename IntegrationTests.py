import unittest
import torch
import time
from BytePairEncoder import BytePairEncoder
from Backend import Embedding, AddNorm, LinearPostAttention, SentenceFeedForward, PhraseFeedForward
from PolarQuant import TurboQuant

TIME_BUDGET_SEC = 0.5

class TestIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(1234)
        cls.prompt = "This is a sample prompt for training the BPE tokenizer and testing integration."
        cls.vocab_size = 500
        cls.embed_dim = 512
        cls.hidden_size = 512
        cls.out_size = 512
        
        # Initialize BPE and train it with the prompt
        cls.bpe = BytePairEncoder(vocab_size=cls.vocab_size)
        cls.tokens = cls.bpe.forward(cls.prompt)

    def test_tokenizer_to_embedding_pipeline(self):
        """Test Tokenizer -> Embedding -> SentenceFeedForward pipeline."""
        t0 = time.perf_counter()
        
        # 1. Tokenize a new sentence
        test_sentence = "integration test sample"
        token_ids = self.bpe.forward(test_sentence)
        self.assertIsInstance(token_ids, list)
        
        # 2. Convert to tensor
        input_tensor = torch.tensor([token_ids], dtype=torch.long) # batch=1
        
        # 3. Embedding layer
        # Note: Backend.py Embedding init calls bpe.forward(prompt) internally but doesn't store the bpe instance.
        # However, it creates a new BytePairEncoder(prompt=prompt).forward(prompt) inside __init__.
        # For integration, we use the Embedding layer.
        embed_layer = Embedding(prompt=self.prompt, vocab_size=self.vocab_size, embedding_dim=self.embed_dim)
        embedded = embed_layer.forward(input_tensor)
        
        # Check dimensionality
        self.assertEqual(embedded.shape, (1, len(token_ids), self.embed_dim))
        
        # 4. SentenceFeedForward (hidden_size must match embed_dim of input)
        # In Backend.py, SentenceFeedForward(hidden_size, output_size)
        sff = SentenceFeedForward(hidden_size=self.embed_dim, output_size=self.out_size)
        output = sff.forward(embedded)
        
        self.assertEqual(output.shape, (1, len(token_ids), self.out_size))
        
        dt = time.perf_counter() - t0
        self.assertLess(dt, TIME_BUDGET_SEC, f"Pipeline 1 too slow: {dt:.4f}s")

    def test_phrase_ff_and_quantization(self):
        """Test PhraseFeedForward -> AddNorm -> TurboQuant."""
        t0 = time.perf_counter()
        
        batch, seq_len, hidden = 2, 16, 512 # use 512 as it's power of 2
        x = torch.randn(batch, seq_len, hidden)
        
        # 1. PhraseFeedForward
        pff = PhraseFeedForward(hidden_size=hidden, output_size=hidden, shrinked_size=hidden//8)
        x_pff = pff.forward(x)
        self.assertEqual(x_pff.shape, (batch, seq_len, hidden))
        
        # Check for parameter conflicts (if any shared names or unexpected interactions)
        # Here we just ensure we can create and run them independently without error
        
        # 2. AddNorm
        norm = AddNorm(hidden_size=hidden)
        x_norm = norm.forward(x_pff, x_pff)
        self.assertEqual(x_norm.shape, (batch, seq_len, hidden))
        
        # 3. LinearPostAttention
        lpa = LinearPostAttention(output_size=hidden)
        x_lpa = lpa.forward(x_norm)
        self.assertEqual(x_lpa.shape, (batch, seq_len, hidden))
        
        # 4. TurboQuant (Quantization)
        tq = TurboQuant(hidden_size=hidden)
        packed, scale, amax = tq.quantize(x_lpa)
        
        # Shape check (packed last dim is hidden // 2)
        self.assertEqual(packed.shape, (batch, seq_len, hidden // 2))
        
        # 5. Dequantization and Accuracy check
        x_hat = tq.dequantize()
        self.assertEqual(x_hat.shape, (batch, seq_len, hidden))
        
        cos_sim = torch.nn.functional.cosine_similarity(x_lpa.flatten(), x_hat.flatten(), dim=0)
        self.assertGreaterEqual(cos_sim.item(), 0.97, f"Integration quantization accuracy too low: {cos_sim.item():.4f}")
        
        dt = time.perf_counter() - t0
        self.assertLess(dt, TIME_BUDGET_SEC, f"Pipeline 2 too slow: {dt:.4f}s")

if __name__ == "__main__":
    unittest.main(verbosity=2)
