import unittest
import torch
from Model_constructor import SLModel

class TestPipeline(unittest.TestCase):
    def test_slmodel_forward(self):
        hidden_size = 512
        embedding_dim = 512
        vocab_size = 1000
        prompt = "test pipeline integration."
        
        model = SLModel(hidden_size=hidden_size, embedding_dim=embedding_dim, vocab_size=vocab_size, prompt=prompt)
        input_data = "test prompt"
        try:
            output = model.forward(input_data)
            self.assertIsInstance(output, torch.Tensor)
        except Exception as e:
            self.fail(f"SLModel.forward failed with error: {e}")

if __name__ == "__main__":
    unittest.main(verbosity=2)
