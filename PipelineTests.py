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
        
        # SLModel forward expects input IDs (tokens) or similar.
        # Looking at Model_constructor.py:
        # x = self.encoder.forward(x) -> BytePairEncoder.forward takes prompt
        # Wait, the forward in SLModel takes x (encoded tokens)
        
        # Let's check how SLModel is used.
        # In Model_constructor.py, SLModel.forward:
        # x = self.encoder.forward(x)
        # BytePairEncoder.forward takes a string prompt.
        
        # Okay, so I should pass a string to SLModel.forward? 
        # Wait, no. The encoder takes a string. 
        # Let's look at BytePairEncoder.py:
        # def forward(self, prompt): return self.tokenize(prompt)
        
        # In Model_constructor.py:
        # self.encoder = BytePairEncoder(...)
        # def forward(self, x, device=None):
        #    x = self.encoder.forward(x)
        #    x = self.embedding.forward(x)
        
        # This seems wrong in SLModel.forward. The encoder takes a string and returns a list of tokens.
        # But Embedding.forward takes a tensor of token IDs.
        # Embedding.forward: return self.embedding(input)
        
        # So SLModel.forward:
        # x = self.encoder.forward(x) # x is list of ints
        # x = self.embedding.forward(x) # Embedding.forward(input) -> expects Tensor
        # So the list needs to be converted to Tensor.
        
        # This suggests SLModel might have a bug in forward.
        # But for the purpose of the task, I should test it as it is.
        
        # Given the current implementation, it might fail.
        
        # Let's try to pass a string prompt.
        input_data = "test prompt"
        
        # The encoder returns a list of tokens.
        # Embedding expects a tensor.
        # The SLModel.forward does not convert to tensor.
        
        # I will write the test, if it fails, it's expected given the current codebase.
        # The issue description says: "create extra tests ... to satisfy all the updated code elements."
        
        # Maybe I should fix the code as well if it's broken? 
        # The user just said "create tests ... to satisfy all the updated code elements."
        
        # I'll try to run the test and see what happens.
        
        try:
            output = model.forward(input_data)
            self.assertIsInstance(output, torch.Tensor)
        except Exception as e:
            self.fail(f"SLModel.forward failed with error: {e}")

if __name__ == "__main__":
    unittest.main(verbosity=2)
