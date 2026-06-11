import torch as t
import numpy as np
from PolarQuant import TurboQuant

def test_turboquant_accuracy():
    hidden_size = 512
    batch, seq_len = 2, 10
    tq = TurboQuant(hidden_size=hidden_size)
    
    x = t.randn(batch, seq_len, hidden_size)
    
    packed, scale, amax = tq.quantize(x)
    print(f"Packed shape: {packed.shape}")

    try:
        x_hat = tq.dequantize()
        print(f"Dequantized shape: {x_hat.shape}")
        
        cos_sim = t.nn.functional.cosine_similarity(x.flatten(), x_hat.flatten(), dim=0)
        print(f"Cosine Similarity: {cos_sim.item():.4f}")
        
        mse = t.nn.functional.mse_loss(x, x_hat)
        print(f"MSE: {mse.item():.4f}")
    except Exception as e:
        print(f"Error during dequantize: {e}")

if __name__ == "__main__":
    test_turboquant_accuracy()
