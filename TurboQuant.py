import torch as t
import numpy as np
import random


from torch import nn

class TurboQuant(nn.Module):
    def __init__(self):
        super().__init__()
