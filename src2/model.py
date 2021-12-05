import torch.nn as nn
import numpy as np
import torch.nn as nn

class Word2Vec(nn.Module):
    def __init__(self, vector_size):
        super(Word2Vec, self).__init__()
        self.vector_size = vector_size
        self.model = nn.Sequential(
            nn.Linear(2*vector_size, vector_size),
            nn.ReLU()
        )