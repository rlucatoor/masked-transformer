import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    
    def __init__(self, output_dim, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon
        self.gamma = nn.Parameter(torch.ones(output_dim))
        self.beta = nn.Parameter(torch.ones(output_dim))
        
    def forward(self, x):
        # calculate mean and variance across the channel dimension
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        # subtract mean from x and divide it by (variance + epsilon), so 
        # that the output has 0 mean and unit variance. Epsilon is added
        # to avoid division by zero errors
        x_hat = (x - mean) / torch.sqrt(variance+self.epsilon)
        # add learnable parameters
        x_hat = self.gamma * x_hat + self.beta
        
        return x_hat