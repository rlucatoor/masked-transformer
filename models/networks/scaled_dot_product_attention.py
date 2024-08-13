import torch
from torch import nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):

    def __init__(self, C, dim_k, dim_v, masked=True):
        super().__init__()
        self.dim_k, self.masked = dim_k, masked
        # learned linear projections that calculate queries, keys and values
        self.w_q = nn.Linear(C, dim_k, bias=False)
        self.w_k = nn.Linear(C, dim_k, bias=False)
        self.w_v = nn.Linear(C, dim_v, bias=False)

    def forward(self, q_embs, k_embs, v_embs):

        B, T, C = v_embs.shape
        
        # make sure tril isn't registered as a model parameter (so it doesn't
        # get trained by the optimizer)
        self.register_buffer('tril', torch.tril(torch.ones(T, T)))
        
        # get keys, queries and values
        q = self.w_q(q_embs) # (B, T, dim_k)
        k = self.w_k(k_embs) # (B, T, dim_k)
        v = self.w_v(v_embs) # (B, T, dim_v)
        
        # raw attention scores
        raw_scores = q @ k.transpose(-2, -1) # (B, T, dim_k) @ (B, dim_k, T) -> (B, T, T)        
        # scaled attention scores (used to prevent the softmax layer from converging towards
        # a one-hot encoded vector)
        scaled_scores = raw_scores * C**-0.5 # (B, T, T)
        # if masked==True, this becomes a Masked Scaled Dot Product Attention layer, meaning
        # masking (past tokens cannot communicate with future tokens) is applied;
        if self.masked:
            scaled_scores = scaled_scores.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        # weights
        wei = F.softmax(scaled_scores, dim=-1) # (B, T, T)
        # weighted values
        out = wei @ v # (B, T, dim_v)

        return out