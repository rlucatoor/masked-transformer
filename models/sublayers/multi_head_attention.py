import torch
from torch import nn

from models.networks.scaled_dot_product_attention import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):

    def __init__(self, C, dim_k, dim_v, num_heads, masked):
        super().__init__()
        self.attention_heads = nn.ModuleList([
            ScaledDotProductAttention(C, dim_k, dim_v, masked) for _ in range(num_heads)
        ])
        self.linear_layer = nn.Linear(dim_k*num_heads, C)

    def forward(self, q_embs, k_embs, v_embs):
        # attention output
        single_att_outputs = [
            head(q_embs, k_embs, v_embs) for head in self.attention_heads # (B, T, dim_k)
        ]
        att_output = torch.cat(single_att_outputs, dim=-1) # (B, T, dim_k*num_heads)
        # projected attention output
        out = self.linear_layer(att_output) # (B, T, C)

        return out