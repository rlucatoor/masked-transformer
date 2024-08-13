from torch import nn

from models.sublayers.position_wise_feed_forward_network import PositionWiseFeedForwardNetwork
from models.sublayers.multi_head_attention import MultiHeadAttention
from models.normalization.layer_norm import LayerNorm


class Encoder(nn.Module):

    def __init__(self, C, dim_ff, dim_k, dim_v, num_heads):
        super().__init__()
        self.layer_norm_1 = LayerNorm(C)
        self.layer_norm_2 = LayerNorm(C)
        self.ff_network = PositionWiseFeedForwardNetwork(C, dim_ff)
        # in the Attention Is All You Need paper, this attention block is not masked, however
        # failing to mask the attention leads to unstable training
        self.multi_head_att = MultiHeadAttention(C, dim_k, dim_v, num_heads, masked=True)

    def forward(self, input_embs):
        # 1. compute self-attention output via Multi-Head self-attention
        self_att_output = self.multi_head_att(q_embs=input_embs, k_embs=input_embs, v_embs=input_embs) # (B, T, C)
        # 2. apply residual connection and layer normalization to self-attention output
        self_att_output = self.layer_norm_1(self_att_output + input_embs) # (B, T, C)
        # 3. pass self-attention output through a Feed-Forward network
        ff_output = self.ff_network(self_att_output) # (B, T, C)
        # 4. apply residual connection and layer normalization to feed forward output
        ff_output = self.layer_norm_2(ff_output + self_att_output) # (B, T, C)

        return ff_output