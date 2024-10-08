from torch import nn

from models.sublayers.position_wise_feed_forward_network import PositionWiseFeedForwardNetwork
from models.sublayers.multi_head_attention import MultiHeadAttention
from models.normalization.layer_norm import LayerNorm


class Decoder(nn.Module):

    def __init__(self, C, dim_ff, dim_k, dim_v, num_heads):
        super().__init__()
        self.ff_network = PositionWiseFeedForwardNetwork(C, dim_ff)
        self.layer_norm_1 = LayerNorm(C)
        self.layer_norm_2 = LayerNorm(C)
        self.layer_norm_3 = LayerNorm(C)
        self.self_multi_head_att = MultiHeadAttention(
            C, dim_k//num_heads, dim_v//num_heads, num_heads, masked=True
        )
        # in the Attention Is All You Need paper, this attention block is not masked, however
        # failing to mask the attention leads to unstable training for sequence
        # completion tasks
        self.cross_multi_head_att = MultiHeadAttention(
            C, dim_k//num_heads, dim_v//num_heads, num_heads, masked=True
        )

    def forward(self, output_embs, enc_output):
        # 1. compute self-attention output via Masked Multi-Head self-attention
        self_att_output = self.self_multi_head_att(q_embs=output_embs, k_embs=output_embs, v_embs=output_embs) # (B, T, C)
        # 2. apply residual connection and layer normalization to self-attention output
        self_att_output = self.layer_norm_1(self_att_output + output_embs) # (B, T, C)
        # 3. compute cross-attention output via Multi-Head cross-attention by using queries generated by
        #    the previous decoder layer and keys and values generated by the Encoder
        cross_att_output = self.cross_multi_head_att(q_embs=self_att_output, k_embs=enc_output, v_embs=enc_output) # (B, T, C)
        # 4. apply residual connection and layer normalization to cross-attention output
        cross_att_output = self.layer_norm_2(cross_att_output + self_att_output) # (B, T, C)
        # 5. pass cross-attention output through a Feed-Forward network
        ff_output = self.ff_network(cross_att_output) # (B, T, C)
        # 6. apply residual connection and layer normalization to feed forward output
        ff_output = self.layer_norm_3(ff_output + cross_att_output) # (B, T, C)

        return ff_output