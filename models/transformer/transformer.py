import torch
from torch import nn
import torch.nn.functional as F

from models.blocks.encoder import Encoder
from models.blocks.decoder import Decoder
from data import vocab
from consts import START_TOKEN


class Transformer(nn.Module):

    def __init__(self, vocab_size, C, T, dim_ff, dim_k, dim_v, num_heads, device):
        super().__init__()
        self.device = device
        self.linear_layer = nn.Linear(C, vocab_size)
        self.T = T
        self.token_embs = nn.Embedding(vocab_size, C) # (vocab_size, vocab_size)
        # TODO: use fixed positional embeddings set to sine and cosine functions
        self.positional_embs = nn.Embedding(T, C)
        self.encoder = Encoder(C, dim_ff, dim_k, dim_v, num_heads)
        self.decoder = Decoder(C, dim_ff, dim_k, dim_v, num_heads)


    def forward(self, enc_input, dec_input, targets=None):
        # if enc_input or dec_input are longer than T tokens, truncate them to the last T tokens
        trunc_enc_input = enc_input[:, -self.T:]
        trunc_dec_input = dec_input[:, -self.T:]
        # compute input embeddings (input token embeddings + input positional embeddings)
        input_token_embs = self.token_embs(trunc_enc_input) # (B, T, C)
        input_pos_embs = self.positional_embs(
            torch.arange(trunc_enc_input.shape[-1], device=self.device) # (T, C)
        )
        input_embs = input_token_embs + input_pos_embs # (B, T, C)
        # compute output embeddings (output token embeddings + output positional embeddings)
        output_token_embs = self.token_embs(trunc_dec_input) # (B, T, C)
        output_pos_embs = self.positional_embs(
            torch.arange(trunc_dec_input.shape[-1], device=self.device) # (T, C)
        )
        output_embs = output_token_embs + output_pos_embs # (B, T, C)
        # run input embeddings through encoder block
        encoder_output = self.encoder(input_embs) # (B, T, C)
        # run output embeddings and encoder output through the decoder block
        decoder_output = self.decoder(output_embs, encoder_output)
        # run decoder output through a linear layer to get logits
        logits = self.linear_layer(decoder_output) # (B, T, vocab_size)

        if targets == None:
            loss = None
        else:
            B, T, vocab_size = logits.shape
            logits = logits.view(B*T, vocab_size) # (B*T, vocab_size)
            targets = targets.view(B*T) # (B*T)
            probs = F.softmax(logits, dim=1) # (B*T, vocab_size)
            logprobs = torch.log(probs) # (B*T, vocab_size)
            relevant_logprobs = logprobs[torch.arange(B*T), targets] # (B*T)
            #loss = F.cross_entropy(logits, targets)
            loss = -sum(relevant_logprobs)/len(relevant_logprobs)

        return logits, loss


    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        out = idx # (len(idx)), increases at each iteration
        
        for _ in range(max_new_tokens):
            # get decoder inputs by shifting x values to the right and 
            # prepending the start token
            tgt_input = torch.full(
                (1,1), vocab[START_TOKEN], dtype=torch.long, device=self.device
            )
            tgt_input = torch.cat((tgt_input, out[:,:-1]), dim=1)
            # compute logits with a forward pass
            logits, loss = self(out, tgt_input) # (B, len(out), vocab_size)
            # only focus on the last token's logits
            logits = logits[:, -1, :] # (B, vocab_size)
            # generate a probability distribution for the next token
            probs = F.softmax(logits, dim=-1) # (B, vocab_size)
            # sample next token from probability distribution
            next_token = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled token to output
            out = torch.cat((out, next_token), dim=1) # (B, len(out) + 1)
            
        return out