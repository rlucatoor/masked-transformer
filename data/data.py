import torch

from consts import START_TOKEN, END_TOKEN, PADDING_TOKEN, UNKNOWN_TOKEN


# get text
with open('data/shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# extract all possible characters
chars = sorted(list(set(text)))
# append start- and end-of-text tokens, padding token and unknown token
chars = [START_TOKEN, END_TOKEN, PADDING_TOKEN, UNKNOWN_TOKEN] + chars
vocab_size = len(chars)

# string to integer and integer to string dicts
vocab = { char: ix for ix, char in enumerate(chars) }
inverse_vocab = { ix: char for ix, char in enumerate(chars) }

# encode and decode functions to move from text to encoded data and vice versa
encode = lambda string : [ vocab.get(char, UNKNOWN_TOKEN) for char in string ]
decode = lambda ixs : ''.join([ inverse_vocab[ix] for ix in ixs ])

# encode data and turn it into a tensor
data = torch.tensor(encode(text), dtype=torch.long)

# train/validation split
train_data = data[:int(0.9*len(data))]
eval_data = data[int(0.9*len(data)):]


# get batches of:
# - encoder inputs (x values)
# - decoder inputs (start token + encoder input shifted right)
# - ground truth data (y values)
def get_batch(split, B, T, device):
    data = eval_data if split == 'eval' else train_data
    ixs = torch.randint(0, len(data)-T, (B,))
    # get x values
    xb = torch.stack([ data[ix:ix+T ] for ix in ixs ])
    # get decoder inputs by shifting x values to the right and 
    # prepending the start token
    tgt_input = torch.full(
        (B, 1), vocab[START_TOKEN], dtype=torch.long, device=device
    )
    tgt_input = torch.cat((tgt_input, xb[:, :-1]), dim=1)
    # get y values
    yb = torch.stack([ data[ix+1:ix+T+1] for ix in ixs ])
    xb, tgt_input, yb = xb.to(device), tgt_input.to(device), yb.to(device)

    return xb, tgt_input, yb