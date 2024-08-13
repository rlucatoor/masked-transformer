import torch
import matplotlib.pyplot as plt

from data import vocab_size, get_batch, decode
from train_utils import estimate_loss
from models.transformer.transformer import Transformer


# params
B, T, C = 4, 32, 64
dim_k = 64
dim_v = dim_k
num_heads = 4
dim_ff = 256
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# how many steps apart should two consecutive evaluation steps be
eval_interval = 100
# iterations per evaluation step
eval_iters = 200
# how many iterations should be run
max_steps = 5000
  
m = Transformer(vocab_size, C, T, dim_ff, dim_k, dim_v, num_heads, device)
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)


# training
train_losses, eval_losses, iterations = [], [], []
for step in range(max_steps):
    # evaluate the loss every eval_interval-many steps
    if step % eval_interval == 0 or step == max_steps-1:
        losses = estimate_loss(m, B, T, device, eval_iters)
        train_losses.append(losses['train'])
        eval_losses.append(losses['eval'])
        iterations.append(step)

    # get batches of input and ground truths
    xb, tgt_input, yb = get_batch('train', B, T, device)
    # perform forward and backward steps
    logits, loss = m(xb, tgt_input, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# plots
plt.plot(iterations, train_losses, label='train loss', color='blue')
plt.plot(iterations, eval_losses, label='eval loss', color='red')
plt.legend()
plt.show()

starting_output = torch.zeros((1, 1), dtype=torch.long)
output = m.generate(starting_output, 200)
print(decode(output[0].tolist()))
