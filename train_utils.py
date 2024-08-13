import torch

from data import get_batch


# function to estimate the train and eval loss
@torch.no_grad()
def estimate_loss(model, B, T, device, eval_iters):
    # set model to evaluation mode
    model.eval()
    out = {}

    for split in ['train', 'eval']:
        losses = torch.zeros(eval_iters)
        for iter in range(eval_iters):
            xb, tgt_input, yb = get_batch(split, B, T, device)
            logits, loss = model(xb, tgt_input, yb)
            losses[iter] = loss.item()
        out[split] = losses.mean()

    # set model back to train mode
    model.train()

    return out