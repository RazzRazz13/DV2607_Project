""" Gradient inversion attack implementation. """

import torch
from config import ITERS, DEVICE, LR, TV

def gradient_inversion(model, target_grads, label):
    """
    Reconstruct input data from gradients via gradient inversion.
    Args:
        model: The neural network model.
        target_grads: The target gradients to match.
        label: The true label corresponding to the input data.
        iters: Number of optimization iterations.
        lr: Learning rate for the optimizer.
        tv: Weight for total variation regularization.
    Returns:
        Reconstructed input data as a torch tensor.
    """

    model.eval()

    dummy = torch.randn((1, 1, 28, 28), requires_grad=True, device=DEVICE)
    y = torch.tensor([label], device=DEVICE)

    optimizer = torch.optim.Adam([dummy], lr=LR)

    for _ in range(ITERS):
        optimizer.zero_grad()

        loss = torch.nn.functional.cross_entropy(model(dummy), y)
        cur_grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)

        match = 0.0
        for g, tg in zip(cur_grads, target_grads):
            if tg is None:
                continue
            match += ((g - tg) ** 2).sum()

        tv_loss = (
            torch.abs(dummy[:, :, :-1] - dummy[:, :, 1:]).sum()
            + torch.abs(dummy[:, :, :, :-1] - dummy[:, :, :, 1:]).sum()
        )

        (match + TV * tv_loss).backward()
        optimizer.step()
        dummy.data.clamp_(-1, 1)

    return dummy.detach().cpu()
