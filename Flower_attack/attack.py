import torch
from config import ITERS, DEVICE, LR, TV

def gradient_inversion(model, target_grads, label, iters=ITERS, lr=LR, tv=TV):
    model.eval()

    dummy = torch.randn((1, 1, 28, 28), requires_grad=True, device=DEVICE)
    y = torch.tensor([label], device=DEVICE)

    optimizer = torch.optim.Adam([dummy], lr=lr)

    for _ in range(iters):
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

        (match + tv * tv_loss).backward()
        optimizer.step()
        dummy.data.clamp_(-1, 1)

    return dummy.detach().cpu()
