
import torch
import os
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from aijack.attack.inversion import GradientInversion_Attack
from model_data import LeNet, prepare_dataloader

import torch.nn.functional as F


#Setting seed
torch.manual_seed(7777)

SCORE_DIR = "../scores"

shape_img = (28, 28)
num_classes = 10
channel = 1
hidden = 588
criterion = nn.CrossEntropyLoss()
num_seeds = 5

device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"

#Loading pictures
dataloader = prepare_dataloader()

print("Loading one batch...")
for data in tqdm(dataloader, desc="Reading batch"):
    xs, ys = data[0], data[1]
    break

batch_size = 2
x_batch = xs[:batch_size]
y_batch = ys[:batch_size]


#Inversion attack
print("Initializing Gradient Inversion Attack Manager...")

net = LeNet(channel=channel, hideen=hidden, num_classes=num_classes)
pred = net(x_batch)
loss = criterion(pred, y_batch)
received_gradients = torch.autograd.grad(loss, net.parameters())
received_gradients = [cg.detach() for cg in received_gradients]

gradinversion = GradientInversion_Attack(
    net,
    (1, 28, 28),
    num_iteration=1000,
    lr=1e2,
    log_interval=0,
    optimizer_class=torch.optim.SGD,
    distancename="l2",
    optimize_label=False,
    bn_reg_layers=[net.body[1], net.body[4], net.body[7]],
    group_num=3,
    tv_reg_coef=0.00,
    l2_reg_coef=0.0001,
    bn_reg_coef=0.001,
    gc_reg_coef=0.001,
)
result = gradinversion.group_attack(received_gradients, batch_size=batch_size)


#Printing images

print("Starting FedAVG Training with Gradient Inversion Attack...")
print("-" * 60)
custom_order = [0,1]
num_rows = 2
num_cols = batch_size

fig = plt.figure(figsize=(num_cols*2, num_rows*2))

# First row: averaged results
for bid in range(batch_size):
    ax = fig.add_subplot(num_rows, num_cols, bid + 1)
    avg_img = (sum(result[0]) / len(result[0])).detach().cpu()[bid][0]
    ax.imshow(avg_img, cmap="gray")
    ax.axis("off")
    if bid == 0:
        ax.set_ylabel("Result", fontsize=12)

# Second row: original images in custom order
for plot_idx, batch_idx in enumerate(custom_order):
    ax = fig.add_subplot(num_rows, num_cols, num_cols + plot_idx + 1)  # second row
    ax.imshow(x_batch[batch_idx].detach().numpy()[0], cmap="gray")
    ax.axis("off")
    if plot_idx == 0:
        ax.set_ylabel("Original", fontsize=12)

plt.tight_layout()
plt.savefig("combined_figure.png")


scores = []
for i in range(batch_size):
    scores.append(F.mse_loss((sum(result[0]) / len(result[0])).detach().cpu()[i][0], x_batch[custom_order[i]][0].cpu()).item())

avg = round(sum(scores) / len(scores), 2)


file_path = os.path.join(SCORE_DIR, "mse_scores.txt")

with open(file_path, "w") as f:
    f.write("MSE scores per image:\n")
    for i, s in enumerate(scores):
        f.write(f"Image {i}: {s:.6f}\n")
    f.write(f"\nAverage MSE: {avg:.2f}\n")
