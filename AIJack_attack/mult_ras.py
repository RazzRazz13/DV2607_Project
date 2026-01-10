import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from aijack.attack.inversion import GradientInversion_Attack
from model_data import LeNet

import torch.nn.functional as F


#Setting seed
torch.manual_seed(7777)

num_classes = 10
channel = 1
hidden = 588
criterion = nn.CrossEntropyLoss()

device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"


#Loading pictures
BASE_DIR = "../base_pic"
REC_DIR = "../rec_pic"
SCORE_DIR = "../scores"
os.makedirs(REC_DIR, exist_ok=True)

image_paths = [
    "rd.jpg",
    "rj.jpg",
]

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

images = []
for name in image_paths:
    img = Image.open(os.path.join(BASE_DIR, name)).convert("RGB")
    img = transform(img)
    images.append(img)

x_batch = torch.stack(images)  # (B,1,28,28)

# Dummy labels
y_batch = torch.tensor([0, 1], dtype=torch.long)
batch_size = x_batch.size(0)

#Inversion attack
net = LeNet(channel=channel, hideen=hidden, num_classes=num_classes)

pred = net(x_batch)
loss = criterion(pred, y_batch)

received_gradients = torch.autograd.grad(loss, net.parameters())
received_gradients = [g.detach() for g in received_gradients]

gradinversion = GradientInversion_Attack(
    net,
    (1, 28, 28),
    num_iteration=1000,
    lr=1e2,
    log_interval=50,
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

result = gradinversion.group_attack(
    received_gradients,
    batch_size=batch_size
)

#Printing images
custom_order = [1, 0]
num_rows = 2
num_cols = batch_size

fig = plt.figure(figsize=(num_cols * 2, num_rows * 2))

# First row: averaged results
for i in range(batch_size):
    ax = fig.add_subplot(num_rows, num_cols, i + 1)
    avg_img = (sum(result[0]) / len(result[0])).detach().cpu()[i][0]
    ax.imshow(avg_img, cmap="gray")
    ax.axis("off")
    if i == 0:
        ax.set_ylabel("Result")

# Second row: original images in custom order
for plot_idx, batch_idx in enumerate(custom_order):
    ax = fig.add_subplot(num_rows, num_cols, num_cols + plot_idx + 1)
    ax.imshow(x_batch[batch_idx][0].cpu(), cmap="gray")
    ax.axis("off")
    if plot_idx == 0:
        ax.set_ylabel("Original")

plt.tight_layout()
plt.savefig(os.path.join(REC_DIR, "combined_figure.png"))
plt.close()

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

