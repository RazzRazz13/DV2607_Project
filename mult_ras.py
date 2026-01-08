import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from aijack.attack.inversion import GradientInversion_Attack
from model_data import LeNet

# -----------------------------
# MAIN
# -----------------------------
torch.manual_seed(7777)

num_classes = 10
channel = 1
hidden = 588
criterion = nn.CrossEntropyLoss()

device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"

# -----------------------------
# PATHS
# -----------------------------
BASE_DIR = "base_pic"
REC_DIR = "rec_pic"
os.makedirs(REC_DIR, exist_ok=True)

image_paths = [
    "rd.jpg",
    "rj.jpg",
    "rl.jpg",
    "rn.jpg",
    "re.png",
    "rj.jpg"
]

# -----------------------------
# LOAD LOCAL IMAGES
# -----------------------------
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
y_batch = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long)
batch_size = x_batch.size(0)

# -----------------------------
# GRADIENT INVERSION ATTACK
# -----------------------------
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

result = gradinversion.group_attack(
    received_gradients,
    batch_size=batch_size
)

avg_result = sum(result[0]) / len(result[0])  # average groups

# -----------------------------
# COMBINED VISUALIZATION
# -----------------------------
custom_order = [2, 1, 3, 4, 5, 0]
num_rows = 2
num_cols = batch_size

fig = plt.figure(figsize=(num_cols * 2, num_rows * 2))

# Reconstructed
for i in range(batch_size):
    ax = fig.add_subplot(num_rows, num_cols, i + 1)
    ax.imshow(avg_result[i][0].cpu(), cmap="gray")
    ax.axis("off")
    if i == 0:
        ax.set_ylabel("Result")

# Original (custom order)
for plot_idx, batch_idx in enumerate(custom_order):
    ax = fig.add_subplot(num_rows, num_cols, num_cols + plot_idx + 1)
    ax.imshow(x_batch[batch_idx][0].cpu(), cmap="gray")
    ax.axis("off")
    if plot_idx == 0:
        ax.set_ylabel("Original")

plt.tight_layout()
plt.savefig(os.path.join(REC_DIR, "combined_figure.png"))
plt.close()
