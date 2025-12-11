import cv2
import copy
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
import torch.optim as optim
from tqdm import tqdm      # use terminal-friendly tqdm

from aijack.collaborative.fedavg import FedAVGAPI, FedAVGClient, FedAVGServer
from aijack.attack.inversion import GradientInversionAttackServerManager
from torch.utils.data import DataLoader, TensorDataset
from aijack.utils import NumpyDataset
from aijack.attack.inversion import GradientInversion_Attack


class LeNet(nn.Module):
    def __init__(self, channel=3, hideen=768, num_classes=10):
        super(LeNet, self).__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(channel, 12, kernel_size=5, padding=5 // 2, stride=2),
            nn.BatchNorm2d(12),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
            nn.BatchNorm2d(12),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
            nn.BatchNorm2d(12),
            act(),
        )
        self.fc = nn.Sequential(nn.Linear(hideen, num_classes))

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def prepare_dataloader(path="MNIST/.", batch_size=64, shuffle=True):
    at_t_dataset_train = torchvision.datasets.MNIST(
        root=path, train=True, download=True
    )
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    dataset = NumpyDataset(
        at_t_dataset_train.train_data.numpy(),
        at_t_dataset_train.train_labels.numpy(),
        transform=transform,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


# -----------------------------
# MAIN
# -----------------------------
torch.manual_seed(7777)

shape_img = (28, 28)
num_classes = 10
channel = 1
hidden = 588
criterion = nn.CrossEntropyLoss()
num_seeds = 5

device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"

# -----------------------------
# LOAD DATA WITH PROGRESS
# -----------------------------
dataloader = prepare_dataloader()

print("Loading one batch...")
for data in tqdm(dataloader, desc="Reading batch"):
    xs, ys = data[0], data[1]
    break

batch_size = 4
x_batch = xs[:batch_size]
y_batch = ys[:batch_size]


# -----------------------------
# SET UP ATTACK MANAGER
# -----------------------------
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


# -----------------------------
# FEDAVG API WITH PROGRESS PRINTS
# -----------------------------
print("Starting FedAVG Training with Gradient Inversion Attack...")
print("-" * 60)
custom_order = [3,2,0,1]
batch_size = len(x_batch)  # make sure batch_size matches your data
num_rows = 2
num_cols = batch_size  # assuming batch_size matches len(x_batch)

fig = plt.figure(figsize=(num_cols*2, num_rows*2))

# First row: averaged results
for bid in range(batch_size):
    ax = fig.add_subplot(num_rows, num_cols, bid + 1)
    avg_img = (sum(result[0]) / len(result[0])).detach().numpy()[bid][0]
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
