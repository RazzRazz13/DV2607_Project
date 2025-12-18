import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from aijack.collaborative.fedavg import FedAVGAPI, FedAVGClient, FedAVGServer
from aijack.attack.inversion import GradientInversionAttackServerManager
from torch.utils.data import DataLoader, TensorDataset
from aijack.attack.inversion import GradientInversion_Attack
from model_data import LeNet, prepare_dataloader
import torch.optim as optim
from PIL import Image
import numpy as np
import torchvision.transforms as T

torch.manual_seed(7777)

shape_img = (28, 28)
num_classes = 10
channel = 1
hidden = 588
criterion = nn.CrossEntropyLoss()
num_seeds = 5

device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"


dataloader = prepare_dataloader()
for data in dataloader:
    xs, ys = data[0], data[1]
    break
img = Image.open("rj.jpg")      # <-- replace with your file
img = img.convert("L")                          # convert to grayscale if needed


transform = T.Compose([
    T.Grayscale(),               # convert to 1 channel
    T.Resize((28, 28)),          # must match MNIST
    T.ToTensor(),                # -> tensor [1,28,28]
])
# Convert to tensor with shape [1, 1, H, W]
x = transform(img)              # x is [1,28,28]
x = x.unsqueeze(0)    
y = torch.tensor([0])
#x = xs[:1]
#y = ys[:1]

fig = plt.figure(figsize=(1, 1))
plt.axis("off")
plt.imshow(x.detach().numpy()[0][0], cmap="gray")
plt.savefig("rj_2.png")


manager = GradientInversionAttackServerManager(
    (1, 28, 28),
    num_trial_per_communication=5,
    log_interval=50,
    num_iteration=100,
    tv_reg_coef=0.01,
    distancename="cossim",
    device=device,
    lr=1.0,
)
GSFedAVGServer = manager.attach(FedAVGServer)

client = FedAVGClient(
    LeNet(channel=channel, hideen=hidden, num_classes=num_classes).to(device),
    lr=1.0,
    device=device,
)
server = GSFedAVGServer(
    [client],
    LeNet(channel=channel, hideen=hidden, num_classes=num_classes).to(device),
    lr=1.0,
    device=device,
)

local_dataloaders = [DataLoader(TensorDataset(x, y))]
local_optimizers = [optim.SGD(client.parameters(), lr=1.0)]

api = FedAVGAPI(
    server,
    [client],
    criterion,
    local_optimizers,
    local_dataloaders,
    num_communication=1,
    local_epoch=1,
    use_gradients=True,
    device=device,
)

api.run()

fig = plt.figure(figsize=(5, 2))
for s, result in enumerate(server.attack_results[0]):
    ax = fig.add_subplot(1, len(server.attack_results[0]), s + 1)
    ax.imshow(result[0].cpu().detach().numpy()[0][0], cmap="gray")
    ax.axis("off")
plt.tight_layout()
plt.savefig("single_picture.png")
plt.show()