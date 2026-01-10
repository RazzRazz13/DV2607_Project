import torch
import torch.nn as nn
import os
from matplotlib import pyplot as plt
from aijack.collaborative.fedavg import FedAVGAPI, FedAVGClient, FedAVGServer
from aijack.attack.inversion import GradientInversionAttackServerManager
from torch.utils.data import DataLoader, TensorDataset
from model_data import LeNet
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F

#Setting seed
torch.manual_seed(7777)

shape_img = (28, 28)
num_classes = 10
channel = 1
hidden = 588

num_seeds = 5

device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"


#Loading picture
BASE_DIR = "../base_pic"
REC_DIR = "../rec_pic"
SCORE_DIR = "../scores"
os.makedirs(REC_DIR, exist_ok=True)

image_path = "rj.jpg"

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

img = Image.open(os.path.join(BASE_DIR, image_path)).convert("RGB")
x = transform(img)
x = x.unsqueeze(0)    
y = torch.tensor([0])

criterion = nn.CrossEntropyLoss()

#Inversion attack
manager = GradientInversionAttackServerManager(
    (1, 28, 28),
    num_trial_per_communication=5,
    log_interval=20,
    num_iteration=100,
    distancename="l2",
    device=device,
    lr=1.0,
)
DLGFedAVGServer = manager.attach(FedAVGServer)

client = FedAVGClient(
    LeNet(channel=channel, hideen=hidden, num_classes=num_classes).to(device),
    lr=1.0,
    device=device,
)
server = DLGFedAVGServer(
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


#Printing images
print("Plotting pictures")
fig = plt.figure(figsize=(5, 2))
ax = fig.add_subplot(1, len(server.attack_results[0]) + 1, 1)
ax.imshow(x[0][0].detach().cpu(), cmap="gray")
ax.axis("off")
for s, result in enumerate(server.attack_results[0]):
    ax = fig.add_subplot(1, len(server.attack_results[0]) + 1, s + 2)
    ax.imshow(result[0][0][0].detach().cpu(), cmap="gray")
    ax.axis("off")
plt.savefig(os.path.join(REC_DIR, "single_picture.png"))
plt.tight_layout()
plt.close()


scores = []

def normalize(img):
    img_min = img.min()
    img_max = img.max()
    return (img - img_min) / (img_max - img_min + 1e-8)

for result in server.attack_results[0]:
    pred = result[0][0][0].detach().cpu()
    target = x[0][0].detach().cpu()
    pred = normalize(pred)
    scores.append(F.mse_loss(pred, target).item())

avg = round(sum(scores) / len(scores), 2)

file_path = os.path.join(SCORE_DIR, "mse_scores.txt")

with open(file_path, "w") as f:
    f.write("MSE scores per image:\n")
    for i, s in enumerate(scores):
        f.write(f"Image {i}: {s:.6f}\n")
    f.write(f"\nAverage MSE: {avg:.6f}\n")  # more precision