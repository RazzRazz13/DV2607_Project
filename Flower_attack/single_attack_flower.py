# ============================================================
# Gradient Leakage Attack in Flower (v1.25.0 compatible)
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
import json
import base64

from PIL import Image
from torch.utils.data import Dataset, DataLoader

import flwr as fl
from flwr.common import Context


# ============================================================
# Global configuration
# ============================================================

DEVICE = torch.device("cpu")

IMG_SHAPE = (28, 28)
CHANNELS = 1
HIDDEN_DIM = 588 
NUM_CLASSES = 10
ITERATIONS = 3000
NUM_CLIENTS = 1
LEARNING_RATE = 0.005
TV_COEFF = 0.01

CRITERION = nn.CrossEntropyLoss()


torch.manual_seed(7777)

# ============================================================
# Dataset wrapper for NumPy arrays
# ============================================================

class NumpyDataset(Dataset):
    """Wrap NumPy arrays as a PyTorch Dataset."""

    def __init__(self, x, y=None, transform=None):
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        if self.transform is not None:
            x = self.transform(x)

        if self.y is None:
            return x
        return x, self.y[idx]


def prepare_dataloader(batch_size=64):
    dataset = torchvision.datasets.MNIST(
        root="MNIST",
        train=True,
        download=True,
    )

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,))
    ])

    wrapped = NumpyDataset(
        dataset.data.numpy(),
        dataset.targets.numpy(),
        transform=transform,
    )

    return DataLoader(wrapped, batch_size=batch_size, shuffle=True)


# ============================================================
# Model
# ============================================================

class LeNet(nn.Module):
    def __init__(self, channels=1, num_classes=10):
        super().__init__()
        act = nn.ReLU

        self.features = nn.Sequential(
            nn.Conv2d(channels, 6, 5),   # 28 → 24
            act(),
            nn.AvgPool2d(2),             # 24 → 12

            nn.Conv2d(6, 16, 5),         # 12 → 8
            act(),
            nn.AvgPool2d(2),             # 8 → 4
        )

        self.classifier = nn.Linear(16 * 4 * 4, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ============================================================
# Flower client
# ============================================================

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, x, y):
        self.model = model
        self.x = x
        self.y = y

    def get_parameters(self, config):
        return [p.detach().cpu().numpy() for p in self.model.parameters()]

    def fit(self, parameters, config):
        for p, w in zip(self.model.parameters(), parameters):
            p.data = torch.tensor(w)

        optimizer = optim.SGD(self.model.parameters(), lr=1.0)

        # one clean step (like aijack)
        optimizer.zero_grad()
        out = self.model(self.x)
        loss = CRITERION(out, self.y)
        loss.backward()
        optimizer.step()

        # recompute grads to leak
        self.model.zero_grad()
        out = self.model(self.x)
        loss = CRITERION(out, self.y)
        loss.backward()

        grads = [
            p.grad.detach().cpu().numpy().tolist()
            for p in self.model.parameters()
        ]

        return self.get_parameters(config), 1, {
            "grads": json.dumps(grads),
            "label": int(self.y.item()),
        }



# ============================================================
# Gradient inversion attack
# ============================================================

def gradient_inversion(model_fn, server_weights, target_grads, label):
    model = model_fn()
    for p, w in zip(model.parameters(), server_weights):
        p.data = torch.tensor(w)

    model.eval()

    dummy = torch.randn((1,1,28,28), requires_grad=True)
    y = torch.tensor([label])

    optimizer = optim.Adam([dummy], lr=0.1)

    target_grads = [
        torch.tensor(g) for g in target_grads
    ]

    for i in range(300):
        optimizer.zero_grad()
        out = model(dummy)
        loss = CRITERION(out, y)

        grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)

        match = 0
        for g, tg in zip(grads, target_grads):
            g = g / (g.norm() + 1e-8)
            tg = tg / (tg.norm() + 1e-8)
            match += 1 - torch.sum(g * tg)

        tv = (
            torch.abs(dummy[:, :, :-1] - dummy[:, :, 1:]).sum()
            + torch.abs(dummy[:, :, :, :-1] - dummy[:, :, :, 1:]).sum()
        )

        total = match + 1e-4 * tv
        total.backward()
        optimizer.step()

    return dummy.detach().cpu()


# ============================================================
# Custom strategy
# ============================================================

class InterceptGradFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, model_fn, **kwargs):
        super().__init__(**kwargs)
        self.model_fn = model_fn
        self.reconstructed = None

    def aggregate_fit(self, rnd, results, failures):
        # Check if we have any successful results
        if len(results) > 0:
            fit_res = results[0][1]
            if fit_res.metrics is not None:
                grads_json = fit_res.metrics.get("grads_json", None)
                label = fit_res.metrics.get("label", None)
                
                if grads_json is not None and label is not None:
                    print("[Server] Running gradient inversion attack...")
                    # Deserialize gradients from JSON
                    grads = json.loads(grads_json)
                    self.reconstructed = gradient_inversion(
                        self.model_fn,
                        grads,
                        label,
                    )

                    # img = self.reconstructed.squeeze().numpy()
                    # plt.imsave("flower_reconstructed.png", img, cmap="gray")
                    img = self.reconstructed.squeeze(0).squeeze(0).numpy()
                    plt.figure(figsize=(5, 5))              # control visual size
                    plt.imshow(img, cmap="gray", interpolation="nearest")
                    plt.axis("off")
                    plt.savefig("flower_reconstructed.png", dpi=200, bbox_inches="tight", pad_inches=0)
                    plt.close()

                    print("[Server] Saved flower_reconstructed.png")
        else:
            print("[Server] No successful client results to attack")

        return super().aggregate_fit(rnd, results, failures)

class AttackFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, model_fn, **kwargs):
        super().__init__(**kwargs)
        self.model_fn = model_fn
        self.server_weights = None

    def aggregate_fit(self, rnd, results, failures):
        aggregated = super().aggregate_fit(rnd, results, failures)

        # SAVE SERVER WEIGHTS
        if aggregated is not None:
            self.server_weights = aggregated[0]

        if results:
            fit_res = results[0][1]
            grads = json.loads(fit_res.metrics["grads"])
            label = fit_res.metrics["label"]

            recon = gradient_inversion(
                self.model_fn,
                self.server_weights,
                grads,
                label,
            )

            plt.imshow(recon[0,0], cmap="gray")
            plt.axis("off")
            plt.savefig("flower_reconstructed.png", dpi=200)
            plt.close()

        return aggregated


# ============================================================
# Main
# ============================================================

def main():
    try:
        img = Image.open("rj.jpg").convert("L")
        transform = T.Compose([
            T.Resize(IMG_SHAPE),
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,))
        ])
        x = transform(img).unsqueeze(0)
        y = torch.tensor([0])
    except Exception:
        loader = prepare_dataloader()
        x, y = next(iter(loader))
        x, y = x[:1], y[:1]

    def model_fn():
        return LeNet(CHANNELS, HIDDEN_DIM, NUM_CLASSES)

    strategy = InterceptGradFedAvg(
        model_fn=model_fn,
        fraction_fit=1.0,
        min_fit_clients=1,
        min_available_clients=1,
    )

    def client_fn(context: Context):
        return FlowerClient(model_fn(), x, y).to_client()

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=1),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
