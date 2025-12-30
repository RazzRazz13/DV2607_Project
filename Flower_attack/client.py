import torch
import json
import flwr as fl
from model import LeNet
from config import DEVICE

class HonestClient(fl.client.NumPyClient):
    def __init__(self, x, y):
        self.model = LeNet().to(DEVICE)
        self.x = x.to(DEVICE)
        self.y = y.to(DEVICE)

    def get_parameters(self, config):
        return [p.detach().cpu().numpy() for p in self.model.state_dict().values()]

    def fit(self, parameters, config):
        # Load global parameters
        state_dict = {
            k: torch.tensor(v)
            for k, v in zip(self.model.state_dict().keys(), parameters)
        }
        self.model.load_state_dict(state_dict, strict=True)

        # self.model.train()
        self.model.eval()
        self.model.zero_grad()

        # 🔑 Compute gradients ONCE, no optimizer step
        loss = torch.nn.functional.cross_entropy(
            self.model(self.x), self.y
        )
        loss.backward()

        grads = [
            p.grad.detach().cpu().numpy().tolist()
            for p in self.model.parameters()
        ]

        return self.get_parameters(config), 1, {
            "grads": json.dumps(grads),
            "label": int(self.y.item())
        }

    def evaluate(self, parameters, config):
        return 0.0, 1, {}
