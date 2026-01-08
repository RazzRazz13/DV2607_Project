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

class NoisyClient(fl.client.NumPyClient):
    def __init__(self, x, y):
        self.model = LeNet().to(DEVICE)
        self.x = x.to(DEVICE)
        self.y = y.to(DEVICE)

    def get_parameters(self, config):
        return [p.detach().cpu().numpy() for p in self.model.state_dict().values()]

    def fit(self, parameters, config):
        # -------------------------
        # Load global parameters
        # -------------------------
        state_dict = {
            k: torch.tensor(v)
            for k, v in zip(self.model.state_dict().keys(), parameters)
        }
        self.model.load_state_dict(state_dict, strict=True)

        self.model.train()
        self.model.zero_grad()

        # -------------------------
        # DEFENSE: input-space mixing
        # -------------------------
        x_real = self.x
        y_real = self.y

        # second, unrelated input
        x_noise = torch.randn_like(x_real) * 0.5
        
        alpha = 0.5  # strength of defense (0.3–0.7 works well)

        logits_real = self.model(x_real)
        logits_noise = self.model(x_noise)

        loss = (
            torch.nn.functional.cross_entropy(logits_real, y_real)
            + alpha * torch.nn.functional.cross_entropy(logits_noise, y_real)
        )

        loss.backward()

        # -------------------------
        # Send gradients
        # -------------------------
        grads = [
            p.grad.detach().cpu().numpy().tolist()
            if p.grad is not None else None
            for p in self.model.parameters()
        ]

        return self.get_parameters(config), 1, {
            "grads": json.dumps(grads),
            "label": int(y_real.item())
        }

    def evaluate(self, parameters, config):
        return 0.0, 1, {}
    
class DetectingClient(fl.client.NumPyClient):
    def __init__(self, x, y):
        self.model = LeNet().to(DEVICE)
        self.x = x.to(DEVICE)
        self.y = y.to(DEVICE)

        # persistent across rounds
        self.baseline_stats = None

    # --------------------------------------------------
    # Flower required
    # --------------------------------------------------
    def get_parameters(self, config):
        return [p.detach().cpu().numpy() for p in self.model.state_dict().values()]

    def evaluate(self, parameters, config):
        return 0.0, 1, {}

    # --------------------------------------------------
    # Detection utilities
    # --------------------------------------------------
    def _layer_stats(self):
        """
        Collect statistics from attack-relevant layers.
        Returns dict: name -> (mean, std, max)
        """
        stats = {}

        for name, param in self.model.named_parameters():
            if (
                "conv.0.weight" in name or
                "fc.weight" in name or
                "fc.bias" in name
            ):
                w = param.detach()
                stats[name] = (
                    w.mean().item(),
                    w.std().item(),
                    w.abs().max().item(),
                )

        return stats

    def _z_score(self, current, history, eps=1e-8):
        """
        Compute max Z-score across tracked layers.
        """
        z_max = 0.0

        for name in current:
            if name not in history:
                continue

            cur_mean, cur_std, cur_max = current[name]
            hist_means = torch.tensor([h[name][0] for h in history])
            hist_stds  = torch.tensor([h[name][1] for h in history])
            hist_maxs  = torch.tensor([h[name][2] for h in history])

            z_mean = abs(cur_mean - hist_means.mean().item()) / (hist_means.std().item() + eps)
            z_std  = abs(cur_std  - hist_stds.mean().item())  / (hist_stds.std().item()  + eps)
            z_maxv = abs(cur_max  - hist_maxs.mean().item())  / (hist_maxs.std().item()  + eps)

            z_max = max(z_max, z_mean, z_std, z_maxv)

        return z_max

    # --------------------------------------------------
    # Warm-up detection
    # --------------------------------------------------
    def warmup_detect(self, steps=5, lr=1e-3, z_thresh=10.0):
        """
        Returns True if model appears malicious.
        """
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        history = []

        self.model.train()

        for _ in range(steps):
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(
                self.model(self.x), self.y
            )
            loss.backward()
            optimizer.step()

            history.append(self._layer_stats())

        # first round establishes baseline
        if self.baseline_stats is None:
            self.baseline_stats = history
            return False

        # compute deviation
        z = self._z_score(history[-1], self.baseline_stats)
        print("Z-Score:", z)

        print(f"[Client] Max Z-score this round: {z:.2f}")

        if z > z_thresh:
            print("[Client] ⚠️ Malicious model detected")
            return True

        # update baseline with benign history
        self.baseline_stats.extend(history)
        return False

    # --------------------------------------------------
    # Fit
    # --------------------------------------------------
    def fit(self, parameters, config):
        # Load global parameters
        state_dict = {
            k: torch.tensor(v)
            for k, v in zip(self.model.state_dict().keys(), parameters)
        }
        self.model.load_state_dict(state_dict, strict=True)

        # 🔐 Detection phase
        suspicious = self.warmup_detect(
            steps=5,
            lr=1e-3,
            z_thresh=10.0
        )

        if suspicious:
            # abort this round
            return self.get_parameters(config), 0, {"aborted": True}

        # --------------------------------------------------
        # Normal gradient computation (for your attack demo)
        # --------------------------------------------------
        self.model.eval()
        self.model.zero_grad()

        loss = torch.nn.functional.cross_entropy(
            self.model(self.x), self.y
        )
        loss.backward()

        grads = [
            p.grad.detach().cpu().numpy().tolist()
            if p.grad is not None else None
            for p in self.model.parameters()
        ]

        return self.get_parameters(config), 1, {
            "grads": json.dumps(grads),
            "label": int(self.y.item())
        }