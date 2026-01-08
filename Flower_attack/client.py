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

    # --------------------------------------------------
    # Flower required methods
    # --------------------------------------------------
    def get_parameters(self, config):
        return [p.detach().cpu().numpy() for p in self.model.state_dict().values()]

    def evaluate(self, parameters, config):
        return 0.0, 1, {}

    # --------------------------------------------------
    # Detection utilities
    # --------------------------------------------------
    def _collect_layer_stats(self):
        """
        Collect statistics from attack-relevant layers.
        Returns dict: layer_name -> (mean, std, max_abs)
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

    def _z_score_between(self, early, late, eps=1e-8):
        """
        Compute max Z-score between two statistic snapshots.
        """
        z_max = 0.0

        for name in early:
            if name not in late:
                continue

            e_mean, e_std, e_max = early[name]
            l_mean, l_std, l_max = late[name]

            z_mean = abs(l_mean - e_mean) / (abs(e_mean) + eps)
            z_std  = abs(l_std  - e_std)  / (abs(e_std)  + eps)
            z_maxv = abs(l_max  - e_max)  / (abs(e_max)  + eps)

            z_max = max(z_max, z_mean, z_std, z_maxv)

        return z_max

    # --------------------------------------------------
    # Stateless warm-up detection
    # --------------------------------------------------
    def warmup_detect(self, steps=5, lr=1e-3, z_thresh=10.0):
        """
        Returns True if model appears malicious.
        """
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        stats_history = []

        self.model.train()

        for _ in range(steps):
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(
                self.model(self.x), self.y
            )
            loss.backward()
            optimizer.step()

            stats_history.append(self._collect_layer_stats())

        # Compare first vs last warm-up step
        z = self._z_score_between(
            stats_history[0],
            stats_history[-1]
        )

        print(f"[Client] Warm-up Z-score: {z:.2f}")

        return z > z_thresh

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
            print("[Client] ⚠️ Malicious model detected — aborting round")
            return self.get_parameters(config), 0, {"aborted": True}

        # --------------------------------------------------
        # Normal gradient computation (for attack demo)
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

class ProjectedClient(fl.client.NumPyClient):
    def __init__(self, x, y, rank_ratio=0.3):
        """
        rank_ratio: fraction of gradient entries to keep (0 < rank_ratio ≤ 1)
        """
        self.model = LeNet().to(DEVICE)
        self.x = x.to(DEVICE)
        self.y = y.to(DEVICE)
        self.rank_ratio = rank_ratio

    # --------------------------------------------------
    # Flower required methods
    # --------------------------------------------------
    def get_parameters(self, config):
        return [
            p.detach().cpu().numpy()
            for p in self.model.state_dict().values()
        ]

    def evaluate(self, parameters, config):
        return 0.0, 1, {}

    # --------------------------------------------------
    # Gradient projection defense
    # --------------------------------------------------
    def _project_gradient(self, grad):
        """
        Randomly keep only a subset of gradient entries.
        """
        flat = grad.view(-1)
        d = flat.numel()
        k = max(1, int(d * self.rank_ratio))

        idx = torch.randperm(d, device=flat.device)[:k]

        projected = torch.zeros_like(flat)
        projected[idx] = flat[idx]

        return projected.view_as(grad)

    # --------------------------------------------------
    # Fit
    # --------------------------------------------------
    def fit(self, parameters, config):
        # Load global parameters
        state_dict = {
            k: torch.tensor(v)
            for k, v in zip(
                self.model.state_dict().keys(),
                parameters
            )
        }
        self.model.load_state_dict(state_dict, strict=True)

        self.model.train()
        self.model.zero_grad()

        # Normal forward/backward
        loss = torch.nn.functional.cross_entropy(
            self.model(self.x), self.y
        )
        loss.backward()

        # 🔐 Apply gradient subspace projection
        grads = []
        for p in self.model.parameters():
            if p.grad is None:
                grads.append(None)
                continue

            g_proj = self._project_gradient(p.grad)
            grads.append(g_proj.detach().cpu().numpy().tolist())

        return self.get_parameters(config), 1, {
            "grads": json.dumps(grads),
            "label": int(self.y.item())
        }