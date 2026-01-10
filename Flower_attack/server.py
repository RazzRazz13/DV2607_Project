""" Flower server with malicious FedAvg strategy performing gradient inversion attack. """

import os
import json
from collections import OrderedDict
import torch
import torch.nn.functional as F
import flwr as fl
import matplotlib.pyplot as plt
from model import LeNet
from config import NUM_ROUNDS, DEVICE
from attack import gradient_inversion

class MaliciousFedAvg(fl.server.strategy.FedAvg):
    """ Custom FedAvg strategy that performs gradient inversion attack. """

    def __init__(self, x_original, client_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x_original = x_original.cpu()
        self.client_name = client_name

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            print("[Server] No client results received")
            return super().aggregate_fit(server_round, results, failures)

        _, fit_res = results[0]

        # ---- Abort-safe guard (DetectingClient) ----
        if fit_res.metrics.get("aborted", False):
            os.makedirs("Flower_attack", exist_ok=True)
            with open("Flower_attack/mse_results.txt", "a") as f:
                f.write(f"{self.client_name},ABORTED\n")

            print(f"[Server] {self.client_name} aborted — logged result")
            return super().aggregate_fit(server_round, results, failures)

        assert fit_res.num_examples == 1

        # ---- Rebuild model ----
        model = LeNet().to(DEVICE)
        params = fl.common.parameters_to_ndarrays(fit_res.parameters)

        state_dict = OrderedDict(
            (k, torch.tensor(v))
            for k, v in zip(model.state_dict().keys(), params)
        )
        model.load_state_dict(state_dict)

        # ---- Load gradients ----
        grads_raw = json.loads(fit_res.metrics["grads"])
        target_grads = [
            torch.tensor(g, device=DEVICE) if g is not None else None
            for g in grads_raw
        ]

        label = fit_res.metrics["label"]

        # ---- Final round: attack + MSE ----
        if server_round == NUM_ROUNDS:
            print(f"[Server] Running gradient inversion ({self.client_name})")

            img = gradient_inversion(model, target_grads, label)

            mse = F.mse_loss(img.cpu(), self.x_original).item()

            os.makedirs("Flower_attack", exist_ok=True)

            # ---- Append MSE result ----
            with open("Flower_attack/mse_results.txt", "a") as f:
                f.write(f"{self.client_name},{mse:.6f}\n")

            # ---- Save reconstructed image ----
            img_path = f"Flower_attack/reconstructed_{self.client_name}.png"
            plt.imshow(img[0, 0], cmap="gray")
            plt.axis("off")
            plt.savefig(img_path, dpi=200)
            plt.close()

            print(f"[Server] Saved {img_path}")
            print(f"[Server] MSE = {mse:.6f}")

        return super().aggregate_fit(server_round, results, failures)