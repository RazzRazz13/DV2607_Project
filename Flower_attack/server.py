""" Flower server with malicious FedAvg strategy performing gradient inversion attack."""

import json
from collections import OrderedDict
import torch
import flwr as fl
from model import LeNet
import matplotlib.pyplot as plt
from config import NUM_ROUNDS, DEVICE
from attack import gradient_inversion

class MaliciousFedAvg(fl.server.strategy.FedAvg):
    """ Custom FedAvg strategy that performs gradient inversion attack. """
    def aggregate_fit(self, server_round, results, failures):
        if not results:
            print("[Server] No client results received")
            return super().aggregate_fit(server_round, results, failures)

        _, fit_res = results[0]

        assert fit_res.num_examples == 1

        model = LeNet().to(DEVICE)
        params = fl.common.parameters_to_ndarrays(fit_res.parameters)
        state_dict = OrderedDict(
            {k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), params)}
        )
        model.load_state_dict(state_dict)

        grads_raw = json.loads(fit_res.metrics["grads"])
        target_grads = [
            torch.tensor(g, device=DEVICE) if g is not None else None
            for g in grads_raw
        ]

        label = fit_res.metrics["label"]

        if server_round == NUM_ROUNDS:
            print(f"[Server] Running gradient inversion attack (round {server_round})...")
            img = gradient_inversion(model, target_grads, label)

            plt.imshow(img[0, 0], cmap="gray")
            plt.axis("off")
            plt.savefig("Flower_attack/reconstructed.png", dpi=200)
            plt.close()

            print("[Server] Saved reconstructed.png")

        return super().aggregate_fit(server_round, results, failures)
