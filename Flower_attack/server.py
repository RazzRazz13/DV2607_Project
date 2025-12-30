import json
import torch
import flwr as fl
from model import LeNet
import matplotlib.pyplot as plt
from collections import OrderedDict
from config import NUM_ROUNDS, DEVICE
from attack import gradient_inversion

class MaliciousFedAvg(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        if not results:
            print("[Server] No client results received")
            return super().aggregate_fit(rnd, results, failures)

        _, fit_res = results[0]

        # Single-client assumption
        assert fit_res.num_examples == 1, \
            "Gradient inversion requires batch_size=1"

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

        if rnd == NUM_ROUNDS:
            print(f"[Server] Running gradient inversion attack (round {rnd})...")
            img = gradient_inversion(model, target_grads, label)

            plt.imshow(img[0, 0], cmap="gray")
            plt.axis("off")
            plt.savefig("Flower_attack/reconstructed.png", dpi=200)
            plt.close()

            print("[Server] Saved reconstructed.png")

        return super().aggregate_fit(rnd, results, failures)


