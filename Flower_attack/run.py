""" Main script to run Flower simulation with different client behaviors. """

import os
import flwr as fl
# pylint: disable=unused-import
from client import (
    HonestClient,
    NoisyClient,
    DetectingClient,
    ProjectedClient,
    NoisyProjectedClient
)
# pylint: enable=unused-import

from server import MaliciousFedAvg
from data import prepare_dataloader
from config import NUM_ROUNDS, NUM_CLIENTS

CLIENT_EXPERIMENTS = [
    ("honest", HonestClient),
    ("noisy", NoisyClient),
    ("projected", ProjectedClient),
    ("noisy_projected", NoisyProjectedClient),
    ("detecting", DetectingClient),
]

dataloader = prepare_dataloader(
    use_selfie=True,
    selfie_path="./base_pic/rd.jpg"
)
x, y = next(iter(dataloader))

os.makedirs("Flower_attack", exist_ok=True)
open("Flower_attack/mse_results.txt", "w").close()

for name, ClientClass in CLIENT_EXPERIMENTS:
    print(f"\n===== Running experiment: {name} =====")

    def client_fn(_):
        if ClientClass in (ProjectedClient, NoisyProjectedClient):
            return ClientClass(
                x.clone(), y.clone(), rank_ratio=0.3
            ).to_client()
        return ClientClass(x.clone(), y.clone()).to_client()

    strategy = MaliciousFedAvg(
        x_original=x.clone(),
        client_name=name,
        fraction_fit=1.0,
        min_fit_clients=1,
        min_available_clients=NUM_CLIENTS,
    )

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )