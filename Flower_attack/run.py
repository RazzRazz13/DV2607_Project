""" Main script to run Flower simulation with different client behaviors. """

import flwr as fl
# pylint: disable=unused-import
from client import (
    HonestClient,
    NoisyClient,
    DetectingClient,
    ProjectedClient,
)
# pylint: enable=unused-import

from server import MaliciousFedAvg
from data import prepare_dataloader
from config import NUM_ROUNDS, NUM_CLIENTS


dataloader = prepare_dataloader(use_selfie=True, selfie_path="./base_pic/rd.jpg")
x, y = next(iter(dataloader))

assert x.shape[0] == 1

def client_fn(_):
    """ Create a Flower client representing a single user. 
        Change return type for different client behaviors.
    """
    return HonestClient(x.clone(), y.clone()).to_client()

strategy = MaliciousFedAvg(
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
