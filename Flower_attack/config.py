""" Configuration settings for Flower attack experiments."""

import torch

DEVICE = torch.device("cpu")

NUM_ROUNDS = 5
NUM_CLIENTS = 2

IMG_SHAPE = (28, 28)
IMG_CHANNELS = 1
NUM_CLASSES = 10
HIDDEN_DIM = 12 * 7 * 7  # 588

BATCH_SIZE = 1
SHUFFLE = False

ITERS = 1500
LR = 0.05
TV = 1e-2
