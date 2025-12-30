import torch

# -------------------------
# System
# -------------------------
DEVICE = torch.device("cpu")

# -------------------------
# Federated Learning
# -------------------------
NUM_ROUNDS = 5
NUM_CLIENTS = 1

# -------------------------
# Data / Model Interface
# -------------------------
IMG_SHAPE = (28, 28)
IMG_CHANNELS = 1
NUM_CLASSES = 10
HIDDEN_DIM = 12 * 7 * 7  # 588

# -------------------------
# DataLoader
# -------------------------
BATCH_SIZE = 1      # MUST be 1 for inversion
SHUFFLE = False

# -------------------------
# Gradient Inversion
# -------------------------
ITERS = 1500
LR = 0.05
TV = 1e-2
