import torch.nn as nn
from config import IMG_CHANNELS, HIDDEN_DIM, NUM_CLASSES

class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.ReLU(),
        )
        self.fc = nn.Linear(64 * 28 * 28, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
