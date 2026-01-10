""" LeNet model definition for image classification."""

from torch import nn

class LeNet(nn.Module):
    """ LeNet model definition for image classification."""
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
        """ Forward pass of the model."""
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
