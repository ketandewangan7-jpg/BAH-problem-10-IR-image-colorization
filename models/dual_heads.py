import torch
import torch.nn as nn

class AnchorHead(nn.Module):
    """
    Head 1 (The Anchor Head): Forces the network to reconstruct
    the sharp, high-resolution Infrared physical signature.
    """
    def __init__(self, input_features=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_features, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1)  # Outputs 1 channel (IR)
        )

    def forward(self, x):
        return self.conv(x)

class PainterHead(nn.Module):
    """
    Head 2 (The Painter Head): Translates structural thermal features
    into a realistic 3-channel RGB visible-spectrum approximation.
    """
    def __init__(self, input_features=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_features, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1)  # Outputs 3 channels (RGB)
        )

    def forward(self, x):
        return self.conv(x)
