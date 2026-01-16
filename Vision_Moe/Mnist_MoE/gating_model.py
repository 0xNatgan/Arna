import torch
import torch.nn as nn

class GatingNetwork(nn.Module):
    def __init__(self, num_experts):
        super(GatingNetwork, self).__init__()
        # Use conv layers to extract spatial features for routing
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 28 -> 14
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 14 -> 7
        )
        self.router = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, num_experts)
        )
    
    def forward(self, x):
        features = self.features(x)
        return self.router(features)