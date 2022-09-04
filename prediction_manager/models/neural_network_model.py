from torch import nn

class NeuralNetworkModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 12),
            nn.ReLU(),
            nn.Linear(12, 3),
        )

    def forward(self, x):
        return self.linear_relu_stack(x)
