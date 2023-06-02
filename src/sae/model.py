import torch
import torch.nn as nn


class SimpleAutoEncoder(nn.Module):
    """
    Simple AutoEncoder to test Geodesic loss
    """
    def __init__(self, input_dim, output_dim, hidden_size):
        super(SimpleAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim),
            nn.Sigmoid()
            )
        self.decoder = nn.Sequential(
            nn.Linear(output_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
