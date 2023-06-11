import torch
import torch.nn as nn
import numpy as np


class ExpertLinear(nn.Module):
    def __init__(self, experts, input_dim, output_dim):
        super(ExpertLinear, self).__init__()
        self.experts = experts
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W = self.weights([experts, input_dim, output_dim])
        self.b = self.weights([experts, 1, output_dim])

    def forward(self, x, weights):
        y = torch.zeros((x.shape[0], self.output_dim), device=x.device, requires_grad=True)
        for i in range(self.experts):
            y = y + weights[:, i].unsqueeze(1) * (x.matmul(self.W[i, :, :]) + self.b[i, :, :])
        return y

    @staticmethod
    def weights(shape):
        alpha_bound = np.sqrt(6.0 / np.prod(shape[-2:]))
        alpha = np.asarray(np.random.uniform(
            low=-alpha_bound, high=alpha_bound, size=shape
        ), dtype=np.float32)
        return nn.Parameter(torch.from_numpy(alpha), requires_grad=True)

    @staticmethod
    def bias(shape):
        return nn.Parameter(torch.zeros(shape, dtype=torch.float), requires_grad=True)


class GatingNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super(GatingNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Dropout(dropout_rate), nn.Linear(input_dim, hidden_dim), nn.ELU(),
            nn.Dropout(dropout_rate), nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Dropout(dropout_rate), nn.Linear(hidden_dim, output_dim), nn.Softmax(dim=1)
        )

    def forward(self, x):
        # batch_size, phase_features
        w = self.model(x)
        return w  # batch_size, experts


class MotionPredictionNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, experts, dropout_rate):
        super(MotionPredictionNetwork, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.E1 = ExpertLinear(experts, input_dim, hidden_dim)
        self.E2 = ExpertLinear(experts, hidden_dim, hidden_dim)
        self.E3 = ExpertLinear(experts, hidden_dim, output_dim)
        self.elu = nn.ELU()

    def forward(self, x, w):
        x = self.dropout(x)
        x = self.E1(x, w)
        x = self.elu(x)

        x = self.dropout(x)
        x = self.E2(x, w)
        x = self.elu(x)

        x = self.dropout(x)
        x = self.E3(x, w)
        return x


class LinearWithBatchNom(nn.Module):
    def __init__(self, input_dim, output_dim, context_length: int = 61):
        super(LinearWithBatchNom, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.batch_norm = nn.BatchNorm1d(context_length)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.batch_norm(x)
        x = torch.relu(x)
        x = self.dropout(x)
        return x


class AudioEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, pivot: int = 30):
        super(AudioEncoder, self).__init__()
        self.highway = nn.Sequential(
            LinearWithBatchNom(input_dim, hidden_dim),
            LinearWithBatchNom(hidden_dim, hidden_dim),
            LinearWithBatchNom(hidden_dim, hidden_dim)
        )
        self.pivot = pivot
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True, num_layers=2)
        self.batch_norm = nn.BatchNorm1d(2*hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.out = nn.Linear(2*hidden_dim, output_dim)

    def forward(self, x):
        # batch_size, seq_len, input_dim
        x = self.highway(x)  # batch_size, seq_len, hidden
        x, _ = self.rnn(x)  # batch_size, seq_len, 2*hidden
        x = x[:, self.pivot, :]  # batch_size, 2*hidden
        x = self.batch_norm(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.out(x)
        return x