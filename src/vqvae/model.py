from torch import nn
import torch.nn.functional as F
import torch


# Encoder, Decoder: https://arxiv.org/abs/2210.01448
# VectorQuantizer: https://github.com/CompVis/taming-transformers

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, max_frames):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.fc = nn.Linear(hidden_dim * max_frames, embedding_dim)

    def forward(self, x):
        # x shape: batch_size, pose_dim: 57, frames_len
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        print(x.shape)
        x = x.view(x.shape[0], -1)  # flatten

        # batch_size, embedding_dim
        return self.fc(x)


class Decoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, max_frames):
        super(Decoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.fc = nn.Linear(embedding_dim, hidden_dim * max_frames)
        self.conv4 = nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv1 = nn.ConvTranspose1d(hidden_dim, output_dim, kernel_size=3, padding=1)

    def forward(self, x):
        # batch_size, embedding_dim
        x = self.fc(x)
        x = x.view(x.shape[0], self.embedding_dim, -1)
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv2(x))
        return self.conv1(x)


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(VectorQuantizer, self).__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # codebook
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1. / num_embeddings, 1. / num_embeddings)

    def forward(self, z):
        z_flattened = z.detach().view(-1, z.shape[-1])

        distances = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        min_encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0],
                                    self.num_embeddings).to(z).scatter_(1, min_encoding_indices, 1)

        encoding_indices = min_encoding_indices.view(*z.shape[:-1])
        quantized = self.embedding(encoding_indices).view(*z.shape)

        return quantized


class VQVAE(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, input_dim, hidden_dim, max_frames):
        super(VQVAE, self).__init__()

        self.encoder = Encoder(input_dim, hidden_dim, embedding_dim, max_frames)
        self.vq = VectorQuantizer(num_embeddings, embedding_dim)
        self.decoder = Decoder(embedding_dim, hidden_dim, input_dim, max_frames)

    def forward(self, x):
        z = self.encoder(x)
        quantized = self.vq(z)
        return self.decoder(quantized)
