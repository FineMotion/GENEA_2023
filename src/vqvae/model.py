from torch import nn
import torch.nn.functional as F
import torch
from typing import Tuple, Any

# VQ-VAE training improvement: https://openreview.net/pdf?id=zEqdFwAPhhO
# EMA: https://arxiv.org/pdf/1711.00937.pdf

INF = 100000


class Encoder(nn.Module):
    """
    https://arxiv.org/abs/2210.01448
    """

    def __init__(
            self, input_dim: int, hidden_dim: int, embedding_dim: int, max_frames: int
    ):
        """
        Encoder class constructor.

        Args:
            input_dim (int): Dimension of the input data.
            hidden_dim (int): Dimension of the hidden layer.
            embedding_dim (int): Dimension of the embeddings.
            max_frames (int): Maximum number of frames.
        """
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.fc = nn.Linear(hidden_dim * max_frames, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Encoder.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Embeddings of the input data.
        """
        # x shape: batch_size, pose_dim: 57, frames_len
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.shape[0], -1)  # flatten

        # batch_size, embedding_dim
        return self.fc(x)


class Decoder(nn.Module):
    """
        https://arxiv.org/abs/2210.01448
    """

    def __init__(
            self, embedding_dim: int, hidden_dim: int, output_dim: int, max_frames: int
    ):
        """
        Decoder class constructor.

        Args:
            embedding_dim (int): Dimension of the embeddings.
            hidden_dim (int): Dimension of the hidden layer.
            output_dim (int): Dimension of the output data.
            max_frames (int): Maximum number of frames.
        """
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.fc = nn.Linear(embedding_dim, hidden_dim * max_frames)
        self.conv4 = nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv1 = nn.ConvTranspose1d(hidden_dim, output_dim, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Decoder.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Reconstructed data.
        """
        # batch_size, embedding_dim
        x = self.fc(x)
        x = x.view(x.shape[0], self.hidden_dim, -1)
        # batch_size, hidden_dim, max_frames
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv2(x))
        return self.conv1(x)


class VectorQuantizer(nn.Module):
    """
    https://github.com/CompVis/taming-transformers
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, beta: float = 0.25):
        """
        VectorQuantizer class constructor.

        Args:
            num_embeddings (int): Number of embeddings.
            embedding_dim (int): Dimension of the embeddings.
            beta (float): Commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2.
        """
        super(VectorQuantizer, self).__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta

        # codebook
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

        # number of uses of each vector from the codebook
        # needs to be reset every epoch!!!
        self.n_entries = torch.zeros(self.num_embeddings, dtype=torch.int64)

        self.ema_init()

    def forward(self, z: torch.Tensor, training: bool = False) -> Tuple[torch.Tensor, torch.Tensor, Any]:
        """
        Forward pass for VectorQuantizer.

        Args:
            z (torch.Tensor): Input data.
            training (bool): If True, used for training. If False, used for inference.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Any]: Quantized data, loss, info.
        """
        self.n_entries = self.n_entries.to(z.device, dtype=torch.int64)

        distances = (torch.sum(z ** 2, dim=1, keepdim=True) + torch.sum(self.embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(z, self.embedding.weight.t()))

        min_encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.num_embeddings, dtype=torch.int64).to(z.device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        encodings = F.one_hot(min_encoding_indices, self.num_embeddings).float()[:, 0, :]

        # Update n_entries
        if training:
            self.n_entries += torch.sum(min_encodings, dim=0)

        min_encodings = min_encodings.double()

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # compute loss for embedding
        loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean(
            (z_q - z.detach()) ** 2
        )

        # preserve gradients
        z_q = z + (z_q - z).detach()

        if training:
            self.ema_update(encodings, z.float())

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

    def reset_codes(self, unused_codes: torch.Tensor):
        """
        Resets the unused embedding vectors in the codebook to random values.

        Args:
            unused_codes (torch.Tensor): Indices of unused codes.
        """

        if len(unused_codes.shape) == 0:
            return
        # uniform dist [0 1)
        new_weight = torch.rand(len(unused_codes), self.embedding_dim)

        # uniform dist [-1. / num_embeddings, 1. / num_embeddings)
        new_weight = (2.0 / self.num_embeddings) * new_weight - 1.0 / self.num_embeddings

        self.embedding.weight.data[unused_codes] = new_weight.to(self.embedding.weight.data[unused_codes])

    # Reset unused codes to inf values (after training)
    def reset_unused_codes_on_end(self, unused_codes: torch.Tensor):
        """
        Resets the unused embedding vectors in the codebook to high values (inf) after training.

        Args:
            unused_codes (torch.Tensor): Indices of unused codes.
        """

        if len(unused_codes.shape) == 0:
            return

        inf_val = INF
        new_weight = torch.ones(len(unused_codes), self.embedding_dim) * inf_val
        self.embedding.weight.data[unused_codes] = new_weight.to(
            self.embedding.weight.data[unused_codes]
        )

    def get_unused_codes(self) -> torch.Tensor:
        """
        Get the unused codes in the codebook.

        Returns:
            torch.Tensor: Indices of unused codes.
        """

        return (self.n_entries == 0).nonzero().squeeze()

    # https://github.com/shuningjin/discrete-text-rep/blob/fce851a81e0b170c04f1df2901700b31702fb7b3/src/models/vq_quantizer.py#L10

    def ema_init(self, decay: float = 0.99, epsilon: float = 1e-9):
        """
        Initializes Exponential Moving Average (EMA) variables.

        Args:
            decay (float): Decay rate for EMA calculation.
            epsilon (float): Small value to prevent zero division.
        """

        self._decay = decay
        self._epsilon = epsilon
        # K
        self.register_buffer("_ema_cluster_size", torch.zeros(self.num_embeddings))
        # (K, D)
        self.register_buffer("_ema_w", torch.Tensor(self.num_embeddings, self.embedding_dim))
        self._ema_w.data = self.embedding.weight.clone()

    def ema_update(self, encodings: torch.Tensor, flat_input: torch.Tensor):
        """
        Updates the Exponential Moving Average (EMA) variables.

        Args:
            encodings (torch.Tensor): One-hot encoded input tensor.
            flat_input (torch.Tensor): Flattened input tensor.
        """

        assert len(encodings.shape) == 2
        with torch.no_grad():
            # N moving average
            self._ema_cluster_size = self._ema_cluster_size * self._decay + (1 - self._decay) * torch.sum(encodings, 0)

            # additive smoothing to avoid zero count
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = ((self._ema_cluster_size + self._epsilon) /
                                      (n + self.num_embeddings * self._epsilon) * n)

            # m moving average
            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = self._ema_w * self._decay + (1 - self._decay) * dw

            # e update
            self.embedding.weight.data.copy_((self._ema_w / self._ema_cluster_size.unsqueeze(1)).double())


class VQVAE(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, input_dim: int, hidden_dim: int, max_frames: int):
        """
        VQVAE class constructor.

        Args:
            num_embeddings (int): Number of embeddings.
            embedding_dim (int): Dimension of the embeddings.
            input_dim (int): Dimension of the input data.
            hidden_dim (int): Dimension of the hidden layer.
            max_frames (int): Maximum number of frames.
        """
        super(VQVAE, self).__init__()

        self.encoder = Encoder(input_dim, hidden_dim, embedding_dim, max_frames)
        self.vq = VectorQuantizer(num_embeddings, embedding_dim)
        self.decoder = Decoder(embedding_dim, hidden_dim, input_dim, max_frames)

    def zero_n_entries(self):
        """
            Resets the count of used codebook entries to zero.
        """
        self.vq.n_entries = torch.zeros(self.vq.num_embeddings)

    def get_unused_codes(self) -> torch.Tensor:
        """
        Retrieves the unused codes in the codebook.

        Returns:
            torch.Tensor: Indices of unused codes.
        """
        unused_codes = self.vq.get_unused_codes()
        return unused_codes

    def reset_codes(self, unused_codes: torch.Tensor):
        """
        Resets the unused embedding vectors in the codebook to random values.

        Args:
            unused_codes (torch.Tensor): Indices of unused codes.
        """
        self.vq.reset_codes(unused_codes)

    def forward(self, x: torch.Tensor, training: bool) -> Tuple[torch.Tensor, torch.Tensor, Any]:
        """
        Forward pass for VQVAE.

        Args:
            x (torch.Tensor): Input data.
            training (bool): If True, used for training. If False, used for inference.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Any]: Reconstructed data, loss, and additional info.
        """
        z = self.encoder(x)
        quantized, loss, info = self.vq(z, training=training)
        return self.decoder(quantized), loss, info
