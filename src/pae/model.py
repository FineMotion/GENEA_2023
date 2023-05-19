import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class InstanceNorm(nn.Module):
    """
    LN_v2 from AI4Animation
    It looks like PyTorch's InstanceNorm
    """

    def __init__(self, dim: int, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon

        self.alpha = nn.Parameter(torch.ones([1, 1, dim]), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros([1, 1, dim]), requires_grad=True)

    def forward(self, x: torch.FloatTensor):
        mean = x.mean(axis=-1, keepdim=True)
        var = ((x-mean)**2).mean(dim=-1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        y = y * self.alpha + self.beta
        return y


class PhaseAutoEncoder(nn.Module):
    """
    PhaseAutoEncoder from AI4Animation:
    https://github.com/sebastianstarke/AI4Animation/blob/master/AI4Animation/SIGGRAPH_2022/PyTorch/PAE/PAE.py
    """

    def __init__(self, input_channels: int, embedding_channels: int, time_range: int, window: float,
                 channels_per_joint: int = 3):
        """

        :param input_channels: num_joints * degrees of freedom
        :param embedding_channels: phases count
        :param time_range: frames in window (window * fps) + 1
        :param window: window size in secs
        :param channels_per_joint: degrees of freedom per joint (3 or 6)
        """
        super(PhaseAutoEncoder, self).__init__()
        self.time_range = time_range
        self.embedding_channels = embedding_channels
        self.window = window
        intermediate_channels = input_channels // channels_per_joint

        # ENCODER
        self.conv1 = nn.Conv1d(input_channels, intermediate_channels, time_range, stride=1,
                               padding=(time_range - 1) // 2, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.norm1 = InstanceNorm(time_range)
        self.conv2 = nn.Conv1d(intermediate_channels, embedding_channels, time_range, stride=1,
                              padding=(time_range - 1) // 2, dilation=1, groups=1, bias=True, padding_mode='zeros')

        # FFT Params
        self.freqs = nn.Parameter(torch.fft.rfftfreq(time_range)[1:] * time_range / self.window, requires_grad=False)
        self.tpi = nn.Parameter(torch.from_numpy(np.array([2.0*np.pi], dtype=np.float32)), requires_grad=False)
        self.args = nn.Parameter(torch.from_numpy(np.linspace(-self.window / 2, self.window / 2, self.time_range,
                                                              dtype=np.float32)), requires_grad=False)

        # Phases
        self.fc = torch.nn.ModuleList()
        for i in range(embedding_channels):
            self.fc.append(nn.Linear(time_range, 2))

        # DECODER
        self.deconv1 = nn.Conv1d(embedding_channels, intermediate_channels, time_range, stride=1,
                                 padding=(time_range - 1) // 2, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.denorm1 = InstanceNorm(time_range)
        self.deconv2 = nn.Conv1d(intermediate_channels, input_channels, time_range, stride=1,
                                 padding=(time_range - 1) // 2, dilation=1, groups=1, bias=True, padding_mode='zeros')

    def encode(self, x: torch.FloatTensor):
        y = x  # batch_size, seq_len, degrees of freedom (3*joints)
        y = y.transpose(1, 2)  # bs, df, sl
        y = self.conv1(y)  # bs, joints, sl
        y = self.norm1(y)
        y = F.elu(y)

        y = self.conv2(y)  # bs, phases, sl

        latent = y

        f, a, b = self.FFT(y, dim=2)

        # v = self.phase(y) # bs, phases, 2*phases != bs, phases, 2 :( - Maybe there is more pretty solution
        # v1 = v[:,:,:self.embedding_channels] # bs, phases, phases
        # v2 = v[:,:,self.embedding_channels:]  # bs, phases, phases
        # torch.atan2(v1, v2) / self.tpi  # bs, phases, phases
        p = torch.empty((y.shape[0], self.embedding_channels), dtype=torch.float32, device=y.device)  # bs, phases
        for i in range(self.embedding_channels):
            v = self.fc[1](y[:, i, :])  # bs, 2
            p[:, i] = torch.atan2(v[:, 0], v[:, 1]) / self.tpi  # bs

        p = p.unsqueeze(2)  # bs, phases, 1
        f = f.unsqueeze(2)  # bs, phases, 1
        a = a.unsqueeze(2)  # bs, phases, 1
        b = b.unsqueeze(2)  # bs, phases, 1
        params = (p, f, a, b)

        return params, latent

    def decode(self, params):
        p, f, a, b = params

        y = a * torch.sin(self.tpi * (f * self.args + p)) + b
        signal = y # bs, phases, sl

        y = self.deconv1(y)  # bs, joints, sl
        y = self.denorm1(y)
        y = F.elu(y)
        y = self.deconv2(y) # bs, df, sl
        y = y.transpose(1, 2)  # bs, sl, df

        return y, signal

    def forward(self, x: torch.FloatTensor):
        params, latent = self.encode(x)
        y, signal = self.decode(params)

        return y, latent, signal, params

    def FFT(self, function, dim):
        rfft = torch.fft.rfft(function, dim=dim)
        magnitudes = rfft.abs()
        spectrum = magnitudes[:, :, 1:]
        power = spectrum**2

        freq = torch.sum(self.freqs * power, dim=dim) / torch.sum(power, dim=dim)
        amp = 2 * torch.sqrt(torch.sum(power, dim=dim)) / self.time_range
        offset = rfft.real[:, :, 0] / self.time_range

        return freq, amp, offset
