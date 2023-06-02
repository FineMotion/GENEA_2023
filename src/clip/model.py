from torch import nn
from esresnext.model.esresnet_fbsp import ESResNeXtFBSP
from typing import Optional

SAMPLE_RATE = 44100
FPS = 30
HOP_LENGTH = SAMPLE_RATE // FPS


class AudioEncoder(nn.Module):
    def __init__(self,
                 embed_dim: int = 1024,
                 # audio
                 n_fft: int = HOP_LENGTH * 2,
                 hop_length: Optional[int] = HOP_LENGTH,
                 win_length: Optional[int] = 2 * HOP_LENGTH,
                 window: Optional[str] = 'blackmanharris',
                 normalized: bool = True,
                 onesided: bool = True,
                 spec_height: int = -1,
                 spec_width: int = -1,
                 apply_attention: bool = True,
                 ):
        super().__init__()

        self.audio = ESResNeXtFBSP(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            normalized=normalized,
            onesided=onesided,
            spec_height=spec_height,
            spec_width=spec_width,
            num_classes=embed_dim,
            apply_attention=apply_attention,
            pretrained=False
        )

    def forward(self, audio):
        # audio shape: batch_size, sequence_length, input_size
        return self.audio(audio.to(self.device))


class GestureEncoder(nn.Module):
    def __init__(self, embedding_dim, input_size=1024, hidden_size=512):
        super().__init__()
        self.linears = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, embedding_dim),
        )

    def forward(self, gestures):
        # gestures shape: batch_size, input_size
        return self.linears(gestures)
