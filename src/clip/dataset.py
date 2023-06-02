import torch
from torch.utils.data import Dataset
import numpy as np
import os
from tqdm.notebook import tqdm
from pathlib import Path
from ..vqvae.system import VQVAESystem

checkpoint_vqvae_patn = "../../checkpoints/vqvae_big_codebook/epoch=20-step=10793.ckpt"


def pad_gestures(gestures, max_frames=18):
    vectors_padded = []
    for g in gestures:
        ones = torch.ones(g.shape[0], max_frames - g.shape[1])
        last_val = g[:, -1].unsqueeze(1)
        last_val = last_val.expand_as(ones)

        vectors_padded.append(torch.cat([g, ones * last_val], dim=1))
    return torch.stack(vectors_padded)


class AudioGestureDataset(Dataset):
    def __init__(self, path: Path):
        self.audios = []
        self.gestures = []
        self.vqvae = VQVAESystem.load_from_checkpoint(checkpoint_path=checkpoint_vqvae_patn,
                                                      num_embeddings=256,
                                                      embedding_dim=256,
                                                      input_dim=54,
                                                      hidden_dim=512,
                                                      max_frames=18)
        self.vqvae.eval()

        for audio_file in tqdm(path.glob('*.npy')):
            if "audio" not in str(audio_file):
                continue

            gesture_file = str(audio_file).replace("audio", "gestures")
            for audio in np.load(audio_file, allow_pickle=True):
                self.audios.append(audio)

            for gesture in np.load(gesture_file, allow_pickle=True):
                self.gestures.append(torch.from_numpy(gesture))

        assert len(self.audios) == len(self.gestures)

        self.gestures = self.vqvae.encoder(pad_gestures(self.gestures))
        self.gestures, _, _ = self.vqvae.vq(self.gestures)

    def __len__(self):
        return len(self.audios)

    def __getitem__(self, idx):
        return torch.from_numpy(self.audios[idx]), self.gestures[idx]
