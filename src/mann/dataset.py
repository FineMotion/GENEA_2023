import math

import numpy as np
import  torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Union, Iterator
from tqdm import tqdm
import logging


class ModeAdaptiveDataset(Dataset):
    def __init__(self, data_files: Iterator[Union[str, Path]], samples: int = 13, window_size: float = 2.0,
                 fps: int = 30, audio_fps: int = 30, vel_included=False):
        self.vel_included = vel_included
        self.fps = fps
        self.gather_padding = int(window_size * fps / 2)
        self.audio_padding = int(window_size * audio_fps / 2)
        self.audio_multiplier = audio_fps / fps
        # self.phases = phases
        self.gather_window = np.linspace(
            -self.gather_padding, self.gather_padding, samples, dtype=int)
        self.audio_window = np.linspace(
            -self.audio_padding, self.audio_padding, samples, dtype=int)
        logging.info(f"Phases gather window: {self.gather_window}")

        self.storage = []
        self.Audio = []
        self.Motion = []
        self.Phase = []

        for file_idx, data_file in tqdm(enumerate(data_files)):
            self.process_file(data_file, file_idx)

    def process_file(self, file_path: Union[str, Path], file_idx: int):
        data = np.load(file_path, allow_pickle=True)
        audio, motion, phase = data['Audio'], data['Motion'], data['Phase']  # seq_len, features
        # logging.info(motion)
        samples = audio.shape[0] if len(motion.shape) == 0 else motion.shape[0]
        for pivot in range(samples):
            self.storage.append((file_idx, pivot))
        self.Audio.append(audio)
        self.Motion.append(motion)
        self.Phase.append(phase)

    def __len__(self):
        return len(self.storage)

    @staticmethod
    def padded_sample(data, pivot, pad):
        window_size = pad*2 + 1
        start = pivot - pad
        end = pivot + pad + 1

        sample = np.zeros((window_size, data.shape[1]))
        sample_start = max(0, -start)
        sample_end = window_size - max(0, end - data.shape[0])

        start = max(0, start)
        end = min(data.shape[0], end)

        sample[sample_start:sample_end] = data[start:end]
        return sample

    @staticmethod
    def padded_next(data, pivot, pad):
        sample = np.zeros((pad+1, data.shape[1]))
        start = pivot + 1
        end = start + pad + 1
        sample_end = pad + 1 - max(0, end - data.shape[0])
        end = min(end, data.shape[0])
        sample[:sample_end] = data[start:end]
        return sample

    def __getitem__(self, item):
        file_idx, pivot = self.storage[item]
        audio, motion, phase = self.Audio[file_idx], self.Motion[file_idx], self.Phase[file_idx]

        # AUDIO CONTROL
        audio_pivot = math.ceil(self.audio_multiplier * pivot)
        audio_window = self.padded_sample(audio, audio_pivot, self.audio_padding)  # window_size, feature_dim
        # audio_current = audio_window[self.audio_window + self.audio_padding, :].flatten()
        if len(motion.shape) == 0:
            return None, torch.FloatTensor(audio_window), None, None

        # PHASE
        phase_window = self.padded_sample(phase, pivot, self.gather_padding)
        phase_current = phase_window[self.gather_window + self.gather_padding, :]

        phase_next_window = self.padded_next(phase, pivot, self.gather_padding)
        gather_padding_next = (self.gather_window.shape[0] - 1) // 2
        phase_next = phase_next_window[
            self.gather_window[gather_padding_next:], :
        ]

        if not self.vel_included:
            phase_velocity = phase_current[gather_padding_next:, :] - phase_next
            phase_y = np.concatenate([phase_next, phase_velocity], axis=-1).flatten()
            phase_x = phase_current.flatten()
        else:
            phase_y = phase_next.flatten()
            phase_start = phase_current.shape[-1] // 2
            phase_x = phase_current[:, :phase_start].flatten()

        # MOTION
        current_frame = motion[pivot]
        next_frame = motion[pivot+1] if pivot + 1 < motion.shape[0] else np.zeros(motion.shape[1])

        # main_input = np.concatenate([current_frame, audio_current], axis=0)
        output = np.concatenate([next_frame, phase_y])
        gating_input = phase_x

        return torch.FloatTensor(current_frame), torch.FloatTensor(audio_window), torch.FloatTensor(output), torch.FloatTensor(gating_input)

    @staticmethod
    def collate_fn(batch):
        x, a, y, p,  = list(zip(*batch))
        return torch.stack(x, dim=0), torch.stack(a, dim=0), torch.stack(y, dim=0), torch.stack(p, dim=0)