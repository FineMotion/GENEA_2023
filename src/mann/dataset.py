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
                 fps: int = 30, audio_fps: int = 30, lazy=False):
        self.lazy = lazy
        self.fps = fps
        self.gather_padding = int(window_size * fps / 2)
        self.audio_padding = int(window_size * audio_fps / 2)
        self.audio_multiplier = audio_fps / fps
        # self.phases = phases
        self.gather_window = np.linspace(-self.gather_padding, self.gather_padding, samples)
        self.audio_window = np.linspace(-self.audio_padding, self.audio_padding, samples)
        logging.info(f"Phases gather window: {self.gather_window}")

        self.storage = []
        self.Audio = []
        self.Motion = []
        self.Phase = []

        for file_idx, data_file in tqdm(enumerate(data_files)):
            self.process_file(data_file, file_idx)

    def process_file(self, file_path: Union[str, Path], file_idx: int):
        data = np.load(file_path)
        audio, motion, phase = data['Audio'], data['Motion'], data['Phase']  # seq_len, features
        for pivot in range(motion.shape[0]):
            self.storage.append((file_idx, pivot))
        self.Audio.append(audio)
        self.Motion.append(motion)
        self.Phase.append(phase)

    def __len__(self):
        return len(self.storage)

    def __getitem__(self, item):
        file_idx, pivot = self.storage[item]
        audio, motion, phase = self.Audio[item], self.Motion[item], self.Phase[item]

        # PHASE
        phase_start = max(0, pivot - self.gather_padding)
        phase_end = min(pivot+self.gather_padding + 1, phase.shape[0])
        phase_window = np.zeros((2 * self.gather_padding + 1, phase.shape[1]))
        phase_window[phase_start: phase_end] = phase[phase_start: phase_end]
        phase_current = phase_window[self.gather_window, :]

        phase_next_start = pivot + 1
        phase_next_end = min(pivot + self.gather_padding + 2, phase.shape[0])
        phase_next_window = np.zeros((self.gather_padding + 1), phase.shape[0])
        if phase_next_start < phase.shape[0]:
            phase_next_window[phase_next_start:phase_next_end] = phase[phase_next_start:phase_next_end]
        phase_next = phase_next_window[
            self.gather_window[self.gather_padding:], :
        ]
        phase_velocity = phase_current[self.gather_padding:, :] - phase_next

        phase_x = phase_current.flatten()
        phase_y = np.concatenate([phase_next, phase_velocity], axis=-1).flatten()


        # MOTION
        current_frame = motion[pivot]
        next_frame = motion[pivot+1] if pivot + 1 < motion.shape[0] else np.zeros(motion.shape[1])

        # AUDIO CONTROL
        audio_pivot = math.ceil(self.audio_multiplier * pivot)
        audio_start = max(0, audio_pivot - self.audio_padding)
        audio_end = min(pivot + self.audio_padding + 1, audio.shape[1])
        audio_window = np.zeros((2*self.audio_padding + 1, audio.shape[1]))
        audio_window[audio_start: audio_end] = audio[audio_start: audio_end]
        audio_current = audio_window[self.audio_window, :].flatten()

        main_input = np.concatenate([current_frame, audio_current], axis=0)
        output = np.concatenate([next_frame, phase_y])
        gating_input = phase_x

        return torch.FloatTensor(main_input), torch.FloatTensor(output), torch.FloatTensor(gating_input)

    @staticmethod
    def collate_fn(batch):
        x, y, p,  = list(zip(*batch))
        return torch.stack(x, dim=0), torch.stack(y, dim=0), torch.stack(p, dim=0)