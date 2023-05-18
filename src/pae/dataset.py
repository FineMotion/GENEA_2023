from pathlib import Path
from typing import Union, Iterator
import torch
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset


class AutoEncoderDataset(Dataset):
    def __init__(self, data_files: Iterator[Union[str, Path]], window: float = 2.0, fps: int = 30):
        self.storage = []
        self.data = []
        self.window = window
        self.frames = int(window*fps) + 1
        self.gather_padding = (self.frames-1)//2
        self.gather_window = np.arange(self.frames) - self.gather_padding

        for file_idx, data_file in tqdm(enumerate(data_files)):
            self.process_file(data_file, file_idx)

    def process_file(self, file_path: Union[str, Path], file_idx: int):
        file_data = np.load(file_path)  # shape: NxD
        for pivot in range(0, file_data.shape[0]):
            self.storage.append((file_idx, pivot))
        self.data.append(file_data)

    def __len__(self):
        return len(self.storage)

    def __getitem__(self, item):
        file_idx, pivot = self.storage[item]
        file_data = self.data[file_idx]

        gather = np.clip(self.gather_window+pivot, 0, file_data.shape[0] - 1)
        sample = file_data[gather]
        return torch.FloatTensor(sample)

    @staticmethod
    def collate_fn(batch):
        return torch.stack(batch, dim=0)
