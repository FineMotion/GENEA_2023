import torch
from  torch.utils.data import Dataset
from pathlib import Path
from typing import Union, Iterator
from tqdm import tqdm
import numpy as np


class SimpleMotionDataset(Dataset):
    def __init__(self, data_files: Iterator[Union[str, Path]]):
        self.data = []

        for file_idx, data_file in tqdm(enumerate(data_files)):
            file_data = np.load(data_file)  # seq_len, features
            self.data.append(file_data)

        self.data = np.concatenate(self.data)
        assert len(self.data.shape) == 2

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        return torch.FloatTensor(self.data[item])

    @staticmethod
    def collate_fn(batch):
        return torch.stack(batch, dim=0)

