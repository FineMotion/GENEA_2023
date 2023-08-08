import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
from pathlib import Path


class GestureDataset(Dataset):
    """
        Custom Dataset for loading Gesture data
    """
    def __init__(self, gestures_path: Path):
        """
        GestureDataset class constructor.

        Args:
            gestures_path (Path): The path where the gesture data files are stored.
        """
        self.gestures = []

        max_frames_len = 0
        for gesture_file in tqdm(gestures_path.rglob("*.npy")):
            for gesture in np.load(gesture_file, allow_pickle=True):
                if len(gesture) > max_frames_len:
                    max_frames_len = len(gesture)
                self.gestures.append(gesture.astype(np.float64).T)

        self.max_frames_len = max_frames_len

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: The total number of samples in the dataset.
        """
        return len(self.gestures)

    def __getitem__(self, idx: int):
        """
        Returns a single sample from the dataset given its index.

        Args:
            idx (int): The index of the sample to fetch.

        Returns:
            torch.Tensor: The gesture data for the given index.
        """
        return torch.from_numpy(self.gestures[idx])
