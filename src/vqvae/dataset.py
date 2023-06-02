import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm.notebook import tqdm
from pathlib import Path


class GestureDataset(Dataset):
    def __init__(self, gestures_path: Path):
        self.gestures = []

        max_frames_len = 0
        for gesture_file in tqdm(gestures_path.rglob('*.npy')):
            if "gestures" not in str(gesture_file):
                continue
            for gesture in np.load(gesture_file, allow_pickle=True):
                if len(gesture) > max_frames_len:
                    max_frames_len = len(gesture)
                self.gestures.append(gesture.astype(np.float64).T)

        self.max_frames_len = max_frames_len

    def __len__(self):
        return len(self.gestures)

    def __getitem__(self, idx):
        return torch.from_numpy(self.gestures[idx])
