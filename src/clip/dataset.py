import torch
from torch.utils.data import Dataset
import numpy as np
import os
from tqdm.notebook import tqdm


class AudioGestureDataset(Dataset):
    def __init__(self, audio_path, gestures_path):
        self.audios = []
        for f in tqdm(os.listdir(audio_path)):
            audio_file = os.path.join(audio_path, f)

            for audio in np.load(audio_file, allow_pickle=True):
                self.audios.append(audio)

        self.gestures = []
        for f in tqdm(os.listdir(gestures_path)):
            gesture_file = os.path.join(gestures_path, f)

            for gesture in np.load(gesture_file, allow_pickle=True):
                print(gesture.shape)
                self.gestures.append(gesture)

        assert len(self.audios) == len(self.gestures)

    def __len__(self):
        return len(self.audios)

    def __getitem__(self, idx):
        return torch.from_numpy(self.audios[idx]).type(torch.LongTensor),\
               torch.from_numpy(self.gestures[idx]).type(torch.LongTensor)
