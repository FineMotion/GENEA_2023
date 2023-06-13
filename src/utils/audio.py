from pathlib import Path
import librosa
import numpy as np


def mfcc(data_path: Path, sr: int, fps:int, n_mfcc:int):
    hop = sr // fps
    wav, sr = librosa.load(str(data_path), sr=sr)
    mfccs = librosa.feature.mfcc(
        y=wav, sr=sr, hop_length=hop, n_fft=hop * 2, n_mfcc=n_mfcc
    )
    return np.transpose(mfccs)
