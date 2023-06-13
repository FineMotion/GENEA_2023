import numpy as np
import pandas as pd
import math
from tqdm import tqdm


class Vocab:

    def __init__(self, embedding_file: str, embedding_dim: int):
        self.weights = [
            [0.0] * embedding_dim,
            [1.0] * embedding_dim
        ]

        self.token_to_idx = {
            '<PAD>': 0,
            '<UNK>': 1,
        }

        self.idx_to_token = {
            0: '<PAD>',
            1: '<UNK>',
        }

        with open(embedding_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                arr = line.strip().split()
                word = arr[0]
                vec = list(map(float, arr[1:]))
                self.token_to_idx[word] = len(self.token_to_idx)
                self.idx_to_token[len(self.idx_to_token)] = word
                self.weights.append(vec)
        self.weights = np.array(self.weights)

    @staticmethod
    def tokenize(word: str):
        token = word
        token = token[:-1] if token[-1] in {':', '?', ';', ',', '!', '.'} else token
        token = token[:-2] if len(token) > 2 and token[-2] in {"'"} else token
        token = token.lower()
        return token

    def encode_tsv(self, tsv_path, fps: int = 30):
        transcripts = pd.read_csv(tsv_path, sep='\t', names=['start', 'end', 'word'])
        frames = math.ceil(fps * list(transcripts['end'])[-1])
        indices = np.zeros(frames, dtype=int)

        for idx, row in transcripts.iterrows():
            start_frame = math.ceil(row.start * fps)
            end_frame = math.ceil(row.end * fps)
            token = self.tokenize(str(row.word))
            index = self.token_to_idx['<UNK>']
            index = self.token_to_idx.get(token, index)
            indices[start_frame: end_frame] = np.array([index] * (end_frame - start_frame))

        return self.weights[indices]
