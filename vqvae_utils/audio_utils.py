import librosa
import numpy as np
from typing import List, Tuple, Union

SAMPLE_RATE: int = 44100
FPS: int = 30
HOP_LENGTH: int = SAMPLE_RATE // FPS

MIN_BEATS_LEN: int = 9
MAX_BEATS_LEN: int = 18


def load_audio_file(file_path: str) -> np.ndarray:
    """
    Loads an audio file.

    Args:
        file_path (str): Path to the audio file.

    Returns:
        np.ndarray: Loaded audio data.
    """

    audio_data, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    return audio_data


def get_borders_of_silence(audio_data: np.ndarray, silence_criterion: int = 441) -> List[List[int]]:
    """
    Identifies periods of silence in the audio data.

    Args:
        audio_data (np.ndarray): The audio data.
        silence_criterion (int): The threshold for considering a segment as silence.

    Returns:
        List[List[int]]: List of start and end indices of silence periods.
    """
    zero_indices = np.where(audio_data == 0)[0]
    if len(zero_indices) == 0:
        return []
    start = zero_indices[0]
    end = zero_indices[0]
    borders = []

    for i in range(1, len(zero_indices)):
        if zero_indices[i] == end + 1:
            end = zero_indices[i]
        else:
            if end - start + 1 >= silence_criterion:
                assert np.all(audio_data[start: end] == 0)
                borders.append([start, end])
            start = zero_indices[i]
            end = zero_indices[i]

    if end - start + 1 >= silence_criterion:
        assert np.all(audio_data[start: end] == 0)
        borders.append([start, end])

    return borders


def erase_borders_of_silence(borders: List[List[int]],
                             borders_of_silence: List[List[int]], min_len: int = 100) -> np.ndarray:
    """
    Removes silent segments from the given borders.

    Args:
        borders (List[List[int]]): The original borders.
        borders_of_silence (List[List[int]]): Borders of silent segments.
        min_len (int): The minimum length of a segment to be kept.

    Returns:
        np.ndarray: New borders with silent segments removed.
    """
    new_borders = []

    is_silence_split = False

    new_silence_border = None
    for i in range(len(borders)):
        border = borders[i]

        if new_silence_border is not None:
            borders_of_silence.append(new_silence_border)
            new_silence_border = None
        for silence_border in borders_of_silence:
            if border[1] >= silence_border[0] >= border[0]:
                is_silence_split = True

                if silence_border[0] - border[0] >= min_len:
                    new_borders.append([border[0], silence_border[0]])

                # промежуток внутри промежутка
                if border[1] >= silence_border[1]:
                    if border[1] - silence_border[1] >= min_len:
                        new_borders.append([silence_border[1], border[1]])
                # Иначе, если промежуток заходит на другой, добавляем его для обработки
                elif i != len(borders) - 1 and silence_border[1] >= borders[i + 1][0]:
                    assert silence_border[1] <= borders[i + 1][1]
                    new_silence_border = [borders[i + 1][0], silence_border[1]]

        if not is_silence_split:
            new_borders.append(border)
        is_silence_split = False

    return np.array(new_borders, dtype=int)


def extract_speech_beats(audio_data: np.ndarray, hop_length: int=HOP_LENGTH) -> np.ndarray:
    """
    Extracts beats from the speech in the audio data.

    Args:
        audio_data (np.ndarray): The audio data.
        hop_length (int): The number of audio frames between STFT columns.

    Returns:
        np.ndarray: Indices of beat events.
    """
    onset_env = librosa.onset.onset_strength(y=audio_data, hop_length=hop_length)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, hop_length=hop_length)
    return beats


def correct_beats_len(beats: np.ndarray, to_train: bool) -> np.ndarray:
    """
    https://arxiv.org/pdf/2210.01448.pdf
    Corrects the length of the beats to ensure they fall within the desired range.

    Args:
        beats (np.ndarray): The original beats.

    Returns:
        np.ndarray: The corrected beats.
    """
    new_beats = []
    if len(beats) == 0:
        return beats
    new_beats.append(beats[0])
    for i in range(1, len(beats)):
        if beats[i] - new_beats[-1] >= MIN_BEATS_LEN:
            if not to_train:
                while beats[i] - new_beats[-1] > MAX_BEATS_LEN:
                    new_beats.append(new_beats[-1] + MIN_BEATS_LEN)

            assert beats[i] - new_beats[-1] <= MAX_BEATS_LEN
            assert beats[i] - new_beats[-1] >= MIN_BEATS_LEN
            new_beats.append(beats[i])

    if new_beats[-1] != beats[-1] and not to_train:
        new_beats.append(beats[-1])

    return np.array(new_beats)


def get_beats(audio: np.ndarray, gest_len: int, to_train: bool) -> np.ndarray:
    """
    Extracts and corrects beats from the audio data.

    Args:
        audio (np.ndarray): The audio data.
        gest_len (int): The length of gestures data.
        to_train (bool): TODO

    Returns:
        np.ndarray: The beats.
    """
    beats = extract_speech_beats(audio)
    if len(beats) == 0 or beats[0] != 0 and not to_train:
        beats = [0] + list(beats)

    if beats[-1] != gest_len and not to_train:
        beats = list(beats) + [gest_len]

    beats = correct_beats_len(beats, to_train)

    if not to_train:
        assert beats[0] == 0
        assert beats[-1] == gest_len

    return beats
