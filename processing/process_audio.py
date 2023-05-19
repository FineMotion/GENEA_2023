import librosa
import numpy as np

SAMPLE_RATE = 44100
FPS = 30
HOP_LENGTH = SAMPLE_RATE // FPS


def load_audio_file(file_path):
    audio_data, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    return audio_data


def get_borders_of_silence(audio_data, silence_criterion=441):
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


def erase_borders_of_silence(borders, borders_of_silence, min_len=100):
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


def extract_speech_beats(audio_data, hop_length=HOP_LENGTH):
    onset_env = librosa.onset.onset_strength(y=audio_data, hop_length=hop_length)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, hop_length=hop_length)
    return beats


def split_audio_into_clips(audio_data, beats, hop_length=HOP_LENGTH):
    clips = []
    for i in range(len(beats) - 1):
        start = beats[i] * hop_length
        end = beats[i + 1] * hop_length
        clip = audio_data[start:end]
        mfcc = librosa.feature.mfcc(y=clip, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, n_fft=HOP_LENGTH * 2, n_mfcc=26)

        clips.append(np.transpose(mfcc))
    return clips
