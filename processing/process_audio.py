import librosa
import numpy as np

SAMPLE_RATE = 44100
FPS = 30
HOP_LENGTH = SAMPLE_RATE // FPS


def load_audio_file(file_path):
    audio_data, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    return audio_data


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
