import os
from pathlib import Path
import logging
from argparse import ArgumentParser
from tqdm import tqdm
import warnings
from typing import Tuple, List

from .motion_utils import *
from .audio_utils import *
from .text_utils import *

warnings.filterwarnings('ignore')

INTERLOCUTOR_NAME_DICT = {
    "main-agent": "interloctr",
    "interloctr": "main-agent"
}


def get_split_borders(audio_data: np.array, tsv_data: pd.DataFrame) -> np.array:
    """
    Computes the split borders for given audio by  talk times in the TSV data.

    Args:
        audio_data (np.array): The audio data.
        tsv_data (pd.DataFrame): The TSV data.

    Returns:
        np.array: The computed split borders.
    """

    borders = get_borders(tsv_data, pause_threshold=0.5) * SAMPLE_RATE
    borders_of_silence = get_borders_of_silence(audio_data)
    borders = erase_borders_of_silence(borders, borders_of_silence)
    return borders


def load_all_data(wav_name: str, speaker_motion: bool) -> Tuple[np.array, pd.DataFrame, np.array]:
    """
    Loads audio, TSV, and BVH data from files.

    Args:
        wav_name (str): The name of the WAV file.
        speaker_motion (bool): Flag indicating whether speaker motion is included.

    Returns:
        Tuple[np.array, np.array, np.array]: The loaded audio data, TSV data, and BVH data.
    """

    tsv_name = str(wav_name).replace("wav", "tsv")
    bvh_name = str(wav_name).replace("wav", "bvh")

    if not speaker_motion:
        keys = list(INTERLOCUTOR_NAME_DICT.keys())
        if keys[0] in bvh_name:
            bvh_name = bvh_name.replace(keys[0], INTERLOCUTOR_NAME_DICT[keys[0]])
        else:
            assert keys[1] in bvh_name
            bvh_name = bvh_name.replace(keys[1], INTERLOCUTOR_NAME_DICT[keys[1]])

    audio_data = load_audio_file(wav_name)
    tsv_data = load_tsv_file(tsv_name)
    pipeline = create_data_pipeline()
    bvh_data = process_data_pipeline(load_bvh_file(bvh_name), pipeline)
    return audio_data, tsv_data, bvh_data


def get_split_data(borders: np.array, audio_data: np.array, bvh_data: np.array) \
        -> Tuple[List[np.array], List[np.array]]:
    """
    Splits the audio and motion data into segments based on the given borders.

    Args:
        borders (np.array): The borders to split the data.
        audio_data (np.array): The audio data to be split.
        bvh_data (np.array): The BVH data to be split.

    Returns:
        Tuple[List[np.array], List[np.array]]: The split audio data and BVH data.
    """

    split_audio = []
    split_gestures = []

    for i in range(len(borders)):
        start = borders[i][0]
        end = borders[i][1]

        cur_audio = audio_data[start: end]

        start //= HOP_LENGTH
        end //= HOP_LENGTH
        cur_gestures = bvh_data[start: end]

        split_audio.append(cur_audio)
        split_gestures.append(cur_gestures)

    return split_audio, split_gestures


def ensure_dir_exists(dir_path: str) -> None:
    """
    Ensures a directory exists by creating it if it does not exist.

    Args:
        dir_path (str): The path of the directory to check/create.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def save_vqvae_features(src: Path, dst_dir: Path, speaker_motion: bool):
    """
    Transforms the src data into VQ-VAE features and saves to the destination directory.

    Args:
        src (Path): The source directory containing the raw data.
        dst_dir (Path): The destination directory to save the processed data.
        speaker_motion (bool): Flag indicating whether speaker motion is included.
    """

    logging.info("Transforming data...")

    speaker_dir = "speaker_motion" if speaker_motion else "interloctr_motion"
    ensure_dir_exists(os.path.join(dst_dir, speaker_dir))
    wav_names = list(src.rglob('*.wav')) if src.is_dir() else [src]

    for wav_name in tqdm(wav_names):
        audio_data, tsv_data, bvh_data = load_all_data(wav_name=wav_name, speaker_motion=speaker_motion)
        if len(audio_data) / HOP_LENGTH != bvh_data.shape[0]:
            continue

        borders = get_split_borders(audio_data=audio_data, tsv_data=tsv_data)

        if len(borders) == 0:
            continue

        split_audio, split_gestures = get_split_data(borders, audio_data, bvh_data)

        gesture_blocks = []
        for i in range(len(split_audio)):
            beats = get_beats(split_audio[i], len(split_gestures[i]))
            bvh_blocks = split_bvh_into_blocks(split_gestures[i], beats)
            gesture_blocks += bvh_blocks

        assert np.concatenate(split_gestures, axis=0).shape == np.concatenate(gesture_blocks, axis=0).shape

        dst_path = os.path.join(dst_dir, speaker_dir, str(wav_name.name).replace(".wav", ".npy"))
        np.save(dst_path, np.array(gesture_blocks, dtype=object))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    arg_parser = ArgumentParser()

    arg_parser.add_argument('--src', help='Path to input data')
    arg_parser.add_argument('--dst', help='Path to store processed data')
    arg_parser.add_argument("--speaker_motion", action="store_true")

    args = arg_parser.parse_args()

    save_vqvae_features(Path(args.src), Path(args.dst), args.speaker_motion)
