import os
import warnings
from typing import Tuple, List

from vqvae_utils.motion_utils import *
from vqvae_utils.audio_utils import *
from vqvae_utils.text_utils import *

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