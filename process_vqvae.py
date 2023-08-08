from pathlib import Path
from vqvae_utils.utils import *
import logging
from argparse import ArgumentParser
from tqdm import tqdm


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
            beats = get_beats(split_audio[i], len(split_gestures[i]), to_train=True)
            bvh_blocks = split_bvh_into_blocks(split_gestures[i], beats)
            gesture_blocks += bvh_blocks

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
