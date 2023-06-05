import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import List
import numpy as np
from tqdm import tqdm


def stack_motion(motion_files: List[Path]):
    motion_data = [np.load(str(motion_file)) for motion_file in motion_files]
    sample_length = motion_data[0].shape[0]
    for i in range(len(motion_data)):
        assert motion_data[i].shape[0] == sample_length

    return np.concatenate(motion_data, axis=-1)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--motion", action="store", type=str, nargs='*', help="List of folders with motion data")
    arg_parser.add_argument("--audio", type=str, help="Folder with audio data")
    arg_parser.add_argument("--phase", type=str, help="Folder with phase data")
    arg_parser.add_argument("--dst", type=str, help="Folder to store resulted dataset")
    args = arg_parser.parse_args()

    dst_folder = Path(args.dst)
    if not dst_folder.exists():
        dst_folder.mkdir(parents=True)

    audio_folder = Path(args.audio)
    motion_folders = [Path(folder) for folder in args.motion]
    print(motion_folders)
    phase_folder = Path(args.phase)

    for audio_path in tqdm(audio_folder.glob('*.npy')):
        phase_path = phase_folder / audio_path.name
        motion_files = [motion_folder / audio_path.name for motion_folder in motion_folders]
        motion_data = stack_motion(motion_files)
        audio_data = np.load(str(audio_path))
        phase_data = np.load(str(phase_path))

        dst_path = dst_folder / phase_path.name.replace('.npy', '.npz')
        np.savez(str(dst_path), Audio=audio_data, Motion=motion_data, Phase=phase_data)



