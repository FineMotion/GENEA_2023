import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import List
import numpy as np
from tqdm import tqdm


def stack_features(features_files: List[Path]):
    data = [np.load(str(fl)) for fl in features_files]
    sample_length = max([ft.shape[0] for ft in data])
    for i in range(len(data)):
        if data[i].shape[0] < sample_length:
            paddings = np.zeros((sample_length - data[i].shape[0], data[i].shape[1]))
            data[i] = np.concatenate([data[i], paddings], axis=0)

    return np.concatenate(data, axis=-1)


if __name__ == '__main__':

    arg_parser = ArgumentParser()
    arg_parser.add_argument("--motion", action="store", type=str, nargs='*', help="List of folders with motion data")
    arg_parser.add_argument("--audio", type=str, nargs='*', help="List of folders with audio data")
    arg_parser.add_argument("--phase", type=str, nargs='*', help="List of folders with phase data")
    arg_parser.add_argument("--dst", type=str, help="Folder to store resulted dataset")
    args = arg_parser.parse_args()

    dst_folder = Path(args.dst)
    if not dst_folder.exists():
        dst_folder.mkdir(parents=True)
    logging.basicConfig(level=logging.INFO, filename=str(dst_folder / "log.txt"))

    motion_folders = [Path(folder) for folder in args.motion] if args.motion is not None else []
    audio_folders = [Path(folder) for folder in args.audio]
    phase_folders = [Path(folder) for folder in args.phase] if args.phase is not None else []
    logging.info(f"Motion: {motion_folders}\n"
                 f"Audio: {audio_folders}\n"
                 f"Phase: {phase_folders}")

    for audio_path in tqdm(audio_folders[0].glob('*.npy')):
        audio_files = []
        for audio_folder in audio_folders:
            if audio_folder.name.find('interloctr') != -1:
                audio_file = audio_folder / audio_path.name.replace('main-agent', 'interloctr')
            else:
                audio_file = audio_folder / audio_path.name
            audio_files.append(audio_file)
        assert len(audio_files) == len(audio_folders)
        audio_data = stack_features(audio_files)

        motion_data = None
        if len(motion_folders) > 0:
            motion_files = [motion_folder / audio_path.name for motion_folder in motion_folders]
            motion_data = stack_features(motion_files)
        phase_data = None
        if len(phase_folders) > 0:
            phase_files = [phase_folder / audio_path.name for phase_folder in phase_folders]
            phase_data = stack_features(phase_files)

        dst_path = dst_folder / audio_path.name.replace('.npy', '.npz')
        logging.info(f"{dst_path}:\n"
                     f"Motion shape: {motion_data.shape if motion_data is not None else None}\n"
                     f"Audio shape: {audio_data.shape}\n"
                     f"Phase shape: {phase_data.shape if phase_data is not None else None}\n")

        np.savez(str(dst_path), Audio=audio_data, Motion=motion_data, Phase=phase_data)



