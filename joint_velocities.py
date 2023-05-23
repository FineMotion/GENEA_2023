from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import logging
from src.utils.filtering import butter
from tqdm import tqdm


def process_file(src_path: Path, dst_folder: Path):
    positions = np.load(str(src_path))
    logging.info(f'{src_path.name} shape: {positions.shape}')

    velocities = np.zeros(positions.shape)
    velocities[1:] = positions[1:] - positions[:-1]

    smoothed = np.zeros(velocities.shape)
    for i in range(smoothed.shape[1]):
        smoothed[:, i] = butter(velocities[:, i])

    dst_path = dst_folder / src_path.name
    np.save(str(dst_path), smoothed)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--src", help="Folder with positions")
    arg_parser.add_argument("--dst", help="Folder to store joint_velocities")
    args = arg_parser.parse_args()

    src_folder = Path(args.src)
    dst_folder = Path(args.dst)
    if not dst_folder.exists():
        dst_folder.mkdir()

    for src_file in tqdm(src_folder.glob('*.npy')):
        process_file(src_file, dst_folder)
