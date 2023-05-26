from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import logging
from src.utils.filtering import butter
from tqdm import tqdm


def joint_velocities(src_path: Path, dst_folder: Path):
    positions = np.load(str(src_path))
    logging.info(f'{src_path.name} shape: {positions.shape}')

    velocities = np.zeros(positions.shape)
    velocities[1:] = positions[1:] - positions[:-1]

    smoothed = np.zeros(velocities.shape)
    for i in range(smoothed.shape[1]):
        smoothed[:, i] = butter(velocities[:, i])

    dst_path = dst_folder / src_path.name
    np.save(str(dst_path), smoothed)


def ortho6d(src_path: Path, dst_folder: Path):
    """
    Converts input
    input: join1_a1x, joint2_a1x, ..... jointN_a1x, joint1_a1y, ... , jointN_a2z, root_Xpos, root_Ypos, root_Zpos
    output: joint1_a1x, joint1_a1y, ..., joint1_a2z, joint2_a1x, ..., jointN_a2z, root_Xvel, root_Yvel, root_Zvel
    """

    data = np.load(str(src_path))
    logging.info(f'{src_path.name} shape: {data.shape}')

    rotation_channels = data.shape[-1] - 3
    assert rotation_channels % 6 == 0

    num_joints = rotation_channels // 6

    joints_rotations = data[:, :-3].reshape(data.shape[0], 6, num_joints).transpose((0, 2, 1)).reshape(data.shape[0], -1)
    root_positions = data[:, -3:]
    root_velocities = np.zeros(root_positions.shape)
    root_velocities[1:] = root_positions[1:] - root_positions[:-1]
    smoothed = np.zeros(root_velocities.shape)
    for i in range(smoothed.shape[1]):
        smoothed[:, i] = butter(root_velocities[:, i])

    result = np.zeros(data.shape)
    result[:, :-3] = joints_rotations
    result[:, -3:] = smoothed

    dst_path = dst_folder / src_path.name
    np.save(str(dst_path), result)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--src", help="Folder with positions")
    arg_parser.add_argument("--dst", help="Folder to store joint_velocities")
    arg_parser.add_argument("--mode", help="Type of preprocessing pipeline", choices=["joint_velocities", "ortho6d"],
                            default="joint_velocities")
    args = arg_parser.parse_args()

    src_folder = Path(args.src)
    dst_folder = Path(args.dst)
    if not dst_folder.exists():
        dst_folder.mkdir()

    for src_file in tqdm(src_folder.glob('*.npy')):
        if args.mode == "joint_velocities":
            joint_velocities(src_file, dst_folder)
        elif args.mode == "ortho6d":
            ortho6d(src_file, dst_folder)
        else:
            assert False
