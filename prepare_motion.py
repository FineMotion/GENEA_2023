from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import logging
from src.utils.filtering import butter
from src.utils.norm import renormalize


def velocities(src_path: Path, dst_folder: Path):
    positions = np.load(str(src_path))
    logging.info(f'{src_path.name} shape: {positions.shape}')

    velocities = np.zeros(positions.shape)
    velocities[1:] = positions[1:] - positions[:-1]

    smoothed = np.zeros(velocities.shape)
    for i in range(smoothed.shape[1]):
        smoothed[:, i] = butter(velocities[:, i])

    dst_path = dst_folder / src_path.name
    np.save(str(dst_path), smoothed)


def ortho6d(src_path: Path, dst_folder: Path, ignore_root: bool = False):
    """
    Converts input
    input: join1_a1x, joint2_a1x, ..... jointN_a1x, joint1_a1y, ... , jointN_a2z, root_Xpos, root_Ypos, root_Zpos
    output: joint1_a1x, joint1_a1y, ..., joint1_a2z, joint2_a1x, ..., jointN_a2z, root_Xvel, root_Yvel, root_Zvel
    """

    data = np.load(str(src_path))
    logging.info(f'{src_path.name} shape: {data.shape}')
    dst_path = dst_folder / src_path.name
    rotation_channels = data.shape[-1] - 3
    assert rotation_channels % 6 == 0

    num_joints = rotation_channels // 6

    joints_rotations = data[:, :-3].reshape(data.shape[0], 6, num_joints).transpose((0, 2, 1)).reshape(data.shape[0], -1)
    if ignore_root:
        logging.info(f'Ignoring root {dst_path.name} shape: {joints_rotations.shape}')
        np.save(str(dst_path), joints_rotations)
        return

    root_positions = data[:, -3:]
    root_velocities = np.zeros(root_positions.shape)
    root_velocities[1:] = root_positions[1:] - root_positions[:-1]
    smoothed = np.zeros(root_velocities.shape)
    for i in range(smoothed.shape[1]):
        smoothed[:, i] = butter(root_velocities[:, i])

    result = np.zeros(data.shape)
    result[:, :-3] = joints_rotations
    result[:, -3:] = smoothed

    np.save(str(dst_path), result)


def ortho6d_inverse(src_path: Path, dst_folder: Path, ignore_root: bool = False):
    """
    Converts input
    input: joint1_a1x, joint1_a1y, ..., joint1_a2z, joint2_a1x, ..., jointN_a2z, root_Xvel, root_Yvel, root_Zvel
    output: join1_a1x, joint2_a1x, ..... jointN_a1x, joint1_a1y, ... , jointN_a2z, root_Xpos, root_Ypos, root_Zpos
    """
    data = np.load(str(src_path))
    logging.info(f'{src_path.name} shape: {data.shape}')

    rotation_channels = data.shape[-1] - 3 if not ignore_root else data.shape[-1]
    assert rotation_channels % 6 == 0

    num_joints = rotation_channels // 6
    if ignore_root:
        channel_rotations = data.reshape(data.shape[0], num_joints, 6).transpose((0, 2, 1)).reshape(data.shape[0], -1)
    else:
        channel_rotations = data[:, :-3].reshape(data.shape[0], num_joints, 6).transpose((0, 2, 1)).reshape(data.shape[0], -1)
    # root_velocities = data[:, -3:]
    # root_positions = np.zeros(root_velocities.shape)
    #
    # if not ignore_root:
    #     root_positions[0] = root_velocities[0]
    #     for i in range(1, root_positions.shape[0]):
    #         root_positions[i] = root_positions[i-1] + root_velocities[i]
    root_positions = data[:, -3:]

    if ignore_root:
        result = np.zeros((data.shape[0], data.shape[1]+3))
    else:
        result = np.zeros(data.shape)

    result[:, :-3] = channel_rotations
    result[:, -3:] = root_positions

    dst_path = dst_folder / src_path.name
    logging.info(f'{dst_path.name} shape: {result.shape}')
    np.save(str(dst_path), result)


def predictions(src_path: Path, dst_folder: Path):
    data = np.load(str(src_path))
    logging.info(f'{src_path.name} shape: {data.shape}')

    angles = data[:, :150]
    positions = data[:, 150:150 + 26 * 3]
    velocities = data[:, 150 + 26 * 3:]

    angles = renormalize(angles, angles_norm['std'], angles_norm['mean'])
    positions = renormalize(positions, positions_norm['std'], positions_norm['mean'])
    velocities = renormalize(velocities, velocities_norm['std'], velocities_norm['mean'])

    root_velocities = velocities[:, :3]
    root_positions = np.zeros(root_velocities.shape)
    root_positions[0] = positions[0, :3]
    for i in range(1, root_positions.shape[0]):
        root_positions[i] = 0.5 * (root_positions[i - 1] + root_velocities[i]) + 0.5 * positions[i, :3]

    angles = angles.reshape(data.shape[0], 25, 6).transpose((0, 2, 1)).reshape(data.shape[0], -1)
    result = np.concatenate([angles, root_positions], axis=-1)

    dst_path = dst_folder / src_path.name
    logging.info(f'{dst_path.name} shape: {result.shape}')
    np.save(str(dst_path), result)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--src", help="Folder with positions")
    arg_parser.add_argument("--dst", help="Folder to store joint_velocities")
    arg_parser.add_argument("--mode", help="Type of preprocessing pipeline",
                            choices=["velocities", "ortho6d", "ortho6d_inverse", "predictions"],
                            default="velocities")
    arg_parser.add_argument("--ignore_root", help="Filter additional root data", action="store_true")
    arg_parser.add_argument("--angles_norm", type=str, help="Path to angles normalization values")
    arg_parser.add_argument("--positions_norm", type=str, help="Path to positions normalization values")
    arg_parser.add_argument("--velocities_norm", type=str, help="Path to velocities normalization values")
    args = arg_parser.parse_args()

    src_folder = Path(args.src)
    dst_folder = Path(args.dst)
    if not dst_folder.exists():
        dst_folder.mkdir(parents=True)

    if src_folder.is_dir():
        src_files = src_folder.glob('*.npy')
    else:
        src_files = [src_folder]

    if args.mode == "predictions":
        angles_norm = np.load(args.angles_norm)
        positions_norm = np.load(args.positions_norm)
        velocities_norm = np.load(args.velocities_norm)

    for src_file in src_files:
        if args.mode == "velocities":
            velocities(src_file, dst_folder)
        elif args.mode == "ortho6d":
            ortho6d(src_file, dst_folder, args.ignore_root)
        elif args.mode == "ortho6d_inverse":
            ortho6d_inverse(src_file, dst_folder, args.ignore_root)
        elif args.mode == "predictions":
            predictions(src_file, dst_folder)
        else:
            assert False