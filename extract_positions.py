from typing import Any, Tuple, Dict, List
import json
from pymo.data import MocapData
from transformations.transformations import identity_matrix
from scipy.spatial.transform import Rotation as R
import numpy as np
from pathlib import Path
from argparse import ArgumentParser
from pymo.parsers import BVHParser


def get_pose(data: MocapData, frame_num: int = 0) -> Dict[str, List[float]]:
    posed = {}  # type: Dict[str, Tuple[R, np.ndarray]]
    for joint in data.traverse():
        if 'Nub' in joint:
            # fictive joint
            rotation = np.zeros(3)
            position = np.array(
                data.skeleton[joint]['offsets']
            )
        else:
            # assume we have ZXY rotation order for all bones
            rotation = [
                data.values[f'{joint}_Zrotation'][frame_num],
                data.values[f'{joint}_Xrotation'][frame_num],
                data.values[f'{joint}_Yrotation'][frame_num]
            ]
            position = np.array([
                data.values[f'{joint}_Xposition'][frame_num],
                data.values[f'{joint}_Yposition'][frame_num],
                data.values[f'{joint}_Zposition'][frame_num]
            ])
        rotmat = R.from_euler('ZXY', rotation, degrees=True)
        posed[joint] = [None, None]
        if joint == data.root_name:
            posed[joint][0] = rotmat
            posed[joint][1] = position
        else:
            parent = data.skeleton[joint]['parent']
            posed[joint][0] = posed[parent][0] * rotmat
            rotated = posed[parent][0].apply(position)
            posed[joint][1] = posed[parent][1] + rotated

    result = {}
    for joint in posed:
        result[joint] = list(posed[joint][1])
    return result


def extract_frames(data: MocapData, dst_folder: Path):
    for i in range(len(data.values)):
        frame_data = get_pose(data, i)
        frame_path = dst_folder / f'frame_{i:04d}.json'
        with open(str(frame_path), 'w') as outfile:
            json.dump(frame_data, outfile, indent=4)


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--src', help='Path to sample BVH-file')
    arg_parser.add_argument('--dst', help='Folder to save extracted positions')
    args = arg_parser.parse_args()

    bvh_parser = BVHParser()
    data = bvh_parser.parse(args.src)
    dst_folder = Path(args.dst)
    if not dst_folder.exists():
        dst_folder.mkdir()

    extract_frames(data, dst_folder)



