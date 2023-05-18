import json
from pymo.data import MocapData
from pathlib import Path
from argparse import ArgumentParser
from pymo.parsers import BVHParser
from src.utils.processing import get_pose


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



