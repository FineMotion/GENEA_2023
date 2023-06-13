import logging
from argparse import ArgumentParser
from pathlib import Path
from src.utils.audio import mfcc
import numpy as np
from tqdm import tqdm


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--src', help='Path to audio folder')
    arg_parser.add_argument('--dst', help='Path to store results')
    arg_parser.add_argument('--sr', help="Audio sample rate", default=44100)
    arg_parser.add_argument('--fps', help="Desired features framerate", default=30)
    arg_parser.add_argument('--n_fft', help="Number of MFCC's coefficients", default=26)
    args = arg_parser.parse_args()

    src_folder = Path(args.src)
    dst_folder = Path(args.dst)

    if not dst_folder.exists():
        dst_folder.mkdir()

    for src_file in tqdm(src_folder.glob('*.wav')):
        features = mfcc(src_file, args.sr, args.fps, args.n_fft)
        dst_path = dst_folder / src_file.name.replace('.wav', '.npy')
        np.save(str(dst_path), features)

