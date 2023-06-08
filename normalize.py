from pathlib import Path
import logging
from typing import Iterable

import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser


def get_normalization_values(data: np.ndarray):
    std = np.std(data, axis=0)
    mean = np.mean(data, axis=0)
    return std, mean


def normalize(data, std, mean):
    return (data - mean[np.newaxis, :]) / std[np.newaxis, :]


def renormalize(data, std, mean):
    return data * std[np.newaxis, :] + mean[np.newaxis, :]


def stack_data(data_files: Iterable[Path]):
    result = []
    for data_file in data_files:
        data = np.load(data_file)
        result.append(data)
    return np.concatenate(result, axis=0)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--src', type=str, help="Path to data folder")
    arg_parser.add_argument('--dst', type=str, help="Folder to store results")
    arg_parser.add_argument('--norm', type=str, help="Path to normalization values")
    arg_parser.add_argument('--mode', choices=['forward', 'backward'], default='forward',
                            help="Normalize (forward) or denormalize (backward) data")
    args = arg_parser.parse_args()

    src_path = Path(args.src)
    src_files = list(src_path.glob('*.npy')) if src_path.is_dir() else [src_path]
    normalization_values = Path(args.norm)

    if normalization_values.exists():
        logging.info("Loading normalization values")
        normalization_data = np.load(normalization_values)
        std, mean = normalization_data['std'], normalization_data['mean']
    else:
        logging.info('Creating normalization values')
        all_data = stack_data(src_files)
        std, mean = get_normalization_values(all_data)
        np.savez(normalization_values, std=std, mean=mean)

    dst_folder = Path(args.dst)
    if not dst_folder.exists():
        dst_folder.mkdir()

    for src_file in tqdm(src_files):
        dst_file = dst_folder / src_file.name
        src_data = np.load(src_file)
        dst_data = normalize(src_data, std, mean) if args.mode == 'forward' else renormalize(src_data, std, mean)
        np.save(dst_file, dst_data)
