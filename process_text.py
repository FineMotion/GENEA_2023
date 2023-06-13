from argparse import ArgumentParser

import numpy as np
from tqdm import tqdm
import logging
from src.utils.text import Vocab
from pathlib import Path


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--src', type=str, help="Path of data folder or tsv file")
    arg_parser.add_argument('--dst', type=str, help="Folder to store result data")
    arg_parser.add_argument('--embeddings', type=str, help="Path to embeddings weights")
    arg_parser.add_argument('--embedding_dim', type=int, help="Dimensionality of input embeddings")
    arg_parser.add_argument('--fps', type=int, help="Framerate of target embeddings", default=30)
    args = arg_parser.parse_args()

    logging.info("Loading embeddings...")
    vocab = Vocab(args.embeddings, args.embedding_dim)

    src_folder = Path(args.src)
    tsv_files = src_folder.glob('*.tsv') if src_folder.is_dir() else [src_folder]

    dst_folder = Path(args.dst)
    if not dst_folder.exists():
        dst_folder.mkdir(parents=True)

    logging.info("Converting TSV's...")
    for tsv in tqdm(tsv_files):
        dst_path = dst_folder / tsv.name.replace('.tsv', '.npy')
        try:
            embeddings = vocab.encode_tsv(tsv, args.fps)
            np.save(dst_path, embeddings)
        except Exception as e:
            logging.info(f"Failing in file: {tsv}")
            logging.error(e)
            np.save(dst_path, np.zeros((1, args.embedding_dim)))
