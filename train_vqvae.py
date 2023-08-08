from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from argparse import ArgumentParser
from pathlib import Path

from src.vqvae.system import VQVAESystem, VQVAEDataModule
import shutil
import random
import logging

random.seed(42)
MAX_BEATS_LEN = 18


class SystemSelector:
    @staticmethod
    def add_system_args(parent_parser: ArgumentParser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--trn_folder', type=str)
        parser.add_argument('--val_folder', type=str)
        parser.add_argument('--batch_size', type=int, default=1024)
        parser.add_argument('--num_embeddings', type=int, default=2048)
        parser.add_argument('--embedding_dim', type=int, default=256)
        parser.add_argument('--input_dim', type=int, default=54)
        parser.add_argument('--hidden_dim', type=int, default=512)
        return parser

    def __init__(self, **kwargs):
        self.kwargs = kwargs

        self.system = None  # type: pl.LightningModule
        self.datamodule = None  # type: pl.LightningDataModule

    def initialize(self):
        # unpack kwargs to initialize datamodule
        trn_folder = self.kwargs['trn_folder']
        val_folder = self.kwargs['val_folder']
        batch_size = self.kwargs['batch_size']

        self.initialize_datamodule(trn_folder=trn_folder, val_folder=val_folder, batch_size=batch_size)

        # unpack kwargs to initialize system
        num_embeddings = self.kwargs['num_embeddings']
        embedding_dim = self.kwargs['embedding_dim']
        input_dim = self.kwargs['input_dim']
        hidden_dim = self.kwargs['hidden_dim']

        self.initialize_system(num_embeddings, embedding_dim, input_dim, hidden_dim)

    def initialize_system(self, num_embeddings: int, embedding_dim: int, input_dim: int,
                          hidden_dim: int, max_frames: int = MAX_BEATS_LEN):
        self.system = VQVAESystem(num_embeddings, embedding_dim, input_dim, hidden_dim, max_frames)

    def initialize_datamodule(self, trn_folder: str, val_folder: str,
                              batch_size: int, max_frames: int = MAX_BEATS_LEN, num_workers: int = 8):
        base_kwargs = {
            'trn_data_path': trn_folder,
            'val_data_path': val_folder,
            'batch_size': batch_size,
            'max_frames': max_frames,
            'num_workers': num_workers
        }

        self.datamodule = VQVAEDataModule(**base_kwargs)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--serialize_dir', type=str)
    arg_parser.add_argument("--force", action="store_true")
    arg_parser = SystemSelector.add_system_args(arg_parser)
    args = arg_parser.parse_args()
    print(Path(args.serialize_dir))
    if Path(args.serialize_dir).exists():
        if args.force:
            logging.warning(f"Force flag activated. Deleting {args.serialize_dir}...")
            shutil.rmtree(args.serialize_dir)
        else:
            logging.error(f"{args.serialize_dir} already exists! Choose another folder or use --force to overwrite")
            exit(-1)

    Path(args.serialize_dir).mkdir(parents=True)

    system_selector = SystemSelector(**vars(args))
    system_selector.initialize()

    wandb_logger = WandbLogger(name=Path(args.serialize_dir).name, project='genea2023')

    #wandb_logger.experiment.config.update(system_selector.kwargs)

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.serialize_dir,
        verbose=True,
        monitor='val/loss',
        mode='min',
        save_top_k=3,
        save_last=True
    )

    patience_callback = EarlyStopping(
        min_delta=0.0,
        mode='min',
        monitor='val/loss',
        patience=20
    )

    trainer = Trainer(accelerator="gpu", devices=-1, logger=wandb_logger,
                      callbacks=[checkpoint_callback, patience_callback])
    trainer.fit(model=system_selector.system, datamodule=system_selector.datamodule)
