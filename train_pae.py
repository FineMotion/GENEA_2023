import logging
from argparse import ArgumentParser
from src.pae.system import PAESystem, PAEDataModule
from pytorch_lightning import Trainer
from pathlib import Path
import shutil
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint


def add_trainer_args(parent_parser: ArgumentParser):
    arg_parser = ArgumentParser(parents=[parent_parser])
    arg_parser.add_argument("--epoches")
    arg_parser.add_argument("--batch_size")
    arg_parser.add_argument("--learning_rate", type=float, default=1e-4)
    return arg_parser


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--serialize_dir", type=str, required=True)
    arg_parser.add_argument("--force", action="store_true")
    arg_parser.add_argument("--trn_folder", type=str, required=True)
    arg_parser.add_argument("--val_folder", type=str, required=True)
    arg_parser = PAESystem.add_system_args(arg_parser)
    arg_parser = add_trainer_args(arg_parser)

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
    wandb_logger = WandbLogger(name=Path(args.serialize_dir).name, project='genea2023_pae')
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.serialize_dir,
        verbose=True,
        monitor='val/loss',
        mode='min',
        save_top_k=3,
        save_last=True
    )

    system = PAESystem(
        joints=args.joints,
        channels=args.channels,
        phases=args.phases,
        window=args.window,
        fps=args.fps,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        trn_folder=args.trn_folder,
        val_folder=args.val_folder
    )

    # data_module = PAEDataModule(
    #     trn_folder=args.trn_folder,
    #     val_folder=args.val_fodler,
    #     window=args.window,
    #     fps=args.fps,
    #     batch_size=args.batch_size
    # )

    trainer = Trainer(logger=wandb_logger, callbacks=[checkpoint_callback])
    trainer.fit(model=system)



