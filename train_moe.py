import logging
from argparse import ArgumentParser
from src.mann.system import ModeAdaptiveSystem
from src.training.args import add_trainer_args
from pathlib import Path
import shutil
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import Trainer


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--serialize_dir", type=str, required=True)
    arg_parser.add_argument("--force", action="store_true")
    arg_parser.add_argument("--trn_folder", type=str, required=True)
    arg_parser.add_argument("--val_folder", type=str, required=True)
    arg_parser = ModeAdaptiveSystem.add_system_args(arg_parser)
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

    serialize_dir = Path(args.serialize_dir)
    serialize_dir.mkdir(parents=True)
    wandb_logger = WandbLogger(name=serialize_dir.name, project='genea2023_moe')
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(serialize_dir),
        verbose=True,
        monitor='val/loss',
        mode='min',
        save_top_k=3,
        save_last=True
    )

    system = ModeAdaptiveSystem(
        trn_folder=args.trn_folder,
        val_folder=args.val_folder,
        fps=args.fps,
        audio_fps=args.audio_fps,
        num_workers=args.num_workers,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        vel_included=args.vel_included
    )

    patience_callback = EarlyStopping(
        min_delta=0.0,
        mode='min',
        monitor='val/loss',
        patience=50
    )

    trainer = Trainer(accelerator=args.accelerator, devices=-1, logger=wandb_logger,
                      callbacks=[checkpoint_callback, patience_callback], max_epochs=50)
    trainer.fit(model=system)