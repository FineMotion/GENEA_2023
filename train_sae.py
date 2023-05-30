import logging
from argparse import ArgumentParser
from src.sae.system import SAESystem
from pytorch_lightning import Trainer
from pathlib import Path
import shutil
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


def add_trainer_args(parent_parser: ArgumentParser):
    arg_parser = ArgumentParser(parents=[parent_parser], add_help=False)
    arg_parser.add_argument("--max_epochs", type=int)
    arg_parser.add_argument("--batch_size", type=int, default=32)
    arg_parser.add_argument("--learning_rate", type=float, default=1e-4)
    arg_parser.add_argument("--accelerator", type=str, choices=["cpu", "gpu", "tpu", "ipu", "hpu", "mps", "auto"],
                            default="cpu")
    arg_parser.add_argument("--patience", type=int, default=20)
    return arg_parser


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--serialize_dir", type=str, required=True)
    arg_parser.add_argument("--force", action="store_true")
    arg_parser.add_argument("--trn_folder", type=str, required=True)
    arg_parser.add_argument("--val_folder", type=str, required=True)
    arg_parser.add_argument("--loss", type=str, choices=["mse", "geodesic"],
                            default="mse")
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

    system = SAESystem(
        input_dim=150,
        trn_folder=args.trn_folder,
        val_folder=args.val_folder,
        loss_name=args.loss,
        batch_size=args.batch_size
    )

    patience_callback = EarlyStopping(
        min_delta=0.0,
        mode='min',
        monitor='val/loss',
        patience=args.patience
    )

    trainer = Trainer(accelerator=args.accelerator, logger=wandb_logger,
                      callbacks=[checkpoint_callback, patience_callback])
    trainer.fit(model=system)



