from argparse import ArgumentParser


def add_trainer_args(parent_parser: ArgumentParser):
    arg_parser = ArgumentParser(parents=[parent_parser], add_help=False)
    arg_parser.add_argument("--max_epochs", type=int)
    arg_parser.add_argument("--batch_size", type=int, default=32)
    arg_parser.add_argument("--learning_rate", type=float, default=1e-4)
    arg_parser.add_argument("--accelerator", type=str, choices=["cpu", "gpu", "tpu", "ipu", "hpu", "mps", "auto"],
                            default="cpu")
    arg_parser.add_argument("--patience", type=int, default=20)
    return arg_parser
