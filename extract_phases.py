import torch
from argparse import ArgumentParser
from pathlib import Path
from src.pae.system import PAESystem
from src.pae.system import AutoEncoderDataset
import numpy as np
import logging


def predict_phases(system, dataset):
    items = [dataset[i] for i in range(len(dataset))]
    batch = torch.stack(items, dim=0)
    phases = []
    with torch.no_grad():
        y, latent, signal, params = system.forward(batch)
        p, f, a, b = params
        for i in range(p.shape[1]):
            phase = p.squeeze(2).detach().numpy()[:, i]
            amp = a.squeeze(2).detach().numpy()[:, i]
            phaseX = amp * np.sin(2*np.pi*phase)
            phaseY = amp * np.cos(2*np.pi*phase)
            phases.append(phaseX)
            phases.append(phaseY)

    return np.stack(phases, axis=-1)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--src', type=str, help="Folder with joint velocities")
    arg_parser.add_argument('--dst', type=str, help='Folder to store phases')
    arg_parser.add_argument('--checkpoint', type=str, help="PAE checkpoint path")
    arg_parser = PAESystem.add_system_args(arg_parser)
    args = arg_parser.parse_args()

    src_folder = Path(args.src)
    dst_folder = Path(args.dst)
    if not dst_folder.exists():
        dst_folder.mkdir()

    system = PAESystem(
        joints=args.joints,
        channels=args.channels,
        phases=args.phases,
        window=args.window,
        fps=args.fps,
        batch_size=1,
        trn_folder=None,
        val_folder=None,
        learning_rate=1e-4
    )

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    system.load_state_dict(checkpoint['state_dict'])
    system.eval()

    for vel_file in src_folder.glob('*.npy'):
        dataset = AutoEncoderDataset([str(vel_file)], window=args.window, fps=args.fps)
        phases = predict_phases(system, dataset)
        logging.info(f"{vel_file.name} output shape: {phases.shape}")
        dst_path = dst_folder / vel_file.name
        np.save(str(dst_path), phases)
