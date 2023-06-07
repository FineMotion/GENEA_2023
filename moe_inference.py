import torch
from src.mann.system import ModeAdaptiveSystem
from src.mann.dataset import ModeAdaptiveDataset
from src.utils.filtering import butter
from argparse import ArgumentParser
import logging
from pathlib import Path
import numpy as np
from tqdm import tqdm


def predict(system, dataset, smooth: bool = False, alpha=0.5):
    phases = dataset.Phase[0].shape[-1]
    pose_size = dataset.Motion[0].shape[-1]

    gather_size = len(dataset.gather_window)
    logging.debug(f"Phases: {phases}")
    predicted_phases = np.zeros((len(dataset)+dataset.gather_padding+1, phases))

    future_gather = (gather_size - 1) // 2
    current_pose = torch.zeros(pose_size)
    predicted_poses = []

    # TO INITIALIZE WITH FIRST FRAME
    # first_sample = dataset[0]
    # main_input, output, gating_input = first_sample
    # first_indices = dataset.gather_window[future_gather:]
    # predicted_phases[first_indices] = gating_input.reshape(
    #     dataset.gather_window.shape[0], phases)[future_gather:].numpy()
    # current_pose = main_input[:pose_size]

    for i in tqdm(range(len(dataset))):
        main_input, _, _ = dataset[i]
        main_input[:pose_size] = current_pose
        phase_window = dataset.padded_sample(predicted_phases, i, dataset.gather_padding)
        gating_input = torch.FloatTensor(phase_window[dataset.gather_window + dataset.gather_padding, :].flatten())

        x, p = main_input.unsqueeze(0), gating_input.unsqueeze(0)
        with torch.no_grad():
            pred = system.forward(x, p)

        next_pose = pred[:, :pose_size]
        next_phase = pred[:, pose_size:]
        # logging.debug(f"Next phase shape: {next_phase.shape}")
        next_phase = next_phase.reshape((future_gather + 1, 2*phases))
        theta = next_phase[:, :phases]
        delta = next_phase[:, phases:]
        next_phase = alpha * theta + (1-alpha) * (
                gating_input.reshape(gather_size, phases)[future_gather:, :] - delta)

        next_window = dataset.gather_window[future_gather:] + i + 1
        predicted_phases[next_window, :] = next_phase.numpy()

        current_pose = next_pose
        predicted_poses.append(current_pose.squeeze(0))

    poses = torch.stack(predicted_poses, dim=0).numpy()
    logging.debug(f"poses shape: {poses.shape}")
    if smooth:
        for i in range(poses.shape[-1]):
            poses[:, i] = butter(poses[:, i])
    return poses


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--src', type=str, help="Path to folder or file with dataset")
    arg_parser.add_argument('--dst', type=str, help="Path to store predictions")
    arg_parser.add_argument('--checkpoint', type=str, help="Path to checkpoint")
    arg_parser.add_argument('--smooth', action='store_true', help="Flag to activate motion smoothing")
    arg_parser = ModeAdaptiveSystem.add_system_args(arg_parser)
    args = arg_parser.parse_args()

    system = ModeAdaptiveSystem(
        trn_folder=args.src,
        val_folder=None,
        fps=args.fps,
        window_size=args.window_size,
        audio_fps=args.audio_fps,
        samples=args.samples,
        batch_size=1
    )

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    system.load_state_dict(checkpoint['state_dict'])
    system.eval()

    src_folder = Path(args.src)
    dst_folder = Path(args.dst)
    if not dst_folder.exists():
        dst_folder.mkdir(parents=True)

    for dataset_sample in src_folder.glob('*.npz'):
        dataset = ModeAdaptiveDataset(
            [dataset_sample], samples=args.samples, window_size=args.window_size, fps=args.fps, audio_fps=args.audio_fps
        )

        predictions = predict(system, dataset, args.smooth)

        logging.info(f"{dataset_sample.name} output shape: {predictions.shape}")
        dst_path = dst_folder / dataset_sample.name.replace('.npz', '.npy')
        np.save(str(dst_path), predictions)


