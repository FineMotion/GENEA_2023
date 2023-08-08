import torch
from src.mann.system import ModeAdaptiveSystem
from src.mann.dataset import ModeAdaptiveDataset
from src.utils.filtering import butter
from argparse import ArgumentParser
import logging
from pathlib import Path
import numpy as np
from tqdm import tqdm
from src.utils.norm import normalize, renormalize


def predict(system, dataset, smooth: bool = False, alpha=0.5, vel_included=False):
    phases = system.trn_dataset.Phase[0].shape[-1] if not vel_included else system.trn_dataset.Phase[0].shape[-1] // 2
    # phases = dataset.Phase[0].shape[-1]
    pose_size = system.trn_dataset.Motion[0].shape[-1]
    gather_window = system.trn_dataset.gather_window
    gather_padding = system.trn_dataset.gather_padding
    gather_size = len(gather_window)

    logging.debug(f"Phases: {phases}")
    predicted_phases = np.zeros((len(dataset)+gather_padding+1, phases))

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
        _, audio_input,  _, _ = dataset[i]
        main_input = current_pose
        phase_window = dataset.padded_sample(predicted_phases, i, gather_padding)
        gating_input = torch.FloatTensor(phase_window[gather_window + gather_padding, :].flatten())

        x, a, p = main_input.unsqueeze(0), audio_input.unsqueeze(0), gating_input.unsqueeze(0)
        with torch.no_grad():
            pred = system.forward(x,a, p)

        next_pose = pred[:, :pose_size]
        next_phase = pred[:, pose_size:]
        # logging.debug(f"Next phase shape: {next_phase.shape}")
        next_phase = next_phase.reshape((future_gather + 1, 2*phases))
        # next_phase = next_phase.reshape((future_gather + 1, phases))
        theta = renormalize(next_phase[:, :phases].numpy(), phase_std, phase_mean)
        delta = renormalize(next_phase[:, phases:].numpy(), vel_std, vel_mean)

        next_phase = alpha * theta + (1-alpha) * (
                renormalize(gating_input.reshape(gather_size, phases)[future_gather:, :].numpy(), phase_std, phase_mean) + delta)

        next_window = gather_window[future_gather:] + i + 1
        predicted_phases[next_window, :] = normalize(next_phase, phase_std, phase_mean)

        current_pose = next_pose.squeeze(0)
        predicted_poses.append(current_pose)

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
    arg_parser.add_argument('--alpha', type=float, help="Blending coefficient", default=0.5)
    arg_parser.add_argument('--phase_norm', type=str, help="Path to phase normalization values")
    arg_parser.add_argument('--vel_norm', type=str, help="Path no phase velocities normalization_values")
    arg_parser.add_argument('--trn_sample', type=str, help="Path to train sample to initialize system with data shapes")
    arg_parser = ModeAdaptiveSystem.add_system_args(arg_parser)
    args = arg_parser.parse_args()

    system = ModeAdaptiveSystem(
        trn_folder=args.src if args.trn_sample is None else args.trn_sample,
        val_folder=None,
        fps=args.fps,
        window_size=args.window_size,
        audio_fps=args.audio_fps,
        samples=args.samples,
        batch_size=1,
        vel_included=args.vel_included
    )

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    system.load_state_dict(checkpoint['state_dict'])
    system.eval()

    src_folder = Path(args.src)
    if src_folder.is_dir():
        src_files = src_folder.glob('*.npz')
    else:
        src_files = [src_folder]

    dst_folder = Path(args.dst)
    if not dst_folder.exists():
        dst_folder.mkdir(parents=True)

    phase_norm = np.load(args.phase_norm)
    phase_std, phase_mean = phase_norm['std'], phase_norm['mean']
    vel_norm = np.load(args.vel_norm)
    vel_std, vel_mean = vel_norm['std'], vel_norm['mean']
    for dataset_sample in src_files:
        dataset = ModeAdaptiveDataset(
            [dataset_sample], samples=args.samples, window_size=args.window_size, fps=args.fps,
            audio_fps=args.audio_fps, vel_included=args.vel_included
        )

        predictions = predict(system, dataset, args.smooth, args.alpha, args.vel_included)

        logging.info(f"{dataset_sample.name} output shape: {predictions.shape}")
        dst_path = dst_folder / dataset_sample.name.replace('.npz', '.npy')
        np.save(str(dst_path), predictions)
