from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
import matplotlib.pyplot as plt
from src.utils.filtering import butter
import io
import imageio


def pos_vel_butter(data: np.array, joint_idx: int = 0, xlim: Tuple[int, int] = (0, 300)):
    """

    :param data: numpy array with shape NxD, N - sequence length, D - degrees of freedom (num joints * 3)
    :param joint_idx: index of interested joint
    :param xlim: span by x-axis to draw
    :return:
    """
    data_xyz = data.reshape(data.shape[0], data.shape[1] // 3, 3) # NxJx3
    filtered_data = data_xyz[xlim[0]:xlim[1], joint_idx] # nx3
    pos_x, pos_y, pos_z = filtered_data[:, 0], filtered_data[:, 1], filtered_data[:, 2]
    vel_x, vel_y, vel_z = pos_x[1:] - pos_x[:-1], pos_y[1:] - pos_y[:-1], pos_z[1:] - pos_z[:-1]

    fig = plt.figure(figsize=(16, 4))
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.plot(pos_x, alpha=0.3)
    ax1.plot(pos_y, alpha=0.3)
    ax1.plot(pos_z, alpha=0.3)
    ax1.set_title('positions')

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.plot(vel_x, alpha=0.3)
    ax2.plot(vel_y, alpha=0.3)
    ax2.plot(vel_z, alpha=0.3)
    ax2.set_ylim([-1, 1])
    ax2.set_title('velocities')

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.plot(butter(vel_x), alpha=0.3)
    ax3.plot(butter(vel_y), alpha=0.3)
    ax3.plot(butter(vel_z), alpha=0.3)
    ax3.set_ylim([-1, 1])
    ax3.set_title('filtered')


def skeleton_pose2d(pose: Dict[str, List[float]], edges: List[Tuple[str, str]]) -> plt.Figure:
    points = np.zeros((len(pose), 3))
    for i, joint in enumerate(pose):
        points[i, :] = np.array(pose[joint])

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlim([-100, 100])
    ax.set_ylim([0, 200])
    ax.scatter(points[:, 0], points[:, 1], s=0.5)
    for edge in edges:
        if edge[0] in pose:
            finish_node = edge[1]
            if finish_node not in pose:
                # try find parent node
                finish_nodes = [e[1] for e in edges if e[0] == finish_node]
                if len(finish_nodes) == 0 or finish_nodes[0] not in pose:
                    continue
                finish_node = finish_nodes[0]
            start = pose[edge[0]]
            finish = pose[finish_node]
            ax.plot([start[0], finish[0]], [start[1], finish[1]], color='green', alpha=0.3)
    return fig


def skeleton_gif2d(poses: List[Dict[str, List[float]]], edges: List[Tuple[str, str]], dst: Path):
    images = []
    for pose in poses:
        fig = skeleton_pose2d(pose, edges)
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        img = data.reshape((int(h), int(w), -1))
        images.append(imageio.core.util.Array(img))
        plt.close(fig)
    imageio.mimsave(str(dst), images, duration=1/30)

