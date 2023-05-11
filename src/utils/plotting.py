from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from src.utils.filtering import butter


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