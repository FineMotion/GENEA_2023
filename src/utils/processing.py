from typing import Dict, List, Tuple

from pymo.data import MocapData
from scipy.spatial.transform import Rotation as R
import numpy as np


def extract_edges(data: MocapData) -> List[Tuple[str, str]]:
    skeleton = data.skeleton
    edges = []  # type: List[Tuple[str, str]]
    for joint in data.traverse():
        parent = skeleton[joint]['parent']
        if parent is not None:
            edges.append((joint, parent))
    return edges


def get_pose(data: MocapData, frame_num: int = 0) -> Dict[str, List[float]]:
    posed = {}  # type: Dict[str, Tuple[R, np.ndarray]]
    for joint in data.traverse():
        if 'Nub' in joint:
            # fictive joint
            rotation = np.zeros(3)
            position = np.array(
                data.skeleton[joint]['offsets']
            )
        else:
            # assume we have ZXY rotation order for all bones
            rotation = [
                data.values[f'{joint}_Zrotation'][frame_num],
                data.values[f'{joint}_Xrotation'][frame_num],
                data.values[f'{joint}_Yrotation'][frame_num]
            ]
            position = np.array([
                data.values[f'{joint}_Xposition'][frame_num],
                data.values[f'{joint}_Yposition'][frame_num],
                data.values[f'{joint}_Zposition'][frame_num]
            ])
        rotmat = R.from_euler('ZXY', rotation, degrees=True)
        posed[joint] = [None, None]
        if joint == data.root_name:
            posed[joint][0] = rotmat
            posed[joint][1] = position
        else:
            parent = data.skeleton[joint]['parent']
            posed[joint][0] = posed[parent][0] * rotmat
            rotated = posed[parent][0].apply(position)
            posed[joint][1] = posed[parent][1] + rotated

    result = {}
    for joint in posed:
        result[joint] = list(posed[joint][1])
    return result


def pose_from_numpy(frame_data: np.ndarray, joints: List[str]) -> Dict[str, List[float]]:
    data_by_axis = frame_data.reshape(frame_data.shape[0] // 3, 3)
    result = {}
    assert data_by_axis.shape[0] == len(joints)
    for i, joint in enumerate(joints):
        result[joint] = list(data_by_axis[i])
    return result
