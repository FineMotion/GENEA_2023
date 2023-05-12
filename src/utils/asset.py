from pymo.data import MocapData
from transformations.transformations import identity_matrix
from scipy.spatial.transform import Rotation as R
import numpy as np


class MotionAsset:
    """
    Helper to store motion data, extract positions and reconstruct bvh back
    """
    def __init__(self):
        self.skeleton = None
        self.joints_data = {}

    def from_mocap_data(self, mocap: MocapData):
        pass

