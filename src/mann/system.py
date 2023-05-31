import pytorch_lightning as pl
from .model import GatingNetwork, MotionPredictionNetwork


class ModeAdaptiveSystem(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # self.gating = GatingNetwork()
        # self.motion = MotionPredictionNetwork()