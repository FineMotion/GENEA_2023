import pytorch_lightning as pl
from .model import GatingNetwork, MotionPredictionNetwork


class ModeAdaptiveSystem(pl.LightningModule):
    def __init__(self, gating_input, gating_hidden, main_input, main_hidden, main_output,
                 experts: int = 8, dropout: float = 0.3):
        super().__init__()
        self.gating = GatingNetwork(gating_input, gating_hidden, experts, dropout)
        self.motion = MotionPredictionNetwork(main_input, main_hidden, main_output, experts, dropout)

    def forward(self, x, p):
        # TODO: Find out how use dataset (concatenate input features or not)
        w = self.gating(p)
        y = self.motion(x, w)
        return y
