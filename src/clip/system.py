import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .dataset import AudioGestureDataset
from .model import AudioEncoder, GestureEncoder
from pathlib import Path
import torch.nn.functional as F
from torch import nn

SAMPLE_RATE = 44100
FPS = 30
HOP_LENGTH = SAMPLE_RATE // FPS


# https://wandb.ai/manan-goel/coco-clip/reports/Implementing-CLIP-With-PyTorch-Lightning--VmlldzoyMzg4Njk1


class CLIPSystem(pl.LightningModule):
    def __init__(self, embedding_dim,
                 temperature: float = 1.0):
        super(CLIPSystem, self).__init__()
        self.audio_encoder = AudioEncoder(embedding_dim)
        self.gesture_encoder = GestureEncoder(embedding_dim)

        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.temperature = temperature

    def _compute_losses(self, audio_embeddings, gesture_embeddings):
        logits = (gesture_embeddings @ audio_embeddings.T) / self.temperature
        images_similarity = audio_embeddings @ audio_embeddings.T
        texts_similarity = gesture_embeddings @ gesture_embeddings.T
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        images_loss = (-targets.T * self.log_softmax(logits.T)).sum(1)
        texts_loss = (-targets * self.log_softmax(logits)).sum(1)
        return (images_loss + texts_loss) / 2.0

    def forward(self, audios, gestures):
        audios_features = self.audio_encoder(audios)
        gestures_features = self.gesture_encoder(gestures)

        audio_embeddings = self.image_projection(audios_features)
        gesture_embeddings = self.text_projection(gestures_features)

        return audio_embeddings, gesture_embeddings

    def training_step(self, batch, batch_idx):
        audio, gestures = batch
        audio_embeddings, gesture_embeddings = self.forward(audio, gestures)

        loss = self._compute_losses(audio_embeddings, gesture_embeddings).mean()
        train_loss = self.all_gather(loss)

        self.log('train/loss', train_loss.mean())
        return {"loss": train_loss}

    def validation_step(self, batch, batch_idx):
        audio, gestures = batch
        audio_embeddings, gesture_embeddings = self.forward(audio, gestures)

        loss = self._compute_losses(audio_embeddings, gesture_embeddings).mean()

        val_loss = self.all_gather(loss)
        self.log("val/loss", val_loss.mean())
        return {"loss": loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


class CLIPDataModule(pl.LightningDataModule):
    def __init__(self, trn_data_path='clip_data_no_root/trn/ma_audio_i_bvh',
                 val_data_path='clip_data_no_root/val/ma_audio_i_bvh',
                 batch_size: int = 512, max_frames: int = 18):
        super().__init__()
        self.val_dataset = None
        self.trn_dataset = None

        self.train_path = Path(trn_data_path)
        self.val_path = Path(val_data_path)

        self.batch_size = batch_size
        self.max_frames = max_frames

    def setup(self, stage=None) -> None:
        self.trn_dataset = AudioGestureDataset(self.train_path)
        self.val_dataset = AudioGestureDataset(self.val_path)

        print("Train size:")
        print(len(self.trn_dataset))
        print("Val size:")
        print(len(self.val_dataset))

    def collate_fn(self, batch):
        gesture_vectors = []
        audio_vectors_padded = []
        for b in batch:
            audio, gesture = b
            zeros = torch.zeros(audio.shape[0], self.max_frames - audio.shape[1])

            gesture_vectors.append(gesture)
            audio_vectors_padded.append(torch.cat([audio, zeros], dim=1))

        return torch.stack(audio_vectors_padded), torch.stack(gesture_vectors)

    def train_dataloader(self):
        return DataLoader(self.trn_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn,
                          num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn,
                          num_workers=8)
