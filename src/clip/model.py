from torch import nn


class AudioEncoder(nn.Module):
    def __init__(self, embedding_dim, input_size=26, hidden_size=64, num_layers=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, embedding_dim)

    def forward(self, audio):
        # audio shape: batch_size, sequence_length, input_size

        print(audio.dtype)
        output, _ = self.lstm(audio)
        last_output = output[:, -1, :]
        embedding = self.fc(last_output)
        return embedding


class GestureEncoder(nn.Module):
    def __init__(self, embedding_dim, input_size=57, hidden_size=64, num_layers=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, embedding_dim)

    def forward(self, gestures):
        # gestures shape: batch_size, sequence_length, input_size
        output, _ = self.lstm(gestures)
        last_output = output[:, -1, :]
        embedding = self.fc(last_output)
        return embedding


class CLIPModel(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.audio_encoder = AudioEncoder(embedding_dim).double()
        self.gesture_encoder = GestureEncoder(embedding_dim).double()

    def forward(self, audio, gestures):
        audio_embeddings = self.audio_encoder(audio)
        gesture_embeddings = self.gesture_encoder(gestures)
        return audio_embeddings, gesture_embeddings
