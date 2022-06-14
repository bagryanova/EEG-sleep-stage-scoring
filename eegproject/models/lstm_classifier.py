import torch
import torch.nn as nn
from eegproject.models.cnn_encoder import CNNEncoder

class LSTMClassifier(nn.Module):
    def __init__(self, encoder, batch_size, hidden_size=128, num_layers=1, dropout=0.0, bidirectional=False):
        super(LSTMClassifier, self).__init__()

        self.encoder = encoder

        # for p in self.encoder.parameters():
        #     p.requires_grad = False

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional)

        self.state_shape = (num_layers * (2 if bidirectional else 1), batch_size, hidden_size)

        self.mlp = nn.Sequential(
            nn.BatchNorm1d(hidden_size * (2 if bidirectional else 1)),
            # nn.Dropout(0.2),
            nn.Linear(hidden_size * (2 if bidirectional else 1), 5)
        )

    def init_state(self, device):
        return (
            torch.randn(self.state_shape, device=device),
            torch.randn(self.state_shape, device=device),
        )

    def forward(self, x, state):
        emb = self.encoder(x.view(-1, x.shape[-1]))

        res, new_state = self.lstm(emb.view(x.shape[0], x.shape[1], -1), state)

        return self.mlp(res.reshape(-1, res.shape[-1])).reshape(x.shape[0], x.shape[1], -1), (new_state[0].detach(), new_state[1].detach())
