import torch
import torch.nn as nn
from eegproject.models.cnn_encoder import CNNEncoder

class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()

        self.encoder = CNNEncoder()
        self.mlp = nn.Sequential(
            nn.Linear(128, 5),
        )

    def forward(self, x):
        f = self.encoder(x)
        return self.mlp(f)
