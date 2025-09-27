import torch
import torch.nn as nn

class Encoder(nn.Module):
    def init(self):
        super(Encoder, self).init()
        self.encoder = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 32)
        )

    def forward(self, x):
        return self.encoder(x)
class AutoEncoder(nn.Module):
    def init(self):
        super(AutoEncoder, self).init()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded