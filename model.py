import torch
import torch.nn as nn

import torch.nn.functional as F


EMBEDDING_DIM = 32  # latent dimension

class Sampling(nn.Module):
    def forward(self, z_mean, z_log_var):
        epsilon = torch.randn_like(z_mean)
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon


class Encoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dims=[128, 64], embedding_dim=32):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(True),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(True)
        )
        self.z_mean = nn.Linear(hidden_dims[1], embedding_dim)
        self.z_log_var = nn.Linear(hidden_dims[1], embedding_dim)
        self.sampling = Sampling()

    def forward(self, x):
        x = self.encoder(x)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        z = self.sampling(z_mean, z_log_var)
        return z_mean, z_log_var, z


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
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
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class Decoder(nn.Module):
    def __init__(self, embedding_dim=EMBEDDING_DIM, hidden_dims=[64, 128], output_dim=784):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dims[0]),
            nn.ReLU(True),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(True),
            nn.Linear(hidden_dims[1], output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.decoder(x)
        x = x.view(-1, 1, 28, 28)
        return x


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        z_mean, z_log_var, z = self.encoder(x)
        decoded = self.decoder(z)
        return decoded, z_mean, z_log_var
