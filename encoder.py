import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(
        self,
        obs_shape,
        embedding_dim,
        hidden_dims=None,
    ):
        super().__init__()

        self.obs_shape = obs_shape
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims

    def forward(self, obs):
        pass