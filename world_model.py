import torch
import torch.nn as nn

from encoder import Encoder
from rssm import RSSM, RSSMState


class Decoder(nn.Module):
    def __init__(
        self,
        feat_dim,
        obs_shape,
        hidden_dims=None,
    ):
        super().__init__()

        self.feat_dim = feat_dim
        self.obs_shape = obs_shape
        self.hidden_dims = hidden_dims

    def forward(self, feat):
        pass


class RewardModel(nn.Module):
    def __init__(
        self,
        feat_dim,
        hidden_dim,
    ):
        super().__init__()

        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim

    def forward(self, feat):
        pass


class WorldModel(nn.Module):
    def __init__(
        self,
        obs_shape,
        action_dim,
        embedding_dim,
        deter_dim,
        stoch_dim,
        model_hidden_dim,
    ):
        super().__init__()

        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.embedding_dim = embedding_dim
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.model_hidden_dim = model_hidden_dim

    def initial_state(self, batch_size, device):
        pass

    def encode(self, obs):
        pass

    def observe_rollout(self, obs, action, state=None):
        pass

    def imagine_rollout(self, action_seq, start_state):
        pass

    def decode(self, feat):
        pass

    def predict_reward(self, feat):
        pass

    def forward(self, obs, action, state=None):
        pass

    def compute_losses(self, obs, action, reward):
        pass