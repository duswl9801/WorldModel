import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class RSSMState:
    deter: torch.Tensor
    stoch: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor


class RSSM(nn.Module):
    def __init__(
        self,
        action_dim,
        embed_dim,
        deter_dim,
        stoch_dim,
        hidden_dim,
    ):
        super().__init__()

        self.action_dim = action_dim
        self.embed_dim = embed_dim
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.hidden_dim = hidden_dim

    def initial_state(self, batch_size, device):
        pass

    def observe(self, embed, action, state=None):
        pass

    def imagine(self, action, state):
        pass

    def obs_step(self, prev_state, prev_action, embed):
        pass

    def img_step(self, prev_state, prev_action):
        pass

    def get_prior(self, deter):
        pass

    def get_posterior(self, deter, embed):
        pass

    def sample_state(self, mean, std, deter):
        pass

    def get_feat(self, state):
        pass