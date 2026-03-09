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
