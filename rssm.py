import torch
import torch.nn as nn
from dataclasses import dataclass
from nets import *

@dataclass
class RSSMState:
    deter: torch.Tensor # h_t
    stoch: torch.Tensor # z_t
    mean: torch.Tensor  # mu_t
    std: torch.Tensor   # sigma_t


class RSSM(nn.Module):
    def __init__(self, action_dim, embed_dim, deter_dim, stoch_dim, hidden_dim):
        super().__init__()

        self.action_dim = action_dim
        self.embed_dim = embed_dim
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.hidden_dim = hidden_dim

        self.recurrent_model = RecurrentModel(stoch_dim, action_dim, deter_dim, hidden_dim)
        self.represent_model = RepresentationModel(embed_dim, deter_dim, stoch_dim, hidden_dim)
        self.transition_model = TransitionModel(deter_dim, stoch_dim, hidden_dim)

    def initial_state(self, batch_size, device):
        return RSSMState(
            deter=torch.zeros(batch_size, self.deter_dim, device=device),
            stoch=torch.zeros(batch_size, self.stoch_dim, device=device),
            mean=torch.zeros(batch_size, self.stoch_dim, device=device),
            std=torch.ones(batch_size, self.stoch_dim, device=device),
        )

    # prior step.
    def prior_step(self, prev_state, prev_act):
        cur_deter = self.recurrent_model(
            prev_state.stoch,
            prev_act,
            prev_state.deter
        )
        prior_mean, prior_std = self.transition_model(cur_deter)
        ######################??????????????????????????????????????
        eps = torch.randn_like(prior_std)
        prior_stoch = prior_mean + prior_std * eps

        prior_state = RSSMState(
            deter=cur_deter,
            stoch=prior_stoch,
            mean=prior_mean,
            std=prior_std
        )

        return prior_state

    # posterior step.
    def posterior_step(self, embed, deter):
        mean, std = self.represent_model(embed, deter)
        return mean, std




