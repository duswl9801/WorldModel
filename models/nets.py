import torch
import torch.nn as nn
import torch.nn.functional as F

# stoch_t-1 + h_t-1 + action_t-1 -> h_t
class RecurrentModel(nn.Module):
    def __init__(self, stochastic_dim, action_dim, deterministic_dim, hidden_dim):  # hidden dim here is internal layer dimension
        super().__init__()
        self.fc = nn.Linear(stochastic_dim + action_dim, hidden_dim)
        self.act = nn.ELU()
        self.gru = nn.GRUCell(hidden_dim, deterministic_dim)

    def forward(self, prev_stoch, prev_action, prev_deter):
        x = torch.cat([prev_stoch, prev_action], dim=-1)
        #print(x.shape) # -> torch.Size([4, 32})
        x = self.fc(x)
        x = self.act(x)
        h = self.gru(x, prev_deter)

        return h

# h_t + obs_t -> stoch_t. make parameters(mu, std) of stochastic *distribution* with observation and recurrent state
class RepresentationModel(nn.Module):
    def __init__(self, observation_dim, deterministic_dim, stochastic_dim, hidden_dim):
        super().__init__()
        self.fc = nn.Linear(observation_dim + deterministic_dim, hidden_dim)
        self.act = nn.ELU()

        self.fc_mean = nn.Linear(hidden_dim, stochastic_dim)
        self.fc_std = nn.Linear(hidden_dim, stochastic_dim)


    def forward(self, cur_obs, cur_deter):
        x = torch.cat([cur_obs, cur_deter], dim=-1)
        x = self.fc(x)
        x = self.act(x)

        # predict means and standard variations of stochastic distribution.
        mean = self.fc_mean(x)
        std = F.softplus(self.fc_std(x)) + 1e-4 # make std positive like dreamer

        return mean, std

# h_t -> stoch_hat_t. make parameters(mu, std) of stochastic *distribution* only with recurrent state
class TransitionModel(nn.Module):
    def __init__(self, deterministic_dim, stochastic_dim, hidden_dim):
        super().__init__()
        self.fc = nn.Linear(deterministic_dim, hidden_dim)
        self.act = nn.ELU()

        self.fc_mean = nn.Linear(hidden_dim, stochastic_dim)
        self.fc_std = nn.Linear(hidden_dim, stochastic_dim)

    def forward(self, cur_deter):
        x = self.fc(cur_deter)
        x = self.act(x)

        # predict means and standard variations of stochastic distribution.
        mean = self.fc_mean(x)
        std = F.softplus(self.fc_std(x)) + 1e-4  # make std positive like dreamer

        return mean, std


def main():
    B = 4
    s_prev = torch.randn(B, 30)
    a_prev = torch.randn(B, 2)
    h_prev = torch.randn(B, 200)

    model = RecurrentModel(30, 2, 200, 200)
    h = model(s_prev, a_prev, h_prev)
    print(h.shape)  # -> torch.Size([4, 200])

    obs = torch.rand(B, 5)
    h_cur = torch.rand(B, 200)

    model = RepresentationModel(5, 200, 30, 100)
    rep_mean, rep_std = model(obs, h_cur)
    print(rep_mean.shape, rep_std.shape)    # -> torch.Size([4, 30]) torch.Size([4, 30])

    model = TransitionModel(200, 30, 400)
    tran_mean, tran_std = model(h_cur)
    print(tran_mean.shape, tran_std.shape)  # -> torch.Size([4, 30]) torch.Size([4, 30])

if __name__ == "__main__":
    main()