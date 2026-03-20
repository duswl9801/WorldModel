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
        self.representation_model = RepresentationModel(embed_dim, deter_dim, stoch_dim, hidden_dim)
        self.transition_model = TransitionModel(deter_dim, stoch_dim, hidden_dim)

    def initial_state(self, batch_size, device):
        return RSSMState(
            deter=torch.zeros(batch_size, self.deter_dim, device=device),
            stoch=torch.zeros(batch_size, self.stoch_dim, device=device),
            mean=torch.zeros(batch_size, self.stoch_dim, device=device),
            std=torch.ones(batch_size, self.stoch_dim, device=device),
        )

    # prior step. compute the prior state without using the current observation
    def prior_step(self, prev_state, prev_act):
        cur_deter = self.recurrent_model(
            prev_state.stoch,
            prev_act,
            prev_state.deter
        )
        prior_mean, prior_std = self.transition_model(cur_deter)

        # Multivariate Gaussian Distribution Sampling. sample stochastic state from the prior Gaussian
        # using the reparameterization trick (for backpropagation)
        eps = torch.randn_like(prior_std)   # sample prior_std_dimension size noise from N(0, 1)
        prior_stoch = prior_mean + prior_std * eps # reparameterization trick

        prior_state = RSSMState(
            deter=cur_deter,
            stoch=prior_stoch,
            mean=prior_mean,
            std=prior_std
        )

        return prior_state

    # posterior step. compute posterior state using current embed and deter state
    def posterior_step(self, embed, cur_deter):
        post_mean, post_std = self.representation_model(embed, cur_deter)

        # Multivariate Gaussian Distribution Sampling. sample stochastic state from the posterior Gaussian
        # using the reparameterization trick (for backpropagation)
        eps = torch.randn_like(post_std)
        post_stoch = post_mean + post_std * eps

        post_state = RSSMState(
            deter=cur_deter,
            stoch=post_stoch,
            mean=post_mean,
            std=post_std
        )

        return post_state

    # rssm step. wrapper method for prior and posterior steps
    def rssm_step(self, prev_state, prev_act, embed):
        prior_state = self.prior_step(prev_state, prev_act)
        post_state = self.posterior_step(embed, prior_state.deter)
        return post_state, prior_state

    # T: sequence length. horizon: rollout length.
    # observe a sequence of embeddings and actions to produce posterior
    # and prior state sequences for world model training.
    def observe(self, embeds, actions, state=None):

        B, T, _ = embeds.shape

        if state is None:
            state = self.initial_state(B, embeds.device)
        else:
            state = state

        prior_deters, prior_stochs, prior_means, prior_stds = [], [], [], []
        post_deters, post_stochs, post_means, post_stds = [], [], [], []

        for t in range(T):
            post_state, prior_state = self.rssm_step(
                prev_state=state,
                prev_act=actions[:, t],
                embed=embeds[:, t]
            )

            prior_deters.append(prior_state.deter)
            prior_stochs.append(prior_state.stoch)
            prior_means.append(prior_state.mean)
            prior_stds.append(prior_state.std)

            post_deters.append(post_state.deter)
            post_stochs.append(post_state.stoch)
            post_means.append(post_state.mean)
            post_stds.append(post_state.std)

            state = post_state

        # convert lists to tensors
        prior = RSSMState(
            deter=torch.stack(prior_deters, dim=1),
            stoch=torch.stack(prior_stochs, dim=1),
            mean=torch.stack(prior_means, dim=1),
            std=torch.stack(prior_stds, dim=1),
        )

        post = RSSMState(
            deter=torch.stack(post_deters, dim=1),
            stoch=torch.stack(post_stochs, dim=1),
            mean=torch.stack(post_means, dim=1),
            std=torch.stack(post_stds, dim=1),
        )

        return post, prior

    # imagine future states by rolling out the prior dynamics inside the world model.
    def imagine(self, start_state, policy, horizon):
        state = start_state
        prior_deters, prior_stochs, prior_means, prior_stds = [], [], [], [] # trajectories

        for t in range(horizon):
            feat = self.get_feature(state)
            action = policy(feat)

            prior_state = self.prior_step(state, action)

            prior_deters.append(prior_state.deter)
            prior_stochs.append(prior_state.stoch)
            prior_means.append(prior_state.mean)
            prior_stds.append(prior_state.std)

            state = prior_state

        # convert lists to tensors
        prior = RSSMState(
            deter=torch.stack(prior_deters, dim=1),
            stoch=torch.stack(prior_stochs, dim=1),
            mean=torch.stack(prior_means, dim=1),
            std=torch.stack(prior_stds, dim=1),
        )

        return prior

    # calculate KL divergence between two Gaussian(post, prior) to regularize posterior
    # posterior, prior: diagonal Gaussian -> use closed-form KL
    def kl_loss(self, post, prior, free_nats=0.0):
        post_mean, post_std = post.mean, post.std
        prior_mean, prior_std = prior.mean, prior.std

        post_var = post_std.pow(2)
        prior_var = prior_std.pow(2)

        kl = torch.log(prior_std) - torch.log(post_std)
        kl += (post_var + (post_mean - prior_mean).pow(2)) / (2.0 * prior_var)
        kl -= 0.5

        # sum over stochastic dimension
        kl = kl.sum(dim=-1)  # (B, T, S) -> (B, T)

        # apply minimum KL threshold to avoid over-penalizing small KL values
        if free_nats > 0.0:
            kl = torch.clamp(kl, min=free_nats)

        return kl.mean() # mean over batch and time (scalar loss)

    def get_feature(self, state):
        return torch.cat([state.deter, state.stoch], dim=-1)


def main():
    torch.manual_seed(0)

    # dummy setup
    B = 4
    T = 6
    horizon = 5

    action_dim = 2
    embed_dim = 4096
    deter_dim = 200
    stoch_dim = 30
    hidden_dim = 200

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build RSSM
    rssm = RSSM(
        action_dim=action_dim,
        embed_dim=embed_dim,
        deter_dim=deter_dim,
        stoch_dim=stoch_dim,
        hidden_dim=hidden_dim,
    ).to(device)

    # fake sequence inputs
    embeds = torch.randn(B, T, embed_dim, device=device)
    actions = torch.randn(B, T, action_dim, device=device)

    print("=== observe() test ===")
    post, prior = rssm.observe(embeds, actions)

    print("post.deter:", post.deter.shape)   # (B, T, deter_dim)
    print("post.stoch:", post.stoch.shape)   # (B, T, stoch_dim)
    print("post.mean :", post.mean.shape)    # (B, T, stoch_dim)
    print("post.std  :", post.std.shape)     # (B, T, stoch_dim)

    print("prior.deter:", prior.deter.shape) # (B, T, deter_dim)
    print("prior.stoch:", prior.stoch.shape) # (B, T, stoch_dim)
    print("prior.mean :", prior.mean.shape)  # (B, T, stoch_dim)
    print("prior.std  :", prior.std.shape)   # (B, T, stoch_dim)

    print("\n=== get_feature() test ===")
    post_feat = rssm.get_feature(post)
    prior_feat = rssm.get_feature(prior)

    print("post feature :", post_feat.shape)   # (B, T, deter_dim + stoch_dim)
    print("prior feature:", prior_feat.shape)  # (B, T, deter_dim + stoch_dim)

    # simple dummy policy for imagine()
    class DummyPolicy(nn.Module):
        def __init__(self, feat_dim, action_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(feat_dim, 128),
                nn.ELU(),
                nn.Linear(128, action_dim),
            )

        def forward(self, feat):
            return self.net(feat)

    policy = DummyPolicy(deter_dim + stoch_dim, action_dim).to(device)

    # use the last posterior state from observe() as imagination start state
    start_state = RSSMState(
        deter=post.deter[:, -1],
        stoch=post.stoch[:, -1],
        mean=post.mean[:, -1],
        std=post.std[:, -1],
    )

    print("\n=== imagine() test ===")
    imagined_prior = rssm.imagine(start_state, policy, horizon)

    print("imagined prior.deter:", imagined_prior.deter.shape)  # (B, horizon, deter_dim)
    print("imagined prior.stoch:", imagined_prior.stoch.shape)  # (B, horizon, stoch_dim)
    print("imagined prior.mean :", imagined_prior.mean.shape)   # (B, horizon, stoch_dim)
    print("imagined prior.std  :", imagined_prior.std.shape)    # (B, horizon, stoch_dim)

    imagined_feat = rssm.get_feature(imagined_prior)
    print("imagined feature    :", imagined_feat.shape)         # (B, horizon, deter_dim + stoch_dim)

if __name__ == "__main__":
    main()

