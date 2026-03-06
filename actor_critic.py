import torch
import torch.nn as nn


class Actor(nn.Module):
    def __init__(
        self,
        feat_dim,
        action_dim,
        hidden_dim,
    ):
        super().__init__()

        self.feat_dim = feat_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

    def forward(self, feat):
        pass

    def sample_action(self, feat, deterministic=False):
        pass

    def log_prob(self, feat, action):
        pass


class Critic(nn.Module):
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


class ActorCritic(nn.Module):
    def __init__(
        self,
        feat_dim,
        action_dim,
        hidden_dim,
    ):
        super().__init__()

        self.feat_dim = feat_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

    def policy(self, feat, deterministic=False):
        pass

    def value(self, feat):
        pass

    def imagine_trajectory(self, world_model, start_state, horizon):
        pass

    def compute_actor_loss(self, imagined_feats, imagined_rewards, imagined_values):
        pass

    def compute_critic_loss(self, imagined_feats, value_targets):
        pass