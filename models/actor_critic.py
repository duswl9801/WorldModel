import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

############################################################
# Actor
#
# Input:
#   RSSM feature = concat(deter, stoch)
#
# Output:
#   Gaussian policy parameters for continuous action
#
# Role:
#   From the current latent state, choose an action.
############################################################
class Actor(nn.Module):
    def __init__(self, feat_dim, action_dim, hidden_dim):
        super().__init__()

        self.actor_net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )

        # separate heads for mean and log-std
        # mean controls the center of the Gaussian policy
        # log_std controls the spread (uncertainty / exploration)
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

        # clamp range for log_std to keep std numerically stable
        # if std becomes too small, training can get unstable
        # if std becomes too large, actions become too noisy
        self.min_log_std = -5.0
        self.max_log_std = 2.0

    def forward(self, feat):
        h = self.actor_net(feat)

        mean = self.mean_head(h)
        log_std = self.log_std_head(h)
        log_std = torch.clamp(log_std, self.min_log_std, self.max_log_std)
        std = torch.exp(log_std)

        return mean, std

    # sample an action from the policy
    # use tanh to squash the action into [-1, 1]
    def sample_action(self, feat, deterministic=False):
        mean, std = self.forward(feat)

        if deterministic:
            # deterministic policy: choose the mean action
            action = torch.tanh(mean)
        else:
            # reparameterized sampling allows gradients to flow through the sample
            dist = Normal(mean, std)
            raw_action = dist.rsample()  # reparameterized sample

            action = torch.tanh(raw_action)

        return action

############################################################
# Critic network
#
# Input:
#   RSSM latent feature
#
# Output:
#   scalar value estimate v_t
#
# Role:
#   Predict the long-term value of the current latent state.
############################################################
class Critic(nn.Module):
    def __init__(self, feat_dim, hidden_dim):
        super().__init__()

        self.value_net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, feat):
        return self.value_net(feat) # (B, horizon, 1)

############################################################
# ActorCritic wrapper
#
# This class bundles:
#   - actor: policy network
#   - critic: value network
#
# for imagine_trajectory():
#   world_model.rssm.imagine(start_state, policy, horizon) returns
#       RSSMState sequence. shape (B, H, ...)
#
#   world_model.rssm.get_feature(state_seq) returns
#       imagined feature sequence. shape (B, H, feat_dim)
#
#   world_model.predict_reward(feat_seq) returns
#       imagined rewards. shape (B, H, 1)
############################################################
class ActorCritic(nn.Module):
    def __init__(self, feat_dim, action_dim, hidden_dim, gamma=0.99, lambda_=0.95):
        super().__init__()

        self.gamma = gamma
        self.lambda_ = lambda_

        self.actor = Actor(feat_dim, action_dim, hidden_dim)
        self.critic = Critic(feat_dim, hidden_dim)

    def policy(self, feat, deterministic=False):
        return self.actor.sample_action(feat, deterministic)

    def value(self, feat):
        return self.critic(feat)

    # Dreamer style imagination in latent space
    # start: real posterior latent state. then, latent state -> actor action -> RSSM prior step -> next latent state
    def imagine_trajectory(self, world_model, start_state, horizon):
        imagined_states = world_model.imagine_rollout(
            policy=self.actor.sample_action,
            start_state=start_state,
            horizon=horizon,
        )

        imagined_feats = world_model.rssm.get_feature(imagined_states)
        imagined_rewards = world_model.predict_reward(imagined_feats)
        imagined_values = self.critic(imagined_feats)
        imagined_cont = world_model.predict_continue(imagined_feats)

        return imagined_states, imagined_feats, imagined_rewards, imagined_values, imagined_cont

    # compute TD(lambda) targets on imagined trajectories
    # predicted rewards + predicted values + predicted discounts/continues -> lambda targets
    def compute_lambda_targets(self, rewards, values, continues, bootstrap=None):
        B, H, _ = rewards.shape
        device = rewards.device

        if bootstrap is None:
            bootstrap = torch.zeros(B, 1, device=device)

        lambda_targets = []
        next_return = bootstrap

        for t in reversed(range(H)):
            if t == H - 1: # the last step
                next_value = bootstrap
            else:
                next_value = values[:, t + 1]

            discount = self.gamma * continues[:, t] # γ*c_t. decide how much to weigh the next value

            # value: critic value
            target = rewards[:, t] + discount * ((1.0 - self.lambda_) * next_value + self.lambda_ * next_return)

            lambda_targets.append(target)
            next_return = target

        lambda_targets.reverse()
        lambda_targets = torch.stack(lambda_targets, dim=1)

        return lambda_targets # (B, H, 1)

    # critic learns to match TD(lambda) targets
    def compute_critic_loss(self, imagined_feats, lambda_targets):
        pred_values = self.critic(imagined_feats)
        critic_loss = F.mse_loss(pred_values, lambda_targets.detach())
        return critic_loss

    # actor learns to choose actions that maximize lambda-return targets
    def compute_actor_loss(self, lambda_targets, continues):
        B, H, _ = lambda_targets.shape

        # cumulative product of predicted discounts
        # w_t = prod_{i < t} gamma * c_i
        weights = []
        running = torch.ones(B, 1, device=lambda_targets.device) # first step weight: 1

        for t in range(H):
            weights.append(running)
            running = running * (self.gamma * continues[:, t])

        weights = torch.stack(weights, dim=1)  # (B, H, 1)

        # maximize predicted return value from imagined trajectory
        # maximize weighted lambda targets <=> minimize negative weighted return
        actor_loss = -(weights.detach() * lambda_targets).mean()
        return actor_loss
