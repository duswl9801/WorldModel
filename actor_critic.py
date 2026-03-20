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

        # Separate heads for mean and log-std.
        # mean controls the center of the Gaussian policy.
        # log_std controls the spread (uncertainty / exploration).
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

        # Clamp range for log_std to keep std numerically stable.
        # If std becomes too small, training can get unstable.
        # If std becomes too large, actions become too noisy.
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
    # use thanh to squash the action into [-1, 1]
    def sample_action(self, feat, deterministic=False):
        mean, std = self.forward(feat)

        if deterministic:
            # deterministic policy: choose the mean action.
            action = torch.tanh(mean)
        else:
            # reparameterized sampling allows gradients to flow through the sample.
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
# ContinueModel
#
# Role:
#   Predict continuation probability c_t in [0,1] (probability that episode continues.)
class ContinueModel(nn.Module):
    def __init__(self, feat_dim, hidden_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, feat):
        return torch.sigmoid(self.net(feat))

############################################################
# ActorCritic wrapper
#
# This class bundles:
#   - actor: policy network
#   - critic: value network
#
# Assumptions for imagine_trajectory():
#   world_model.rssm.imagine(start_state, policy, horizon) returns
#       an RSSMState sequence with shape (B, H, ...)
#
#   world_model.rssm.get_feature(state_seq) returns
#       imagined feature sequence with shape (B, H, feat_dim)
#
#   world_model.predict_reward(feat_seq) returns
#       imagined rewards with shape (B, H, 1)
############################################################
class ActorCritic(nn.Module):
    def __init__(self, feat_dim, action_dim, hidden_dim, gamma=0.99, lambda_=0.95):
        super().__init__()

        self.gamma = gamma
        self.lambda_ = lambda_

        self.actor = Actor(feat_dim, action_dim, hidden_dim)
        self.critic = Critic(feat_dim, hidden_dim)

        self.continue_model = ContinueModel(feat_dim, hidden_dim)

    def policy(self, feat, deterministic=False):
        return self.actor.sample_action(feat, deterministic)

    def value(self, feat):
        return self.critic(feat)

    # Dreamer-style imagination in latent space.
    # Start: real posterior latent state. Then, latent state -> actor action -> RSSM prior step -> next latent state
    def imagine_trajectory(self, world_model, start_state, horizon):
        imagined_states = world_model.imagine_rollout(
            policy=self.actor.sample_action,
            start_state=start_state,
            horizon=horizon,
        )

        imagined_feats = world_model.rssm.get_feature(imagined_states)
        imagined_rewards = world_model.predict_reward(imagined_feats)
        imagined_values = self.critic(imagined_feats)
        imagined_cont = self.continue_model(imagined_feats)

        return imagined_states, imagined_feats, imagined_rewards, imagined_values, imagined_cont

    # Compute TD(lambda) targets on imagined trajectories.
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dummy config
    B = 4
    H = 6
    deter_dim = 200
    stoch_dim = 30
    feat_dim = deter_dim + stoch_dim
    action_dim = 3
    hidden_dim = 128

    ############################################################
    # Dummy RSSM state and dummy world model for testing
    ############################################################
    class DummyState:
        def __init__(self, deter, stoch):
            self.deter = deter
            self.stoch = stoch

    class DummyRSSM:
        def __init__(self, deter_dim, stoch_dim, action_dim):
            self.deter_dim = deter_dim
            self.stoch_dim = stoch_dim
            self.action_dim = action_dim

        def get_feature(self, state):
            # concat(deter, stoch) -> (B, H, feat_dim)
            return torch.cat([state.deter, state.stoch], dim=-1)

    class DummyWorldModel(nn.Module):
        def __init__(self, deter_dim, stoch_dim, action_dim):
            super().__init__()
            self.rssm = DummyRSSM(deter_dim, stoch_dim, action_dim)

            self.reward_head = nn.Sequential(
                nn.Linear(deter_dim + stoch_dim, hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, 1),
            )

            self.action_to_deter = nn.Linear(action_dim, deter_dim)
            self.deter_to_stoch = nn.Linear(deter_dim, stoch_dim)

        def predict_reward(self, feat):
            return self.reward_head(feat)  # (B, H, 1)

        def imagine_rollout(self, policy, start_state, horizon):
            deter = start_state.deter
            stoch = start_state.stoch

            deter_seq = []
            stoch_seq = []

            for t in range(horizon):
                feat = torch.cat([deter, stoch], dim=-1)  # (B, feat_dim)

                action = policy(feat)  # (B, action_dim)

                # simple differentiable latent transition
                next_deter = torch.tanh(deter + self.action_to_deter(action))
                next_stoch = torch.tanh(self.deter_to_stoch(next_deter))

                deter_seq.append(next_deter)
                stoch_seq.append(next_stoch)

                deter = next_deter
                stoch = next_stoch

            deter_seq = torch.stack(deter_seq, dim=1)  # (B, H, deter_dim)
            stoch_seq = torch.stack(stoch_seq, dim=1)  # (B, H, stoch_dim)

            return DummyState(deter_seq, stoch_seq)

    ############################################################
    # Build actor-critic and dummy world model
    ############################################################
    actor_critic = ActorCritic(
        feat_dim=feat_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        gamma=0.99,
        lambda_=0.95,
    ).to(device)

    world_model = DummyWorldModel(
        deter_dim=deter_dim,
        stoch_dim=stoch_dim,
        action_dim=action_dim,
    ).to(device)

    ############################################################
    # Dummy start state: real posterior state at current time
    ############################################################
    start_state = DummyState(
        deter=torch.randn(B, deter_dim, device=device),
        stoch=torch.randn(B, stoch_dim, device=device),
    )

    ############################################################
    # 1. Policy test
    ############################################################
    print("===== policy test =====")
    start_feat = torch.cat([start_state.deter, start_state.stoch], dim=-1)
    sampled_action = actor_critic.policy(start_feat)
    print("sampled_action shape:", sampled_action.shape)  # (B, action_dim)

    ############################################################
    # 2. Imagine trajectory test
    ############################################################
    print("\n===== imagine trajectory test =====")
    imagined_states, imagined_feats, imagined_rewards, imagined_values, imagined_cont = \
        actor_critic.imagine_trajectory(world_model, start_state, H)

    print("imagined_feats shape:", imagined_feats.shape)  # (B, H, feat_dim)
    print("imagined_rewards shape:", imagined_rewards.shape)  # (B, H, 1)
    print("imagined_values shape:", imagined_values.shape)  # (B, H, 1)
    print("imagined_cont shape:", imagined_cont.shape)  # (B, H, 1)

    ############################################################
    # 3. Lambda target test
    ############################################################
    print("\n===== lambda target test =====")
    bootstrap = imagined_values[:, -1].detach()  # (B, 1)
    lambda_targets = actor_critic.compute_lambda_targets(
        rewards=imagined_rewards,
        values=imagined_values,
        continues=imagined_cont,
        bootstrap=bootstrap,
    )
    print("lambda_targets shape:", lambda_targets.shape)  # (B, H, 1)

    ############################################################
    # 4. Loss test
    ############################################################
    print("\n===== loss test =====")
    critic_loss = actor_critic.compute_critic_loss(imagined_feats, lambda_targets)
    actor_loss = actor_critic.compute_actor_loss(lambda_targets, imagined_cont)

    total_loss = actor_loss + critic_loss

    print(f"actor_loss:  {actor_loss.item():.6f}")
    print(f"critic_loss: {critic_loss.item():.6f}")
    print(f"total_loss:  {total_loss.item():.6f}")

    ############################################################
    # 5. Backward test
    ############################################################
    print("\n===== backward test =====")
    optimizer = torch.optim.Adam(
        list(actor_critic.parameters()) + list(world_model.parameters()),
        lr=1e-4,
    )

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    print("backward success")
    print("optimizer step success")

if __name__ == "__main__":
    main()
