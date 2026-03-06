import torch

from world_model import WorldModel
from actor_critic import ActorCritic


class ReplayBuffer:
    def __init__(
        self,
        capacity,
        obs_shape,
        action_dim,
        device,
    ):
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.device = device

    def add(self, obs, action, reward, next_obs, done):
        pass

    def sample_batch(self, batch_size, seq_len):
        pass

    def __len__(self):
        pass


class DreamerTrainer:
    def __init__(
        self,
        env,
        world_model,
        actor_critic,
        replay_buffer,
        config,
        device,
    ):
        self.env = env
        self.world_model = world_model
        self.actor_critic = actor_critic
        self.replay_buffer = replay_buffer
        self.config = config
        self.device = device

    def collect_random_episodes(self, num_steps):
        pass

    def collect_policy_episodes(self, num_steps):
        pass

    def preprocess_obs(self, obs):
        pass

    def train_world_model(self, batch):
        pass

    def train_actor_critic(self, batch):
        pass

    def update(self):
        pass

    def evaluate(self, num_episodes):
        pass

    def visualize_prediction(self, batch, horizon):
        pass

    def save_checkpoint(self, path):
        pass

    def load_checkpoint(self, path):
        pass

    def train(self, num_iterations):
        pass


def main():
    pass


if __name__ == "__main__":
    main()