import gymnasium as gym
import highway_env

import matplotlib.pyplot as plt
import numpy as np
import json
import cv2
import pickle

output_path = "./outputs"

def preprocess_frame(frame, size=(64, 64)):
    frame = cv2.resize(frame, size) # downscale the image for model input
    frame = frame.astype(np.float32) / 255.0 # normalize
    frame = np.transpose(frame, (2, 0, 1))  # HWC -> CHW
    return frame


# collect random transitions for warm start
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    # when the buffer is full. start overwrite from the beginning.
    def add(self, frame, action, reward, next_frame, done):
        transition = {
            "frame": frame,
            "action": action,
            "reward": reward,
            "next_frame": next_frame,
            "done": done,
        }

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition

        self.position = (self.position + 1) % self.capacity

    # randomly select batches from the buffer
    def sample_batch(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        return batch

    def __len__(self):
        return len(self.buffer)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.buffer, f)

    def load(self, path):
        with open(path, "rb") as f:
            self.buffer = pickle.load(f)

def collect_random_steps(env, replay_buffer, num_steps):
    obs, info = env.reset()

    frame = env.render()
    frame = preprocess_frame(frame)

    episode_reward = 0.0
    episode_steps = 0
    episode_count = 0

    for step in range(num_steps):
        action = env.action_space.sample()

        next_obs, reward, done, truncated, info = env.step(action)

        next_frame = env.render()
        next_frame = preprocess_frame(next_frame)

        terminal = done or truncated

        replay_buffer.add(
            frame=frame,
            action=action,
            reward=reward,
            next_frame=next_frame,
            done=terminal
        )

        frame = next_frame
        episode_reward += reward
        episode_steps += 1

        if terminal:
            print(
                f"Episode {episode_count} finished | "
                f"steps={episode_steps}, reward={episode_reward:.4f}"
            )

            obs, info = env.reset()
            frame = env.render()
            frame = preprocess_frame(frame)

            episode_reward = 0.0
            episode_steps = 0
            episode_count += 1

def main():
    with open("config.json", "r") as f:
        config = json.load(f)
    env = gym.make("highway-v0", render_mode="rgb_array", config=config)
    replay_buffer = ReplayBuffer(capacity=10000)

    steps = 5000

    collect_random_steps(env=env, replay_buffer=replay_buffer, num_steps=steps)
    replay_buffer.save(output_path + f"replay_buffer_{steps}.pkl")
    print("Replay buffer saved.")

    print("Collected transitions:", len(replay_buffer))

    sample = replay_buffer.buffer[0]
    print("Frame shape:", sample["frame"].shape)
    print("Action shape:", np.array(sample["action"]).shape)
    print("Reward:", sample["reward"])
    print("Next frame shape:", sample["next_frame"].shape)
    print("Done:", sample["done"])

    env.close()

    """
    obs, info = env.reset()

    print("Observation type:", type(obs))
    print("Observation:", obs)
    print("Observation shape:", obs.shape if hasattr(obs, "shape") else "no shape")
    print("Action space:", env.action_space)
    print("Observation space:", env.observation_space)
    print("Info keys:", info.keys())

    frame = env.render()
    print("Rendered frame shape:", frame.shape)

    """""""
    
    plt.imshow(frame)
    plt.title("highway-v0 first frame")
    plt.axis("off")
    plt.show()
    """"""

    # run episode with random action
    done = False
    truncated = False
    step_count = 0
    total_reward = 0.0

    while not done and not truncated:
        action = env.action_space.sample()
        print(action)

        next_obs, reward, done, truncated, info = env.step(action)

        print(
            f'step={step_count}, '
            f'action={np.array(action)}, '
            f'reward={reward:.3f}, '
            f'done={done}, truncated={truncated}'
        )

        obs = next_obs
        total_reward += reward
        step_count += 1

    print("Episode finished")
    print("Total steps:", step_count)
    print("Total reward:", total_reward)
    
    env.close()
    """

if __name__ == "__main__":
    main()