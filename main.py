import os
import time
import random
import numpy as np
import torch
import gymnasium as gym
import highway_env

from config import Config, save_config
from agent import Agent
import utils
import run
from replay_buffer import ReplayBuffer
from logger import Logger

# utilize functions
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def make_env(config):
    return gym.make(
        "highway-v0",
        config=config.env_config,
        render_mode="rgb_array",
    )

def main():
    config = Config()
    seed = 42
    set_seed(seed)

    ####### config settings
    device = torch.device(config.device)
    config.output_dir = 'outputs'


    # change the name for each experiment
    run_dir = os.path.join(config.output_dir, "recon_tun01")
    config.debug_dir = os.path.join(run_dir, "debug")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(config.debug_dir, exist_ok=True)

    env = make_env(config)
    eval_env = make_env(config)

    config.action_dim = env.action_space.shape[0]

    ####### create agent, buffer, loffer
    agent = Agent(config=config, device=device)

    replay_buffer = ReplayBuffer(
        capacity=config.buffer_capacity,
        seq_len=config.seq_len,
    )

    train_logger = Logger(run_dir, filename="train.csv")
    eval_logger = Logger(run_dir, filename="eval.csv")

    ####### load random buffer
    buffer_path = "outputs/warmup_buffer_10000.pkl"
    replay_buffer.load(buffer_path)
    print(f"Warmup done. buffer size = {len(replay_buffer)}")

    ####### experiment setup
    config.mode = "train"

    config.train_steps = 5000
    config.collect_steps = 10
    config.log_every = 200

    config.save_every = 200
    config.eval_every = 100

    save_config(config, run_dir)
    #save_config(config, run_dir + "/eval")

    if config.mode == "train":
        run.train(
            config=config,
            env=env,
            agent=agent,
            replay_buffer=replay_buffer,
            train_logger=train_logger,
            run_dir=run_dir,
        )

    elif config.mode == "eval":
        checkpoint_path = "outputs/tuning01/checkpoint_step_600.pt"
        checkpoint = torch.load(checkpoint_path, map_location=device)
        agent.world_model.load_state_dict(checkpoint["world_model"])
        agent.actor_critic.load_state_dict(checkpoint["actor_critic"])
        metrics, frames = run.eval(
            config=config,
            env=eval_env,
            agent=agent,
            eval_logger=eval_logger,
            save_frames=False,
        )

    elif config.mode == "train_eval":
        metrics, frames = run.train_eval(
            config=config,
            env=env,
            eval_env=eval_env,
            agent=agent,
            replay_buffer=replay_buffer,
            train_logger=train_logger,
            eval_logger=eval_logger,
            run_dir=run_dir,
            save_frames=False,
        )

    else:
        raise Exception("Unknown config.mode")

if __name__ == "__main__":
    main()