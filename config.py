import json
import os
from dataclasses import dataclass, field, asdict

@dataclass
class Config:
    output_dir: str = "/content/drive/MyDrive/WorldModel/outputs"

    # env
    env_config: dict = field(default_factory=lambda: {
        "action": {
            "type": "ContinuousAction"
        },
        "lanes_count": 4,
    })

    # mode
    mode: str = "train_eval"   # "train", "eval", "train_eval"

    crop_size: tuple = (64, 64)

    # replay buffer
    buffer_capacity: int = 100000
    batch_size: int = 16
    seq_len: int = 20

    # architecture
    input_channels: int = 3

    embedding_dim: int = 4096
    deter_dim: int = 200
    stoch_dim: int = 30
    model_hidden_dim: int = 200
    ac_hidden_dim: int = 128

    lr_world_model: float = 1e-4
    lr_actor_critic: float = 1e-4
    imagination_horizon: int = 20

    action_dim: int | None = None  # insert action dim after creating environment
    gamma: float = 0.99
    lambda_: float = 0.95

    grad_clip: float = 100.0

    # training
    warmup_steps: int = 5000
    train_steps: int = 10000
    wm_updates_per_step: int = 1
    ac_updates_per_step: int = 1
    collect_steps: int = 100

    # evaluation
    eval_episodes: int = 5
    eval_every: int = 200

    # logging / saving
    log_every: int = 50
    save_every: int = 500

    # device
    device: str = "cuda"

def save_config(config, run_dir):
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(asdict(config), f, indent=2)