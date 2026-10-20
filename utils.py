import torch
import os
import cv2
import numpy as np
from datetime import datetime
import imageio
from PIL import Image

def preprocess_frame(frame, crop_size):
    frame = cv2.resize(frame, crop_size) # downscale the image for model input
    frame = frame.astype(np.float32) / 255.0 # normalize
    frame = np.transpose(frame, (2, 0, 1))  # HWC -> CHW
    return frame

def collect_random_steps(env, replay_buffer, num_steps, crop_size, seed=None):
    obs, _ = env.reset(seed=seed)

    frame = env.render()
    frame = preprocess_frame(frame, crop_size)

    for step in range(num_steps):
        action = env.action_space.sample()
        _, reward, done, truncated, _ = env.step(action)

        next_frame = env.render()
        next_frame = preprocess_frame(next_frame, crop_size)

        replay_buffer.add(
            frame=frame,
            action=action,
            reward=reward,
            next_frame=next_frame,
            done=done,
            truncated=truncated,
        )

        frame = next_frame

        if done or truncated:
            _, _ = env.reset(seed=seed)
            frame = env.render()
            frame = preprocess_frame(frame, crop_size)

def collect_agent_steps(env, agent, replay_buffer, num_steps, crop_size, deterministic=False, seed=None):
    _, _ = env.reset(seed=seed)

    frame = env.render()
    frame = preprocess_frame(frame, crop_size)

    prev_state = None
    prev_action = None

    for _ in range(num_steps):
        action, post_state = agent.act(
            frame=frame,
            prev_state=prev_state,
            prev_action=prev_action,
            deterministic=deterministic,
        )

        _, reward, done, truncated, _ = env.step(action)

        next_frame = env.render()
        next_frame = preprocess_frame(next_frame, crop_size)

        replay_buffer.add(
            frame=frame,
            action=action,
            reward=reward,
            next_frame=next_frame,
            done=done,
            truncated=truncated,
        )

        frame = next_frame

        if done or truncated:
            _, _ = env.reset(seed=seed)
            frame = env.render()
            frame = preprocess_frame(frame, crop_size)
            prev_state = None
            prev_action = None
        else:
            prev_state = post_state
            prev_action = action

@torch.no_grad()
def evaluate_agent(env, agent, crop_size,  num_episodes=5, save_frames=False):
    rewards = []
    all_frames = []

    for _ in range(num_episodes):
        _, _ = env.reset()

        frame = env.render()
        frame = preprocess_frame(frame, crop_size)

        done = False
        truncated = False
        episode_reward = 0.0

        prev_state = None
        prev_action = None

        while not (done or truncated):
            if save_frames:
                render_frame = env.render()
                if render_frame is not None:
                    all_frames.append(render_frame)

            action, post_state = agent.act(
                frame=frame,
                prev_state=prev_state,
                prev_action=prev_action,
                deterministic=True,
            )

            _, reward, done, truncated, _ = env.step(action)

            next_frame = env.render()
            next_frame = preprocess_frame(next_frame, crop_size)

            episode_reward += reward
            frame = next_frame
            prev_state = post_state
            prev_action = action

        rewards.append(episode_reward)

    metrics = {
        "eval/episode_reward_mean": float(np.mean(rewards)),
        "eval/episode_reward_std": float(np.std(rewards)),
        "eval/episode_reward_min": float(np.min(rewards)),
        "eval/episode_reward_max": float(np.max(rewards)),
    }

    return metrics, all_frames

def to_tensor(x, device):
    return torch.as_tensor(x, dtype=torch.float32, device=device)

def make_run_dir(output_dir, run_name=None):
    if run_name is None:
        run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")

    run_dir = os.path.join(output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def save_video(frames, run_dir, fps=10):
    os.makedirs(os.path.dirname(run_dir), exist_ok=True)
    imageio.mimsave(run_dir, frames, fps=fps)

def save_target_recon_video(target_frames, recon_frames, run_dir, fps=10):
    os.makedirs(os.path.dirname(run_dir), exist_ok=True)

    video_frames = []
    for target, recon in zip(target_frames, recon_frames):
        combined = np.concatenate([target, recon], axis=1)
        video_frames.append(combined)

    imageio.mimsave(run_dir, video_frames, fps=fps)

# debug reconstruction
def save_reconstruction_grid(frames, recons, debug_dir, max_t=8):
    """
    frames:  (B, T, C, H, W)
    recons:  (B, T, C, H, W)
    save_path: output image path
    """
    os.makedirs(os.path.dirname(debug_dir), exist_ok=True)

    # first batch only
    frames = frames[0].detach().cpu()
    recons = recons[0].detach().cpu()

    T = min(frames.shape[0], max_t)

    rows = []
    for t in range(T):
        gt = frames[t]
        rc = recons[t]

        # CHW -> HWC
        gt = gt.permute(1, 2, 0).numpy()
        rc = rc.permute(1, 2, 0).numpy()

        # assume normalized to [0,1]
        gt = np.clip(gt, 0.0, 1.0)
        rc = np.clip(rc, 0.0, 1.0)

        gt = (gt * 255).astype(np.uint8)
        rc = (rc * 255).astype(np.uint8)

        # concat original and recon horizontally
        pair = np.concatenate([gt, rc], axis=1)
        rows.append(pair)

    # stack time steps vertically
    grid = np.concatenate(rows, axis=0)
    Image.fromarray(grid).save(debug_dir)