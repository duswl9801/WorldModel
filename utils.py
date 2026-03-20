import torch
import os
import cv2
import numpy as np
from datetime import datetime
import imageio

def to_tensor(x, device):
    return torch.as_tensor(x, dtype=torch.float32, device=device)

def preprocess_frame(frame, crop_size):
    frame = cv2.resize(frame, crop_size) # downscale the image for model input
    frame = frame.astype(np.float32) / 255.0 # normalize
    frame = np.transpose(frame, (2, 0, 1))  # HWC -> CHW
    return frame

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

