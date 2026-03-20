import torch
import torch.nn as nn

from models.rssm import RSSMState
from models.world_model import WorldModel
from models.actor_critic import ActorCritic
from utils import *

class Agent:
    def __init__(self, config, device):
        self.config = config
        self.device = device

        self.frame_shape = (
            config.input_channels,
            config.crop_size[0],
            config.crop_size[1],
        )
        self.action_dim = config.action_dim
        self.feat_dim = config.deter_dim + config.stoch_dim

        # --------------------------------------------------
        # models
        # --------------------------------------------------
        self.world_model = WorldModel(
            obs_shape=self.frame_shape,
            action_dim=config.action_dim,
            embedding_dim=config.embedding_dim,
            deter_dim=config.deter_dim,
            stoch_dim=config.stoch_dim,
            model_hidden_dim=config.model_hidden_dim,
        ).to(self.device)

        self.actor_critic = ActorCritic(
            feat_dim=self.feat_dim,
            action_dim=config.action_dim,
            hidden_dim=config.ac_hidden_dim,
            gamma=config.gamma,
            lambda_=config.lambda_,
        ).to(self.device)

        # optimizers
        self.wm_optimizer = torch.optim.Adam(
            self.world_model.parameters(),
            lr=config.lr_world_model,
        )
        self.ac_optimizer = torch.optim.Adam(
            self.actor_critic.parameters(),
            lr=config.lr_actor_critic,
        )

    def prepare_batch(self, batch):
        frames = to_tensor(batch["frames"], self.device)  # (B, T, C, H, W)
        actions = to_tensor(batch["actions"], self.device)  # (B, T, A)

        rewards = to_tensor(batch["rewards"], self.device)  # (B, T) or (B, T, 1)
        if rewards.dim() == 2:
            rewards = rewards.unsqueeze(-1)  # -> (B, T, 1)

        dones = batch.get("dones", None)
        if dones is not None:
            dones = to_tensor(dones, self.device)
            if dones.dim() == 2:
                dones = dones.unsqueeze(-1)

        truncateds = batch.get("truncateds", None)
        if truncateds is not None:
            truncateds = to_tensor(truncateds, self.device)
            if truncateds.dim() == 2:
                truncateds = truncateds.unsqueeze(-1)

        return frames, actions, rewards, dones, truncateds

    def initial_state(self, batch_size):
        return self.world_model.initial_state(batch_size, self.device)

    # extract the last posterior state from the real sequence,
    # and use it as the starting state for imagination rollout
    def last_state(self, state_seq):
        return RSSMState(
            deter=state_seq.deter[:, -1].detach(),
            stoch=state_seq.stoch[:, -1].detach(),
            mean=state_seq.mean[:, -1].detach(),
            std=state_seq.std[:, -1].detach(),
        )

    # action selection
    @torch.no_grad()
    def act(self, frame, prev_state=None, prev_action=None, deterministic=False):
        self.world_model.eval()
        self.actor_critic.eval()

        frame = to_tensor(frame, self.device).unsqueeze(0)  # (1, C, H, W)

        embed = self.world_model.encoder(frame)  # (1, embedding_dim)

        if prev_state is None:
            prev_state = self.initial_state(batch_size=1)

        if prev_action is None:
            prev_action = torch.zeros(1, self.config.action_dim, device=self.device)
        else:
            prev_action = to_tensor(prev_action, self.device)
            if prev_action.dim() == 1:
                prev_action = prev_action.unsqueeze(0)

        post_state, _ = self.world_model.rssm.rssm_step(
            prev_state=prev_state,
            prev_act=prev_action,
            embed=embed,
        )

        feat = self.world_model.rssm.get_feature(post_state)  # (1, feat_dim)
        action = self.actor_critic.policy(feat, deterministic=deterministic)  # (1, A)

        return action.squeeze(0).cpu().numpy(), post_state

    # world model train
    def train_world_model(self, batch):
        self.world_model.train()

        frames, actions, rewards, dones, truncateds = self.prepare_batch(batch)
        init_state = self.initial_state(frames.shape[0])

        losses = self.world_model.compute_losses(
            frame=frames,
            action=actions,
            reward=rewards,
            dones=dones,
            truncateds=truncateds,
            state=init_state,
        )

        self.wm_optimizer.zero_grad()
        losses["total_loss"].backward()

        if getattr(self.config, "grad_clip", None) is not None:
            nn.utils.clip_grad_norm_(self.world_model.parameters(), self.config.grad_clip)

        self.wm_optimizer.step()

        return {
            "wm_total_loss": losses["total_loss"].item(),
            "wm_reward_loss": losses["reward_loss"].item(),
            "wm_recon_loss": losses["recon_loss"].item(),
            "wm_kl_loss": losses["kl_loss"].item(),
            "wm_continue_loss": losses["continue_loss"].item(),
        }

    # actor-critic train
    def train_actor_critic(self, batch):
        self.world_model.train()
        self.actor_critic.train()

        frames, actions, _, _, _ = self.prepare_batch(batch)
        init_state = self.initial_state(frames.shape[0])

        # freeze world model parameters during actor-critic update
        for p in self.world_model.parameters():
            p.requires_grad = False

        # posterior from real replay sequence
        post, _, _, _ = self.world_model(frames, actions, init_state)
        start_state = self.last_state(post)

        imagined_states, imagined_feats, imagined_rewards, imagined_values, imagined_cont = (
            self.actor_critic.imagine_trajectory(
                world_model=self.world_model,
                start_state=start_state,
                horizon=self.config.imagination_horizon,
            )
        )

        bootstrap = imagined_values[:, -1].detach()

        lambda_targets = self.actor_critic.compute_lambda_targets(
            rewards=imagined_rewards,
            values=imagined_values,
            continues=imagined_cont,
            bootstrap=bootstrap,
        )

        actor_loss = self.actor_critic.compute_actor_loss(
            lambda_targets=lambda_targets,
            continues=imagined_cont,
        )

        critic_loss = self.actor_critic.compute_critic_loss(
            imagined_feats=imagined_feats,
            lambda_targets=lambda_targets,
        )

        total_loss = actor_loss + critic_loss

        self.ac_optimizer.zero_grad()
        total_loss.backward()

        if getattr(self.config, "grad_clip", None) is not None:
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.config.grad_clip)

        self.ac_optimizer.step()

        # clear possible leftover grads on world model from imagination graph
        self.wm_optimizer.zero_grad(set_to_none=True)

        # unfreeze world model again
        for p in self.world_model.parameters():
            p.requires_grad = True

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "ac_total_loss": total_loss.item(),
        }

    # evaluation helper. returns tensors for visualization / sanity check
    @torch.no_grad()
    def report(self, batch):
        self.world_model.eval()
        self.actor_critic.eval()

        frames, actions, rewards, _, _ = self.prepare_batch(batch)
        init_state = self.initial_state(frames.shape[0])

        post, prior, pred_reward, recon_obs = self.world_model(
            frames, actions, init_state
        )

        metrics = {
            "reward_mse": torch.mean((pred_reward - rewards) ** 2).item(),
            "recon_mse": torch.mean((recon_obs - frames) ** 2).item(),
            "kl": self.world_model.rssm.kl_loss(post, prior).item(),
        }

        return {
            "metrics": metrics,
            "target_frames": frames.detach().cpu(),
            "recon_frames": recon_obs.detach().cpu(),
            "target_rewards": rewards.detach().cpu(),
            "pred_rewards": pred_reward.detach().cpu(),
            "post": post,
            "prior": prior,
        }

    # --------------------------------------------------
    # checkpoint
    # --------------------------------------------------
    def save_checkpoint(self, run_dir, step=None, best_score=None):
        os.makedirs(os.path.dirname(run_dir), exist_ok=True)

        checkpoint = {
            "world_model": self.world_model.state_dict(),
            "actor_critic": self.actor_critic.state_dict(),
            "wm_optimizer": self.wm_optimizer.state_dict(),
            "ac_optimizer": self.ac_optimizer.state_dict(),
            "step": step,
            "best_score": best_score,
        }
        torch.save(checkpoint, run_dir)

    def load_checkpoint(self, path, load_optimizer=True):
        checkpoint = torch.load(path, map_location=self.device)

        self.world_model.load_state_dict(checkpoint["world_model"])
        self.actor_critic.load_state_dict(checkpoint["actor_critic"])

        if load_optimizer:
            if "wm_optimizer" in checkpoint:
                self.wm_optimizer.load_state_dict(checkpoint["wm_optimizer"])
            if "ac_optimizer" in checkpoint:
                self.ac_optimizer.load_state_dict(checkpoint["ac_optimizer"])

        return checkpoint