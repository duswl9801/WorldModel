import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder import Encoder
from models.rssm import RSSM, RSSMState

# reconstructing observation
# output shape is the same as the preprocessed input frame shape
class Decoder(nn.Module):
    def __init__(self, feat_dim, obs_shape, hidden_dims=None):
        super().__init__()

        self.feat_dim = feat_dim
        self.obs_shape = obs_shape
        self.hidden_dims = hidden_dims

        # try mirroring encoder
        self.fc = nn.Linear(feat_dim, 256*4*4)

        # 4x4 -> 8x8
        self.conv1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)

        # 8x8 -> 16x16
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)

        # 16x16 -> 32x32
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)

        # 32x32 -> 64x64
        self.conv4 = nn.Conv2d(32, 3, kernel_size=3, padding=1)

        self.gelu = nn.GELU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, feat):
        # output shape (B, T, 3, 64, 64)
        B, T, D = feat.shape
        feat = feat.reshape(B*T, D)

        x = self.fc(feat)
        x = x.view(-1, 256, 4, 4)  # (B*T, 256, 4, 4)

        x = F.interpolate(x, scale_factor=2, mode="nearest")  # 4 -> 8
        x = self.gelu(self.conv1(x))  # (B*T, 128, 8, 8)

        x = F.interpolate(x, scale_factor=2, mode="nearest")  # 8 -> 16
        x = self.gelu(self.conv2(x))  # (B*T, 64, 16, 16)

        x = F.interpolate(x, scale_factor=2, mode="nearest")  # 16 -> 32
        x = self.gelu(self.conv3(x))  # (B*T, 32, 32, 32)

        x = F.interpolate(x, scale_factor=2, mode="nearest")  # 32 -> 64
        x = self.sigmoid(self.conv4(x))  # (B*T, 3, 64, 64)

        x = x.view(B, T, 3, 64, 64)

        return x

# predict reward from RSSM latent features
class RewardModel(nn.Module):
    def __init__(self, feat_dim, hidden_dim):
        super().__init__()

        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim

        self.reward_mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, feat):
        return self.reward_mlp(feat)    # output shape (B, T, 1)

# predict continuation probability c_t in [0,1] (probability that episode continues.)
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

class WorldModel(nn.Module):
    def __init__(self, obs_shape, action_dim, embedding_dim, deter_dim, stoch_dim, model_hidden_dim):
        super().__init__()

        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.embedding_dim = embedding_dim
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.model_hidden_dim = model_hidden_dim

        self.encoder = Encoder()
        self.rssm =  RSSM(self.action_dim,self.embedding_dim, self.deter_dim, self.stoch_dim, self.model_hidden_dim)
        self.decoder = Decoder(self.deter_dim + self.stoch_dim, self.obs_shape, self.model_hidden_dim)
        self.reward_model = RewardModel(self.deter_dim + self.stoch_dim, self.model_hidden_dim)
        self.continue_model = ContinueModel(self.deter_dim + self.stoch_dim, self.model_hidden_dim)

    def initial_state(self, batch_size, device):
        return RSSMState(
            deter=torch.zeros(batch_size, self.deter_dim, device=device),
            stoch=torch.zeros(batch_size, self.stoch_dim, device=device),
            mean=torch.zeros(batch_size, self.stoch_dim, device=device),
            std=torch.ones(batch_size, self.stoch_dim, device=device),
        )

    def encode(self, processed_frame):
        B, T, C, H, W = processed_frame.shape

        processed_frame = processed_frame.reshape(B * T, C, H, W) # merge batch and time because the encoder expects 4D input
        embed = self.encoder(processed_frame) # (B*T, embedding_dim)
        embed = embed.reshape(B, T, -1)

        return embed

    def observe_rollout(self, emb, action, state=None):
        post, prior = self.rssm.observe(emb, action, state)
        return post, prior

    def imagine_rollout(self, policy, start_state, horizon):
        prior = self.rssm.imagine(start_state, policy, horizon)
        return prior

    def decode(self, feat):
        reconstruct_frame = self.decoder(feat)
        return reconstruct_frame

    def predict_reward(self, feat):
        pred_reward = self.reward_model(feat)
        return  pred_reward

    def predict_continue(self, feat):
        pred_continue = self.continue_model(feat)
        return pred_continue

    def forward(self, frame, action, state=None):
        emb = self.encode(frame)

        post, prior = self.observe_rollout(emb, action, state)
        post_feat = self.rssm.get_feature(post)

        pred_reward = self.predict_reward(post_feat)
        recons_frame = self.decode(post_feat)

        return post, prior, pred_reward, recons_frame

    # world model losses inspired by the ELBO(Evidence Lower Bound)
    # 1. reward_loss (predicted reward vs. target reward)
    # 2. kl_loss (prior vs. posterior)
    # 3. recon_loss (reconstructed frame vs. original frame )
    def compute_losses(self, frame, action, reward, dones=None, truncateds=None, state=None):
        kl_scale = 1.0 # can be tuned later to balance reconstruction, reward, and KL losses
        continue_scale = 1.0  # tune later if needed

        post, prior, pred_reward, recon_obs = self.forward(frame, action, state)

        # latent feature from posterior state
        post_feat = self.rssm.get_feature(post)  # (B, T, feat_dim)

        reward_loss = F.mse_loss(pred_reward, reward)
        recon_loss = F.mse_loss(recon_obs, frame)
        kl_value = self.rssm.kl_loss(post, prior)

        # initialize continue loss
        continue_loss = torch.tensor(0.0, device=frame.device)

        terminals = None
        if dones is not None and truncateds is not None:
            terminals = torch.maximum(dones.float(), truncateds.float())  # (B, T, 1)
        elif dones is not None:
            terminals = dones.float()
        elif truncateds is not None:
            terminals = truncateds.float()

        if terminals is not None:
            continue_target = 1.0 - terminals  # continue=1, terminal=0
            pred_continue = self.predict_continue(post_feat)  # (B, T, 1)
            continue_loss = F.binary_cross_entropy(pred_continue, continue_target) # continue model is binary

        total_loss = reward_loss + recon_loss + kl_scale * kl_value + continue_scale * continue_loss

        return {
            "total_loss": total_loss,
            "reward_loss": reward_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_value,
            "continue_loss": continue_loss,
        }
