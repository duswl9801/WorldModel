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
    def compute_losses(self, frame, action, reward, state=None):
        kl_scale = 1.0 # can be tuned later to balance reconstruction, reward, and KL losses

        post, prior, pred_reward, recon_obs = self.forward(frame, action, state)

        reward_loss = F.mse_loss(pred_reward, reward)
        recon_loss = F.mse_loss(recon_obs, frame)
        kl_value = self.rssm.kl_loss(post, prior)

        print("reward:", reward_loss.item())
        print("recon:", recon_loss.item())
        print("kl:", kl_value.item())

        total_loss = reward_loss + recon_loss + kl_scale * kl_value

        return {
            "total_loss": total_loss,
            "reward_loss": reward_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_value,
        }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dummy config
    B = 4
    T = 6
    C, H, W = 3, 64, 64
    action_dim = 3
    embedding_dim = 4096  # encoder output dim
    deter_dim = 200
    stoch_dim = 30
    model_hidden_dim = 200

    # build world model
    world_model = WorldModel(
        obs_shape=(C, H, W),
        action_dim=action_dim,
        embedding_dim=embedding_dim,
        deter_dim=deter_dim,
        stoch_dim=stoch_dim,
        model_hidden_dim=model_hidden_dim,
    ).to(device)

    # dummy batch
    frames = torch.randn(B, T, C, H, W, device=device)
    actions = torch.randn(B, T, action_dim, device=device)
    rewards = torch.randn(B, T, 1, device=device)

    # initial rssm state
    init_state = world_model.initial_state(B, device)

    print("===== forward test =====")
    post, prior, pred_reward, recon_frame = world_model(frames, actions, init_state)

    print("pred_reward shape:", pred_reward.shape)  # expected: (B, T, 1)
    print("recon_frame shape:", recon_frame.shape)  # expected: (B, T, 3, 64, 64)
    print("post.deter shape:", post.deter.shape)  # expected: (B, T, deter_dim)
    print("post.stoch shape:", post.stoch.shape)  # expected: (B, T, stoch_dim)
    print("prior.deter shape:", prior.deter.shape)  # expected: (B, T, deter_dim)
    print("prior.stoch shape:", prior.stoch.shape)  # expected: (B, T, stoch_dim)

    print("\n===== loss test =====")
    losses = world_model.compute_losses(frames, actions, rewards, init_state)

    for name, value in losses.items():
        print(f"{name}: {value.item():.6f}")

    print("\n===== backward test =====")
    losses["total_loss"].backward()
    print("backward success")

    # optional optimizer step test
    optimizer = torch.optim.Adam(world_model.parameters(), lr=1e-4)
    optimizer.step()
    optimizer.zero_grad()

    print("optimizer step success")

if __name__ == "__main__":
    main()