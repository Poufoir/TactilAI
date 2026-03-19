"""
tactilai/agents/network.py

Neural network architecture for TactilAI PPO agent.

Observation pipeline
────────────────────
The 388-feature flat observation is split into two streams:

  grid_obs  : first 378 features → reshaped to (9, 16, 16) → CNN encoder
  unit_obs  : last  10 features  → MLP branch

The 9 CNN input channels are:
  0  terrain type        (normalised to [0,1])
  1  blue unit presence  (1.0 if blue unit on tile)
  2  red unit presence   (1.0 if red unit on tile)
  3  blue HP ratio       (hp/max_hp, 0 elsewhere)
  4  red HP ratio        (hp/max_hp, 0 elsewhere)
  5  blue has_moved      (1.0 if unit on tile has moved)
  6  blue has_acted      (1.0 if unit on tile has acted)
  7  red has_moved
  8  red has_acted

The terrain channel uses the first 256 features directly.
Unit channels are rebuilt from the 120 unit features (12 units × 10 features).

Architecture
────────────
  CNN encoder  : 3 conv layers → flat feature vector (256-d)
  Unit MLP     : 2 linear layers → 64-d
  Fusion MLP   : concat(256 + 64) → 256-d shared trunk
  Actor head   : Linear(256, ACTION_SIZE)   → logits
  Critic head  : Linear(256, 1)             → state value V(s)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from tactilai.env.grid import GRID_SIZE, NUM_UNITS
from tactilai.env.gym_wrapper import ACTION_SIZE, GRID_OBS_SIZE, UNIT_OBS_SIZE

# ── Constants ─────────────────────────────────────────────────────────────────

CNN_CHANNELS = 9  # number of spatial input channels
CNN_FEATURES = 256  # flattened CNN output size
UNIT_FEATURES = 64  # unit MLP output size
TRUNK_FEATURES = 256  # shared fusion trunk size


# ── Observation preprocessing ─────────────────────────────────────────────────


def preprocess_obs(obs: Tensor) -> tuple[Tensor, Tensor]:
    """
    Splits and reshapes a flat observation batch into CNN and unit streams.

    Parameters
    ----------
    obs : Tensor of shape (B, OBS_SIZE)

    Returns
    -------
    grid_tensor : Tensor of shape (B, CNN_CHANNELS, GRID_SIZE, GRID_SIZE)
    unit_tensor : Tensor of shape (B, UNIT_OBS_SIZE)
    """
    B = obs.shape[0]

    # ── Split ─────────────────────────────────────────────────────────────────
    grid_flat = obs[:, :GRID_OBS_SIZE]  # (B, 378)
    unit_obs = obs[:, GRID_OBS_SIZE:]  # (B, 10)

    terrain_flat = grid_flat[:, : GRID_SIZE * GRID_SIZE]  # (B, 256)
    units_flat = grid_flat[:, GRID_SIZE * GRID_SIZE : -2]  # (B, 120)
    # meta (2 features) is included in unit_obs implicitly via unit_obs concat

    # ── Terrain channel (B, 16, 16) ───────────────────────────────────────────
    terrain_ch = terrain_flat.view(B, 1, GRID_SIZE, GRID_SIZE)  # (B,1,16,16)

    # ── Unit channels ─────────────────────────────────────────────────────────
    # units_flat : (B, 120) = 12 units × 10 features
    # Feature layout per unit (from to_obs_vector):
    #   0: class_id  1: team  2: row  3: col  4: hp_ratio
    #   5: has_moved 6: has_acted 7: movement 8: min_range 9: max_range
    n_units = NUM_UNITS * 2
    unit_feats = units_flat.view(B, n_units, 10)  # (B, 12, 10)

    # Initialise spatial channels
    presence_blue = torch.zeros(B, 1, GRID_SIZE, GRID_SIZE, device=obs.device)
    presence_red = torch.zeros(B, 1, GRID_SIZE, GRID_SIZE, device=obs.device)
    hp_blue = torch.zeros(B, 1, GRID_SIZE, GRID_SIZE, device=obs.device)
    hp_red = torch.zeros(B, 1, GRID_SIZE, GRID_SIZE, device=obs.device)
    moved_blue = torch.zeros(B, 1, GRID_SIZE, GRID_SIZE, device=obs.device)
    acted_blue = torch.zeros(B, 1, GRID_SIZE, GRID_SIZE, device=obs.device)
    moved_red = torch.zeros(B, 1, GRID_SIZE, GRID_SIZE, device=obs.device)
    acted_red = torch.zeros(B, 1, GRID_SIZE, GRID_SIZE, device=obs.device)

    for i in range(n_units):
        team = unit_feats[:, i, 1]  # 0=blue, 1=red
        row_idx = (unit_feats[:, i, 2] * (GRID_SIZE - 1)).long()
        col_idx = (unit_feats[:, i, 3] * (GRID_SIZE - 1)).long()
        hp_ratio = unit_feats[:, i, 4]
        has_moved = unit_feats[:, i, 5]
        has_acted = unit_feats[:, i, 6]
        alive_mask = hp_ratio > 0  # dead units have zero hp_ratio

        for b in range(B):
            if not alive_mask[b]:
                continue
            r, c = row_idx[b].item(), col_idx[b].item()
            if team[b] < 0.5:  # blue
                presence_blue[b, 0, r, c] = 1.0
                hp_blue[b, 0, r, c] = hp_ratio[b]
                moved_blue[b, 0, r, c] = has_moved[b]
                acted_blue[b, 0, r, c] = has_acted[b]
            else:  # red
                presence_red[b, 0, r, c] = 1.0
                hp_red[b, 0, r, c] = hp_ratio[b]
                moved_red[b, 0, r, c] = has_moved[b]
                acted_red[b, 0, r, c] = has_acted[b]

    grid_tensor = torch.cat(
        [
            terrain_ch,
            presence_blue,
            presence_red,
            hp_blue,
            hp_red,
            moved_blue,
            acted_blue,
            moved_red,
            acted_red,
        ],
        dim=1,
    )  # (B, 9, 16, 16)

    return grid_tensor, unit_obs


# ── CNN encoder ───────────────────────────────────────────────────────────────


class CNNEncoder(nn.Module):
    """
    3-layer CNN that encodes the (9, 16, 16) spatial grid into a 256-d vector.

    Conv layers use padding=1 to preserve spatial dimensions, followed by
    a final adaptive pool to fix the output size regardless of grid dimensions.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(CNN_CHANNELS, 32, kernel_size=3, padding=1),  # (B,32,16,16)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # (B,64,16,16)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # (B,64,16,16)
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # (B,64,4,4)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, CNN_FEATURES),
            nn.ReLU(),
        )

    def forward(self, grid: Tensor) -> Tensor:
        """
        Parameters
        ----------
        grid : Tensor of shape (B, 9, 16, 16)

        Returns
        -------
        Tensor of shape (B, CNN_FEATURES)
        """
        x = self.conv(grid)
        x = x.flatten(start_dim=1)
        return self.fc(x)


# ── Actor-Critic network ──────────────────────────────────────────────────────


class ActorCritic(nn.Module):
    """
    Full Actor-Critic network for PPO with action masking.

    Architecture:
      obs → preprocess → CNN encoder (256) + unit MLP (64)
                       → fusion trunk (256)
                       → actor head (ACTION_SIZE logits)
                       → critic head (1 scalar)

    Usage
    -----
    net = ActorCritic(device)
    logits, value = net(obs)                     # forward pass
    dist  = net.masked_distribution(obs, mask)   # masked categorical
    action = dist.sample()
    log_prob = dist.log_prob(action)
    """

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.device = device

        # CNN stream
        self.cnn_encoder = CNNEncoder()

        # Unit feature MLP stream
        self.unit_mlp = nn.Sequential(
            nn.Linear(UNIT_OBS_SIZE, 64),
            nn.ReLU(),
            nn.Linear(64, UNIT_FEATURES),
            nn.ReLU(),
        )

        # Shared fusion trunk
        self.trunk = nn.Sequential(
            nn.Linear(CNN_FEATURES + UNIT_FEATURES, TRUNK_FEATURES),
            nn.ReLU(),
            nn.Linear(TRUNK_FEATURES, TRUNK_FEATURES),
            nn.ReLU(),
        )

        # Heads
        self.actor_head = nn.Linear(TRUNK_FEATURES, ACTION_SIZE)
        self.critic_head = nn.Linear(TRUNK_FEATURES, 1)

        self._init_weights()
        self.to(device)

    def _init_weights(self) -> None:
        """Orthogonal init — standard for PPO."""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Smaller gain for actor head (common PPO trick)
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)

    def forward(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        obs : Tensor of shape (B, OBS_SIZE)

        Returns
        -------
        logits : Tensor of shape (B, ACTION_SIZE)
        value  : Tensor of shape (B, 1)
        """
        obs = obs.to(self.device)
        grid, unit = preprocess_obs(obs)

        cnn_feat = self.cnn_encoder(grid)
        unit_feat = self.unit_mlp(unit)

        trunk_in = torch.cat([cnn_feat, unit_feat], dim=-1)
        trunk = self.trunk(trunk_in)

        logits = self.actor_head(trunk)
        value = self.critic_head(trunk)
        return logits, value

    def masked_distribution(
        self, obs: Tensor, mask: Tensor
    ) -> torch.distributions.Categorical:
        """
        Returns a Categorical distribution with illegal actions masked to -inf.

        Parameters
        ----------
        obs  : Tensor of shape (B, OBS_SIZE)
        mask : Tensor of shape (B, ACTION_SIZE) — 1 = legal, 0 = illegal
               dtype can be bool or int8
        """
        logits, _ = self.forward(obs)
        mask = mask.to(self.device).bool()
        # Set logits of illegal actions to -inf before softmax
        logits = logits.masked_fill(~mask, float("-inf"))
        return torch.distributions.Categorical(logits=logits)

    def get_value(self, obs: Tensor) -> Tensor:
        """Returns V(s) only — used in GAE computation."""
        _, value = self.forward(obs)
        return value.squeeze(-1)
