"""
tactilai/agents/icm.py

Intrinsic Curiosity Module (ICM) — Pathak et al. 2017
https://arxiv.org/abs/1705.05363

Architecture
────────────

  φ (encoder)
  ───────────
  obs (388,) → CNN + MLP → latent state (LATENT_DIM,)
  Reuses the same preprocessing as ActorCritic (preprocess_obs)
  but has its own weights — ICM learns a different representation
  focused on controllable aspects of the environment.

  Inverse model
  ─────────────
  [φ(s) || φ(s')] (2 * LATENT_DIM,) → MLP → action logits (ACTION_SIZE,)
  Trained with cross-entropy loss against the true action.
  This forces φ to encode only state features the agent can influence.

  Forward model
  ─────────────
  [φ(s) || action_onehot] (LATENT_DIM + ACTION_SIZE,) → MLP → φ̂(s') (LATENT_DIM,)
  Trained with MSE loss against φ(s').
  Prediction error = intrinsic reward signal.

Loss
────
  L_icm = β * L_forward + (1 - β) * L_inverse

  where β ∈ [0,1] controls the balance between the two models.
  Typical value: β = 0.2 (more weight on inverse to stabilise encoder).

Intrinsic reward
────────────────
  r_i = η / 2 * ||φ̂(s') - φ(s')||²

  where η is a scaling factor (default 0.01).
  This is added to the extrinsic reward before GAE computation.

Integration with PPO
────────────────────
  total_reward = r_extrinsic + η * r_intrinsic
  ICM is updated jointly with the PPO network using a combined loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from tactilai.agents.network import CNN_FEATURES, UNIT_FEATURES, preprocess_obs
from tactilai.env.gym_wrapper import ACTION_SIZE

# ── Hyperparameters ───────────────────────────────────────────────────────────

LATENT_DIM = 256  # dimension of the encoded state φ(s)
ICM_BETA = 0.2  # weight of forward loss vs inverse loss
ICM_ETA = 0.01  # intrinsic reward scaling factor


# ── State encoder (φ) ─────────────────────────────────────────────────────────


class StateEncoder(nn.Module):
    """
    Encodes a raw observation into a compact latent state φ(s).

    Uses the same preprocess_obs pipeline as ActorCritic but with
    independent weights — ICM learns a representation focused on
    controllable state features, not value estimation.

    Input  : obs (B, OBS_SIZE)
    Output : φ(s) (B, LATENT_DIM)
    """

    def __init__(self) -> None:
        super().__init__()

        # CNN stream (mirrors CNNEncoder from network.py)
        self.conv = nn.Sequential(
            nn.Conv2d(9, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.cnn_fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, CNN_FEATURES),
            nn.ReLU(),
        )

        # Unit feature stream
        self.unit_mlp = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, UNIT_FEATURES),
            nn.ReLU(),
        )

        # Fusion → latent
        self.fusion = nn.Sequential(
            nn.Linear(CNN_FEATURES + UNIT_FEATURES, LATENT_DIM),
            nn.ReLU(),
        )

    def forward(self, obs: Tensor) -> Tensor:
        """
        Parameters
        ----------
        obs : Tensor of shape (B, OBS_SIZE)

        Returns
        -------
        phi : Tensor of shape (B, LATENT_DIM)
        """
        grid, unit = preprocess_obs(obs)

        cnn_out = self.conv(grid).flatten(start_dim=1)
        cnn_feat = self.cnn_fc(cnn_out)
        unit_feat = self.unit_mlp(unit)

        fused = torch.cat([cnn_feat, unit_feat], dim=-1)
        return self.fusion(fused)


# ── Inverse model ─────────────────────────────────────────────────────────────


class InverseModel(nn.Module):
    """
    Predicts the action taken between two consecutive states.

    Input  : [φ(s) || φ(s')] of shape (B, 2 * LATENT_DIM)
    Output : action logits    of shape (B, ACTION_SIZE)

    Loss   : CrossEntropyLoss(logits, true_action)
    """

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * LATENT_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, ACTION_SIZE),
        )

    def forward(self, phi_s: Tensor, phi_s_next: Tensor) -> Tensor:
        """
        Parameters
        ----------
        phi_s      : Tensor (B, LATENT_DIM) — encoded current state
        phi_s_next : Tensor (B, LATENT_DIM) — encoded next state

        Returns
        -------
        logits : Tensor (B, ACTION_SIZE)
        """
        x = torch.cat([phi_s, phi_s_next], dim=-1)
        return self.net(x)


# ── Forward model ─────────────────────────────────────────────────────────────


class ForwardModel(nn.Module):
    """
    Predicts the next latent state given the current state and action.

    Input  : [φ(s) || action_onehot] of shape (B, LATENT_DIM + ACTION_SIZE)
    Output : φ̂(s') of shape (B, LATENT_DIM)

    Loss   : MSE(φ̂(s'), φ(s').detach())
    Note   : φ(s') is detached — the forward model does not update the encoder.
             Only the inverse model drives encoder learning.
    """

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(LATENT_DIM + ACTION_SIZE, 256),
            nn.ReLU(),
            nn.Linear(256, LATENT_DIM),
        )

    def forward(self, phi_s: Tensor, action: Tensor) -> Tensor:
        """
        Parameters
        ----------
        phi_s  : Tensor (B, LATENT_DIM)
        action : Tensor (B,) of dtype long — action indices

        Returns
        -------
        phi_s_next_hat : Tensor (B, LATENT_DIM) — predicted next latent state
        """
        # One-hot encode the action
        action_onehot = F.one_hot(action, num_classes=ACTION_SIZE).float()
        x = torch.cat([phi_s, action_onehot], dim=-1)
        return self.net(x)


# ── ICM module ────────────────────────────────────────────────────────────────


class ICM(nn.Module):
    """
    Full Intrinsic Curiosity Module.

    Combines the state encoder, inverse model, and forward model.
    Provides:
      - intrinsic_reward() : per-step curiosity reward
      - loss()             : combined ICM training loss

    Parameters
    ----------
    device : torch.device
    beta   : float — forward/inverse loss balance (default 0.2)
    eta    : float — intrinsic reward scaling     (default 0.01)

    Example
    -------
    icm = ICM(device)

    # During rollout collection
    r_i = icm.intrinsic_reward(obs, obs_next, action)   # (B,)

    # During PPO update
    loss = icm.loss(obs, obs_next, action)
    loss.backward()
    """

    def __init__(
        self,
        device: torch.device,
        beta: float = ICM_BETA,
        eta: float = ICM_ETA,
    ) -> None:
        super().__init__()
        self.device = device
        self.beta = beta
        self.eta = eta

        self.encoder = StateEncoder()
        self.inverse_model = InverseModel()
        self.forward_model = ForwardModel()

        self._init_weights()
        self.to(device)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _encode(self, obs: Tensor) -> Tensor:
        """Encodes obs to latent state, moving to device."""
        return self.encoder(obs.to(self.device))

    def intrinsic_reward(
        self,
        obs: Tensor,
        obs_next: Tensor,
        action: Tensor,
    ) -> Tensor:
        """
        Computes the per-step intrinsic (curiosity) reward.

        r_i = η / 2 * ||φ̂(s') - φ(s')||²

        Parameters
        ----------
        obs      : Tensor (B, OBS_SIZE)
        obs_next : Tensor (B, OBS_SIZE)
        action   : Tensor (B,) long

        Returns
        -------
        r_intrinsic : Tensor (B,) — detached, ready to add to extrinsic reward
        """
        with torch.no_grad():
            phi_s = self._encode(obs)
            phi_s_next = self._encode(obs_next)
            phi_s_next_hat = self.forward_model(phi_s, action.to(self.device))

            # Per-sample MSE
            r_i = self.eta / 2.0 * ((phi_s_next_hat - phi_s_next) ** 2).sum(dim=-1)

        return r_i  # (B,) — already detached via no_grad

    def loss(
        self,
        obs: Tensor,
        obs_next: Tensor,
        action: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Computes the combined ICM loss.

        L_icm = β * L_forward + (1 - β) * L_inverse

        Parameters
        ----------
        obs      : Tensor (B, OBS_SIZE)
        obs_next : Tensor (B, OBS_SIZE)
        action   : Tensor (B,) long

        Returns
        -------
        loss_icm     : Tensor scalar — total ICM loss
        loss_forward : Tensor scalar — forward model MSE
        loss_inverse : Tensor scalar — inverse model cross-entropy
        """
        action = action.to(self.device)

        phi_s = self._encode(obs)
        phi_s_next = self._encode(obs_next)

        # ── Forward loss ──────────────────────────────────────────────────────
        phi_s_next_hat = self.forward_model(phi_s, action)
        # Detach target — forward model does not update encoder
        loss_forward = F.mse_loss(phi_s_next_hat, phi_s_next.detach())

        # ── Inverse loss ──────────────────────────────────────────────────────
        action_logits = self.inverse_model(phi_s, phi_s_next)
        loss_inverse = F.cross_entropy(action_logits, action)

        # ── Combined ──────────────────────────────────────────────────────────
        loss_icm = self.beta * loss_forward + (1.0 - self.beta) * loss_inverse

        return loss_icm, loss_forward, loss_inverse
