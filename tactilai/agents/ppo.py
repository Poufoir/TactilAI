"""
tactilai/agents/ppo.py

PPO agent with:
  - Generalized Advantage Estimation (GAE)
  - Mini-batch updates over multiple epochs
  - Action masking integrated in policy loss
  - Separate optimizers for ActorCritic and ICM
  - Intrinsic reward from ICM added to extrinsic reward

Hyperparameters
───────────────
  ROLLOUT_STEPS  : N steps collected before each update
  N_EPOCHS       : number of passes over the rollout buffer per update
  MINI_BATCH     : size of each mini-batch
  GAMMA          : discount factor
  GAE_LAMBDA     : GAE smoothing parameter (0=TD, 1=MC)
  CLIP_EPS       : PPO clipping epsilon
  VF_COEF        : value loss coefficient
  ENT_COEF       : entropy bonus coefficient (encourages exploration)
  MAX_GRAD_NORM  : gradient clipping norm
  LR_POLICY      : learning rate for ActorCritic
  LR_ICM         : learning rate for ICM

PPO loss
────────
  L = -L_clip + VF_COEF * L_value - ENT_COEF * L_entropy + L_icm

  L_clip    = E[min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)]
  L_value   = MSE(V(s), V_target)
  L_entropy = E[H(π(·|s))]
  L_icm     = β * L_forward + (1-β) * L_inverse

GAE
───
  δ_t    = r_t + γ * V(s_{t+1}) * (1 - done_t) - V(s_t)
  A_t    = Σ_{k=0}^{T-t} (γλ)^k * δ_{t+k}
  V_targ = A_t + V(s_t)
"""

from __future__ import annotations

import dataclasses
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from tactilai.agents.icm import ICM
from tactilai.agents.network import ActorCritic

# ── Hyperparameters ───────────────────────────────────────────────────────────

ROLLOUT_STEPS = 512  # steps collected before each update
N_EPOCHS = 4  # passes over rollout buffer per update
MINI_BATCH = 128  # mini-batch size
GAMMA = 0.99  # discount factor
GAE_LAMBDA = 0.95  # GAE lambda
CLIP_EPS = 0.2  # PPO clip epsilon
VF_COEF = 0.5  # value loss weight
ENT_COEF = 0.01  # entropy bonus weight
MAX_GRAD_NORM = 0.5  # gradient clipping
LR_POLICY = 3e-4  # ActorCritic learning rate
LR_ICM = 1e-3  # ICM learning rate


# ── Rollout buffer ────────────────────────────────────────────────────────────


@dataclasses.dataclass
class RolloutBuffer:
    """
    Stores a fixed-length sequence of transitions for one PPO update.

    All tensors live on CPU during collection and are moved to device
    only during the update step to save VRAM.

    Fields
    ------
    obs       : (T, OBS_SIZE)       current observations
    obs_next  : (T, OBS_SIZE)       next observations
    actions   : (T,)                actions taken
    log_probs : (T,)                log π(a|s) at collection time
    rewards   : (T,)                extrinsic rewards
    values    : (T,)                V(s) estimates
    dones     : (T,)                episode termination flags
    masks     : (T, ACTION_SIZE)    action masks
    """

    capacity: int = ROLLOUT_STEPS

    obs: list[Tensor] = dataclasses.field(default_factory=list)
    obs_next: list[Tensor] = dataclasses.field(default_factory=list)
    actions: list[Tensor] = dataclasses.field(default_factory=list)
    log_probs: list[Tensor] = dataclasses.field(default_factory=list)
    rewards: list[float] = dataclasses.field(default_factory=list)
    values: list[Tensor] = dataclasses.field(default_factory=list)
    dones: list[bool] = dataclasses.field(default_factory=list)
    masks: list[Tensor] = dataclasses.field(default_factory=list)

    def add(
        self,
        obs: Tensor,
        obs_next: Tensor,
        action: Tensor,
        log_prob: Tensor,
        reward: float,
        value: Tensor,
        done: bool,
        mask: Tensor,
    ) -> None:
        self.obs.append(obs.cpu())
        self.obs_next.append(obs_next.cpu())
        self.actions.append(action.cpu())
        self.log_probs.append(log_prob.cpu())
        self.rewards.append(reward)
        self.values.append(value.cpu())
        self.dones.append(done)
        self.masks.append(mask.cpu())

    def __len__(self) -> int:
        return len(self.rewards)

    @property
    def is_full(self) -> bool:
        return len(self) >= self.capacity

    def clear(self) -> None:
        self.obs.clear()
        self.obs_next.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
        self.masks.clear()

    def as_tensors(self) -> dict[str, Tensor]:
        """Stack all lists into tensors (still on CPU)."""
        return {
            "obs": torch.stack(self.obs),
            "obs_next": torch.stack(self.obs_next),
            "actions": torch.stack(self.actions),
            "log_probs": torch.stack(self.log_probs),
            "rewards": torch.tensor(self.rewards, dtype=torch.float32),
            "values": torch.stack(self.values),
            "dones": torch.tensor(self.dones, dtype=torch.float32),
            "masks": torch.stack(self.masks),
        }


# ── GAE computation ───────────────────────────────────────────────────────────


def compute_gae(
    rewards: Tensor,
    values: Tensor,
    dones: Tensor,
    last_value: Tensor,
    gamma: float = GAMMA,
    lam: float = GAE_LAMBDA,
) -> tuple[Tensor, Tensor]:
    """
    Computes Generalized Advantage Estimation and value targets.

    Parameters
    ----------
    rewards    : (T,)  extrinsic + intrinsic rewards
    values     : (T,)  V(s_t) estimates from the critic
    dones      : (T,)  1.0 if episode ended at step t
    last_value : ()    V(s_T) bootstrap value for the last state
    gamma      : float discount factor
    lam        : float GAE lambda

    Returns
    -------
    advantages : (T,)  GAE advantages A_t
    returns    : (T,)  value targets V_targ = A_t + V(s_t)
    """
    T = len(rewards)
    advantages = torch.zeros(T)
    gae = 0.0

    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1]
        next_non_term = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * next_non_term - values[t]
        gae = delta + gamma * lam * next_non_term * gae
        advantages[t] = gae

    returns = advantages + values
    return advantages, returns


def _mini_batches(
    data: dict[str, Tensor], batch_size: int
) -> Iterator[dict[str, Tensor]]:
    """Yields random mini-batches from a dict of tensors."""
    T = data["obs"].shape[0]
    indices = torch.randperm(T)
    for start in range(0, T, batch_size):
        idx = indices[start : start + batch_size]
        yield {k: v[idx] for k, v in data.items()}


# ── PPO Agent ─────────────────────────────────────────────────────────────────


class PPOAgent:
    """
    PPO agent combining ActorCritic and ICM.

    Parameters
    ----------
    device     : torch.device
    lr_policy  : float  learning rate for ActorCritic
    lr_icm     : float  learning rate for ICM
    clip_eps   : float  PPO clipping epsilon
    vf_coef    : float  value loss coefficient
    ent_coef   : float  entropy bonus coefficient
    n_epochs   : int    update epochs per rollout
    batch_size : int    mini-batch size

    Usage
    -----
    agent = PPOAgent(device)

    # Collection loop
    obs, info = env.reset()
    for step in range(ROLLOUT_STEPS):
        action, log_prob, value = agent.select_action(obs, mask)
        obs_next, reward, done, _, info = env.step(action)
        agent.buffer.add(obs, obs_next, action, log_prob, reward, value, done, mask)
        obs = obs_next

    # Update
    metrics = agent.update(last_obs)
    """

    def __init__(
        self,
        device: torch.device,
        lr_policy: float = LR_POLICY,
        lr_icm: float = LR_ICM,
        clip_eps: float = CLIP_EPS,
        vf_coef: float = VF_COEF,
        ent_coef: float = ENT_COEF,
        n_epochs: int = N_EPOCHS,
        batch_size: int = MINI_BATCH,
    ) -> None:
        self.device = device
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        # Networks
        self.actor_critic = ActorCritic(device=device)
        self.icm = ICM(device=device)

        # Separate optimizers
        self.opt_policy = torch.optim.Adam(
            self.actor_critic.parameters(), lr=lr_policy, eps=1e-5
        )
        self.opt_icm = torch.optim.Adam(self.icm.parameters(), lr=lr_icm, eps=1e-5)

        # Rollout buffer
        self.buffer = RolloutBuffer(capacity=ROLLOUT_STEPS)

        # Running metrics (last update)
        self.metrics: dict[str, float] = {}

    # ── Action selection ──────────────────────────────────────────────────────

    @torch.no_grad()
    def select_action(
        self, obs: np.ndarray, mask: np.ndarray
    ) -> tuple[int, Tensor, Tensor]:
        """
        Samples an action from the masked policy.

        Parameters
        ----------
        obs  : np.ndarray (OBS_SIZE,)
        mask : np.ndarray (ACTION_SIZE,) dtype int8

        Returns
        -------
        action   : int
        log_prob : Tensor scalar
        value    : Tensor scalar
        """
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        mask_t = torch.tensor(mask, dtype=torch.bool).unsqueeze(0)

        dist = self.actor_critic.masked_distribution(obs_t, mask_t)
        action = dist.sample()

        log_prob = dist.log_prob(action)
        value = self.actor_critic.get_value(obs_t)

        return action.item(), log_prob.squeeze(), value.squeeze()

    # ── Update ────────────────────────────────────────────────────────────────

    def update(self, last_obs: np.ndarray) -> dict[str, float]:
        """
        Runs the full PPO + ICM update over the collected rollout.

        Parameters
        ----------
        last_obs : np.ndarray (OBS_SIZE,) — observation after last step,
                   used to bootstrap V(s_T) for GAE.

        Returns
        -------
        metrics : dict with loss components for TensorBoard logging
        """
        # ── 1. Compute intrinsic rewards ──────────────────────────────────────
        data = self.buffer.as_tensors()

        r_intrinsic = self.icm.intrinsic_reward(
            data["obs"], data["obs_next"], data["actions"]
        ).cpu()
        total_rewards = data["rewards"] + r_intrinsic

        # ── 2. Bootstrap last value ───────────────────────────────────────────
        with torch.no_grad():
            last_obs_t = torch.tensor(last_obs, dtype=torch.float32).unsqueeze(0)
            last_value = self.actor_critic.get_value(last_obs_t).cpu().squeeze()

        # ── 3. GAE ────────────────────────────────────────────────────────────
        advantages, returns = compute_gae(
            rewards=total_rewards,
            values=data["values"],
            dones=data["dones"],
            last_value=last_value,
        )

        # Normalise advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Augment data with computed targets
        data["advantages"] = advantages
        data["returns"] = returns

        # ── 4. Mini-batch PPO epochs ──────────────────────────────────────────
        metrics = {
            "loss/policy": 0.0,
            "loss/value": 0.0,
            "loss/entropy": 0.0,
            "loss/icm": 0.0,
            "loss/forward": 0.0,
            "loss/inverse": 0.0,
            "loss/total": 0.0,
            "misc/approx_kl": 0.0,
            "misc/clip_frac": 0.0,
        }
        n_updates = 0

        for _ in range(self.n_epochs):
            for batch in _mini_batches(data, self.batch_size):
                # Move batch to device
                obs_b = batch["obs"].to(self.device)
                actions_b = batch["actions"].to(self.device)
                old_lp_b = batch["log_probs"].to(self.device)
                adv_b = batch["advantages"].to(self.device)
                returns_b = batch["returns"].to(self.device)
                masks_b = batch["masks"].to(self.device).bool()
                obs_next_b = batch["obs_next"].to(self.device)

                # ── Policy loss ───────────────────────────────────────────────
                dist = self.actor_critic.masked_distribution(obs_b, masks_b)
                new_lp = dist.log_prob(actions_b)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_lp - old_lp_b)
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_b
                loss_policy = -torch.min(surr1, surr2).mean()

                # ── Value loss ────────────────────────────────────────────────
                _, values_pred = self.actor_critic(obs_b)
                loss_value = nn.functional.mse_loss(values_pred.squeeze(-1), returns_b)

                # ── PPO total loss ────────────────────────────────────────────
                loss_ppo = (
                    loss_policy + self.vf_coef * loss_value - self.ent_coef * entropy
                )

                self.opt_policy.zero_grad()
                loss_ppo.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), MAX_GRAD_NORM)
                self.opt_policy.step()

                # ── ICM loss ──────────────────────────────────────────────────
                loss_icm, loss_fwd, loss_inv = self.icm.loss(
                    obs_b, obs_next_b, actions_b
                )

                self.opt_icm.zero_grad()
                loss_icm.backward()
                nn.utils.clip_grad_norm_(self.icm.parameters(), MAX_GRAD_NORM)
                self.opt_icm.step()

                # ── Track metrics ─────────────────────────────────────────────
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - (new_lp - old_lp_b)).mean().item()
                    clip_frac = (
                        ((ratio - 1).abs() > self.clip_eps).float().mean().item()
                    )

                metrics["loss/policy"] += loss_policy.item()
                metrics["loss/value"] += loss_value.item()
                metrics["loss/entropy"] += entropy.item()
                metrics["loss/icm"] += loss_icm.item()
                metrics["loss/forward"] += loss_fwd.item()
                metrics["loss/inverse"] += loss_inv.item()
                metrics["loss/total"] += loss_ppo.item() + loss_icm.item()
                metrics["misc/approx_kl"] += approx_kl
                metrics["misc/clip_frac"] += clip_frac
                n_updates += 1

        # Average over all mini-batch updates
        self.metrics = {k: v / n_updates for k, v in metrics.items()}

        # Clear buffer for next rollout
        self.buffer.clear()

        return self.metrics

    # ── Checkpointing ─────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Saves all network weights and optimizer states."""
        torch.save(
            {
                "actor_critic": self.actor_critic.state_dict(),
                "icm": self.icm.state_dict(),
                "opt_policy": self.opt_policy.state_dict(),
                "opt_icm": self.opt_icm.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        """Loads all network weights and optimizer states."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint["actor_critic"])
        self.icm.load_state_dict(checkpoint["icm"])
        self.opt_policy.load_state_dict(checkpoint["opt_policy"])
        self.opt_icm.load_state_dict(checkpoint["opt_icm"])
