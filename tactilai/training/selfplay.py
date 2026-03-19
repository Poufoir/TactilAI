"""
tactilai/training/selfplay.py

Main self-play training loop for TactilAI.

Two independent PPO agents (BLUE and RED) are trained simultaneously.
Each agent collects rollouts against a randomly sampled opponent from
the checkpoint pool, then updates its own policy.

Training loop (per iteration)
──────────────────────────────
  1. Sample opponent from pool for each agent
  2. Collect ROLLOUT_STEPS steps per agent (alternating turns in env)
  3. PPO + ICM update for each agent
  4. Evaluate: run EVAL_EPISODES full episodes, record win rates
  5. Update ELO ratings
  6. Save checkpoints to pool every SAVE_EVERY updates
  7. Log all metrics to TensorBoard

Episode structure
─────────────────
  - One TactilAIEnv instance runs the game
  - Agent BLUE acts on BLUE turns, agent RED acts on RED turns
  - Both agents collect transitions simultaneously into their own buffers
  - When a buffer is full, that agent performs a PPO update immediately
    (the other agent's buffer may not be full yet — that's fine)

Self-play stability measures
────────────────────────────
  - Pool sampling prevents cycling
  - Opponent weights are frozen during collection (no_grad)
  - ELO tracks relative progress of both agents
  - Win rate window of last EVAL_EPISODES episodes
"""

from __future__ import annotations

import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from tactilai.agents.ppo import ROLLOUT_STEPS, PPOAgent
from tactilai.env.gym_wrapper import TactilAIEnv
from tactilai.env.unit import Team
from tactilai.training.elo import ELOTracker
from tactilai.training.pool import SAVE_EVERY, CheckpointPool

# ── Hyperparameters ───────────────────────────────────────────────────────────

TOTAL_UPDATES = 2000  # total PPO updates per agent
EVAL_EPISODES = 20  # episodes per evaluation
EVAL_EVERY = 50  # evaluate every N updates
LOG_DIR = "runs/selfplay"
CHECKPOINT_DIR = "checkpoints"


# ── Self-play trainer ─────────────────────────────────────────────────────────


class SelfPlayTrainer:
    """
    Orchestrates two PPO agents training via adversarial self-play.

    Parameters
    ----------
    device      : torch.device
    total_updates : int    total PPO updates to run
    seed        : int | None
    log_dir     : str      TensorBoard log directory
    checkpoint_dir : str   directory for checkpoints and pool
    """

    def __init__(
        self,
        device: torch.device,
        total_updates: int = TOTAL_UPDATES,
        seed: int | None = None,
        log_dir: str = LOG_DIR,
        checkpoint_dir: str = CHECKPOINT_DIR,
    ) -> None:
        self.device = device
        self.total_updates = total_updates
        self.seed = seed

        # ── Agents ────────────────────────────────────────────────────────────
        self.agent_blue = PPOAgent(device=device)
        self.agent_red = PPOAgent(device=device)

        # Frozen opponent agents (weights updated from pool, never trained)
        self.opp_blue = PPOAgent(device=device)
        self.opp_red = PPOAgent(device=device)

        # ── Environment ───────────────────────────────────────────────────────
        self.env = TactilAIEnv(team=Team.BLUE, seed=seed)

        # ── Pool & ELO ────────────────────────────────────────────────────────
        self.pool_blue = CheckpointPool(save_dir=f"{checkpoint_dir}/blue", seed=seed)
        self.pool_red = CheckpointPool(save_dir=f"{checkpoint_dir}/red", seed=seed)
        self.elo = ELOTracker()
        self.elo.register("agent_blue")
        self.elo.register("agent_red")

        # ── Logging ───────────────────────────────────────────────────────────
        self.writer = SummaryWriter(log_dir=log_dir)
        self.update = 0
        self.episode = 0

        # Running stats
        self._win_buffer: dict[str, list[float]] = {"blue": [], "red": [], "draw": []}

    # ── Main training loop ────────────────────────────────────────────────────

    def train(self) -> None:
        """Runs the full self-play training loop."""
        print(f"Starting self-play training on {self.device}")
        print(f"Total updates : {self.total_updates}")
        print(f"Rollout steps : {ROLLOUT_STEPS} per agent\n")

        obs, info = self.env.reset(seed=self.seed)
        t_start = time.time()

        # Save initial checkpoints to pool so sampling works from step 0
        self.pool_blue.save_checkpoint(self.agent_blue, update_step=0)
        self.pool_red.save_checkpoint(self.agent_red, update_step=0)

        # Load initial opponents
        self.pool_blue.load_random(self.opp_blue)
        self.pool_red.load_random(self.opp_red)

        while self.update < self.total_updates:
            obs, info = self._collect_rollout(obs, info)

            # ── Update whichever agent has a full buffer ───────────────────
            for agent, opp_pool, name in (
                (self.agent_blue, self.pool_red, "blue"),
                (self.agent_red, self.pool_blue, "red"),
            ):
                if agent.buffer.is_full:
                    metrics = agent.update(obs)
                    self.update += 1
                    self._log_metrics(metrics, name, self.update)

                    # Refresh opponent from pool
                    if name == "blue":
                        self.pool_red.load_random(self.opp_red)
                    else:
                        self.pool_blue.load_random(self.opp_blue)

                    # Save checkpoint to pool
                    if self.update % SAVE_EVERY == 0:
                        if name == "blue":
                            self.pool_blue.save_checkpoint(self.agent_blue, self.update)
                        else:
                            self.pool_red.save_checkpoint(self.agent_red, self.update)

                    # Evaluation
                    if self.update % EVAL_EVERY == 0:
                        self._evaluate()
                        self._log_elo()
                        elapsed = time.time() - t_start
                        print(
                            f"Update {self.update:5d}/{self.total_updates} | "
                            f"ELO blue={self.elo.rating('agent_blue'):.0f} "
                            f"red={self.elo.rating('agent_red'):.0f} | "
                            f"elapsed={elapsed:.0f}s"
                        )

        self.writer.close()
        print("\nTraining complete.")

    # ── Rollout collection ────────────────────────────────────────────────────

    def _collect_rollout(
        self,
        obs: np.ndarray,
        info: dict,
    ) -> tuple[np.ndarray, dict]:
        """
        Collects one step in the environment and stores the transition
        in the appropriate agent's buffer.

        The active team determines which agent acts:
          - BLUE turn → agent_blue acts, opp_red is frozen
          - RED turn  → agent_red acts, opp_blue is frozen

        Returns updated obs and info.
        """
        active_team = self.env._grid.active_team
        mask = info["action_mask"]

        if active_team == Team.BLUE:
            acting_agent = self.agent_blue
        else:
            acting_agent = self.agent_red

        action, log_prob, value = acting_agent.select_action(obs, mask)
        obs_next, reward, terminated, truncated, info_next = self.env.step(action)

        acting_agent.buffer.add(
            obs=torch.tensor(obs, dtype=torch.float32),
            obs_next=torch.tensor(obs_next, dtype=torch.float32),
            action=torch.tensor(action, dtype=torch.long),
            log_prob=log_prob.detach().cpu(),
            reward=float(reward),
            value=value.detach().cpu(),
            done=terminated or truncated,
            mask=torch.tensor(mask, dtype=torch.int8),
        )

        if terminated or truncated:
            self.episode += 1
            winner = info_next.get("winner")
            self._record_episode_result(winner)
            obs_next, info_next = self.env.reset()

        return obs_next, info_next

    # ── Evaluation ───────────────────────────────────────────────────────────

    def _evaluate(self) -> dict[str, float]:
        """
        Runs EVAL_EPISODES full episodes between agent_blue and agent_red.
        Records win rates and updates ELO.
        """
        results = {"blue": 0, "red": 0, "draw": 0}
        eval_env = TactilAIEnv(team=Team.BLUE, seed=self.seed)

        for ep in range(EVAL_EPISODES):
            obs, info = eval_env.reset(seed=ep)
            done = False

            while not done:
                active_team = eval_env._grid.active_team
                mask = info["action_mask"]
                agent = self.agent_blue if active_team == Team.BLUE else self.agent_red
                action, _, _ = agent.select_action(obs, mask)
                obs, _, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated

            winner = info.get("winner")
            if winner == "BLUE":
                results["blue"] += 1
                self.elo.update("agent_blue", "agent_red", winner="agent_blue")
            elif winner == "RED":
                results["red"] += 1
                self.elo.update("agent_blue", "agent_red", winner="agent_red")
            else:
                results["draw"] += 1
                self.elo.update("agent_blue", "agent_red", winner=None)

        eval_env.close()

        # Log win rates
        total = EVAL_EPISODES
        win_rates = {k: v / total for k, v in results.items()}
        self.writer.add_scalar("eval/win_rate_blue", win_rates["blue"], self.update)
        self.writer.add_scalar("eval/win_rate_red", win_rates["red"], self.update)
        self.writer.add_scalar("eval/draw_rate", win_rates["draw"], self.update)
        return win_rates

    # ── Logging helpers ───────────────────────────────────────────────────────

    def _record_episode_result(self, winner: str | None) -> None:
        """Records episode outcome to running win buffer."""
        self._win_buffer["blue"].append(1.0 if winner == "BLUE" else 0.0)
        self._win_buffer["red"].append(1.0 if winner == "RED" else 0.0)
        self._win_buffer["draw"].append(1.0 if winner is None else 0.0)
        # Keep last 100 episodes
        for k in self._win_buffer:
            if len(self._win_buffer[k]) > 100:
                self._win_buffer[k].pop(0)

    def _log_metrics(
        self, metrics: dict[str, float], agent_name: str, step: int
    ) -> None:
        for key, value in metrics.items():
            self.writer.add_scalar(f"{agent_name}/{key}", value, step)

        # Rolling win rate
        if self._win_buffer["blue"]:
            self.writer.add_scalar(
                f"{agent_name}/win_rate_rolling",
                float(np.mean(self._win_buffer[agent_name])),
                step,
            )

    def _log_elo(self) -> None:
        self.writer.add_scalar("elo/blue", self.elo.rating("agent_blue"), self.update)
        self.writer.add_scalar("elo/red", self.elo.rating("agent_red"), self.update)
