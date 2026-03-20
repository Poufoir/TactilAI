"""
tactilai/training/selfplay.py

Main self-play training loop with curriculum learning.

Matchup selection
─────────────────
  At each update, the CurriculumScheduler decides independently for
  each agent whether it plays vs the heuristic bot or a pool checkpoint.

  Phase 1 (0→200)   : 100% vs bot   — learn basics
  Phase 2 (200→500) :  70% vs bot   — introduce self-play
  Phase 3 (500→1k)  :  40% vs bot   — self-play dominant
  Phase 4 (1k+)     :  10% vs bot   — mostly self-play

WandB logging
─────────────
  All metrics logged per agent (blue/, red/) and globally (eval/, elo/).
  Curriculum phase logged as curriculum/phase and curriculum/bot_prob.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import wandb

from tactilai.agents.ppo import PPOAgent, ROLLOUT_STEPS
from tactilai.env.gym_wrapper import TactilAIEnv
from tactilai.env.unit import Team
from tactilai.training.curriculum import CurriculumScheduler
from tactilai.training.elo import ELOTracker
from tactilai.training.heuristic_bot import HeuristicBot
from tactilai.training.pool import CheckpointPool, SAVE_EVERY


# ── Hyperparameters ───────────────────────────────────────────────────────────

TOTAL_UPDATES  = 2000
EVAL_EPISODES  = 20
EVAL_EVERY     = 50
CHECKPOINT_DIR = "checkpoints"
WANDB_PROJECT  = "tactilai"


# ── Self-play trainer ─────────────────────────────────────────────────────────

class SelfPlayTrainer:
    """
    Orchestrates two PPO agents with curriculum-based matchup selection.

    Parameters
    ----------
    device         : torch.device
    total_updates  : int
    seed           : int | None
    checkpoint_dir : str
    wandb_run_name : str | None
    wandb_run_id   : str | None   for resuming a WandB run
    """

    def __init__(
        self,
        device:         torch.device,
        total_updates:  int        = TOTAL_UPDATES,
        seed:           int | None = None,
        checkpoint_dir: str        = CHECKPOINT_DIR,
        wandb_run_name: str | None = None,
        wandb_run_id:   str | None = None,
    ) -> None:
        self.device        = device
        self.total_updates = total_updates
        self.seed          = seed

        # ── Agents ────────────────────────────────────────────────────────────
        self.agent_blue = PPOAgent(device=device)
        self.agent_red  = PPOAgent(device=device)

        # Pool opponents (frozen weights)
        self.opp_blue = PPOAgent(device=device)
        self.opp_red  = PPOAgent(device=device)

        # Heuristic bots
        self.bot_blue = HeuristicBot(team=Team.BLUE, seed=seed)
        self.bot_red  = HeuristicBot(team=Team.RED,  seed=seed)

        # ── Environment ───────────────────────────────────────────────────────
        self.env = TactilAIEnv(team=Team.BLUE, seed=seed)

        # ── Pool, ELO, Curriculum ─────────────────────────────────────────────
        self.pool_blue = CheckpointPool(
            save_dir=f"{checkpoint_dir}/blue", seed=seed
        )
        self.pool_red = CheckpointPool(
            save_dir=f"{checkpoint_dir}/red", seed=seed
        )
        self.elo = ELOTracker()
        self.elo.register("agent_blue")
        self.elo.register("agent_red")
        self.elo.register("heuristic_bot")

        self.curriculum = CurriculumScheduler(seed=seed)

        # Current matchup per agent ("self" or "vs_bot")
        self._matchup_blue = "vs_bot"
        self._matchup_red  = "vs_bot"

        # ── WandB ─────────────────────────────────────────────────────────────
        wandb.init(
            project = WANDB_PROJECT,
            name    = wandb_run_name,
            id      = wandb_run_id,
            resume  = "allow" if wandb_run_id else None,
            config  = {
                "total_updates":  total_updates,
                "rollout_steps":  ROLLOUT_STEPS,
                "eval_episodes":  EVAL_EPISODES,
                "seed":           seed,
                "device":         str(device),
                "curriculum":     True,
            },
        )

        # ── State ─────────────────────────────────────────────────────────────
        self.update  = 0
        self.episode = 0
        self._win_buffer: dict[str, list[float]] = {
            "blue": [], "red": [], "draw": []
        }

    # ── Main training loop ────────────────────────────────────────────────────

    def train(self) -> None:
        print(self.curriculum)
        print(f"\nStarting training on {self.device} | {self.total_updates} updates\n")

        obs, info = self.env.reset(seed=self.seed)
        t_start   = time.time()

        # Initial checkpoints
        self.pool_blue.save_checkpoint(self.agent_blue, update_step=0)
        self.pool_red.save_checkpoint(self.agent_red,   update_step=0)
        self._refresh_opponents()

        while self.update < self.total_updates:
            obs, info = self._collect_rollout(obs, info)

            for agent, name in (
                (self.agent_blue, "blue"),
                (self.agent_red,  "red"),
            ):
                if not agent.buffer.is_full:
                    continue

                metrics = agent.update(obs)
                self.update += 1

                # Decide next matchup for this agent
                matchup = self.curriculum.matchup(self.update)
                if name == "blue":
                    self._matchup_blue = matchup
                else:
                    self._matchup_red = matchup

                self._refresh_opponents()
                self._log_metrics(metrics, name, self.update)
                self._log_curriculum()

                # Save checkpoint
                if self.update % SAVE_EVERY == 0:
                    if name == "blue":
                        self.pool_blue.save_checkpoint(
                            self.agent_blue, self.update
                        )
                    else:
                        self.pool_red.save_checkpoint(
                            self.agent_red, self.update
                        )

                # Evaluation
                if self.update % EVAL_EVERY == 0:
                    win_rates = self._evaluate()
                    self._log_elo()
                    elapsed = time.time() - t_start
                    phase   = self.curriculum.phase(self.update)
                    print(
                        f"Update {self.update:5d}/{self.total_updates} | "
                        f"Phase {phase} | "
                        f"ELO blue={self.elo.rating('agent_blue'):.0f} "
                        f"red={self.elo.rating('agent_red'):.0f} | "
                        f"win_blue={win_rates['blue']:.0%} | "
                        f"elapsed={elapsed:.0f}s"
                    )

        wandb.finish()
        print("\nTraining complete.")

    # ── Rollout collection ────────────────────────────────────────────────────

    def _collect_rollout(
        self, obs: np.ndarray, info: dict
    ) -> tuple[np.ndarray, dict]:
        """
        One environment step. The active team determines which agent acts.
        If that agent's matchup is "vs_bot", the opponent acts via heuristic.
        """
        active_team = self.env._grid.active_team
        mask        = info["action_mask"]

        if active_team == Team.BLUE:
            acting_agent = self.agent_blue
            matchup      = self._matchup_blue
        else:
            acting_agent = self.agent_red
            matchup      = self._matchup_red

        action, log_prob, value = acting_agent.select_action(obs, mask)
        obs_next, reward, terminated, truncated, info_next = self.env.step(action)

        acting_agent.buffer.add(
            obs      = torch.tensor(obs,      dtype=torch.float32),
            obs_next = torch.tensor(obs_next, dtype=torch.float32),
            action   = torch.tensor(action,   dtype=torch.long),
            log_prob = log_prob.detach().cpu(),
            reward   = float(reward),
            value    = value.detach().cpu(),
            done     = terminated or truncated,
            mask     = torch.tensor(mask, dtype=torch.int8),
        )

        if terminated or truncated:
            self.episode += 1
            self._record_episode_result(info_next.get("winner"))
            obs_next, info_next = self.env.reset()

        return obs_next, info_next

    # ── Opponent refresh ──────────────────────────────────────────────────────

    def _refresh_opponents(self) -> None:
        """
        Loads the appropriate opponent based on current matchup.
        Bot matchups don't need a checkpoint — bot acts directly via grid.
        Pool matchups load a random checkpoint.
        """
        if self._matchup_blue == "self" and not self.pool_red.is_empty:
            self.pool_red.load_random(self.opp_red)
        if self._matchup_red == "self" and not self.pool_blue.is_empty:
            self.pool_blue.load_random(self.opp_blue)

    def _get_opponent_action(
        self, team: Team, obs: np.ndarray, mask: np.ndarray
    ) -> int:
        """Returns the opponent's action (bot or pool agent)."""
        matchup = self._matchup_blue if team == Team.BLUE else self._matchup_red
        if matchup == "vs_bot":
            bot = self.bot_blue if team == Team.BLUE else self.bot_red
            return bot.select_action(obs, mask, self.env._grid)
        else:
            opp = self.opp_blue if team == Team.BLUE else self.opp_red
            action, _, _ = opp.select_action(obs, mask)
            return action

    # ── Evaluation ───────────────────────────────────────────────────────────

    def _evaluate(self) -> dict[str, float]:
        """Runs EVAL_EPISODES between agent_blue and agent_red."""
        results  = {"blue": 0, "red": 0, "draw": 0}
        eval_env = TactilAIEnv(team=Team.BLUE, seed=self.seed)

        for ep in range(EVAL_EPISODES):
            obs, info = eval_env.reset(seed=ep)
            done      = False

            while not done:
                active_team = eval_env._grid.active_team
                mask        = info["action_mask"]
                agent = (
                    self.agent_blue
                    if active_team == Team.BLUE
                    else self.agent_red
                )
                action, _, _ = agent.select_action(obs, mask)
                obs, _, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated

            winner = info.get("winner")
            if winner == "BLUE":
                results["blue"] += 1
                self.elo.update("agent_blue", "agent_red", "agent_blue")
            elif winner == "RED":
                results["red"] += 1
                self.elo.update("agent_blue", "agent_red", "agent_red")
            else:
                results["draw"] += 1
                self.elo.update("agent_blue", "agent_red", None)

        eval_env.close()

        total     = EVAL_EPISODES
        win_rates = {k: v / total for k, v in results.items()}
        wandb.log({
            "eval/win_rate_blue": win_rates["blue"],
            "eval/win_rate_red":  win_rates["red"],
            "eval/draw_rate":     win_rates["draw"],
        }, step=self.update)

        return win_rates

    # ── Logging ───────────────────────────────────────────────────────────────

    def _record_episode_result(self, winner: str | None) -> None:
        self._win_buffer["blue"].append(1.0 if winner == "BLUE" else 0.0)
        self._win_buffer["red"].append( 1.0 if winner == "RED"  else 0.0)
        self._win_buffer["draw"].append(1.0 if winner is None   else 0.0)
        for k in self._win_buffer:
            if len(self._win_buffer[k]) > 100:
                self._win_buffer[k].pop(0)

    def _log_metrics(
        self, metrics: dict[str, float], agent_name: str, step: int
    ) -> None:
        log_data = {f"{agent_name}/{k}": v for k, v in metrics.items()}
        if self._win_buffer[agent_name]:
            log_data[f"{agent_name}/win_rate_rolling"] = float(
                np.mean(self._win_buffer[agent_name])
            )
        wandb.log(log_data, step=step)

    def _log_elo(self) -> None:
        wandb.log({
            "elo/blue": self.elo.rating("agent_blue"),
            "elo/red":  self.elo.rating("agent_red"),
        }, step=self.update)

    def _log_curriculum(self) -> None:
        wandb.log({
            "curriculum/phase":    self.curriculum.phase(self.update),
            "curriculum/bot_prob": self.curriculum.bot_probability(self.update),
            "curriculum/matchup_blue": 1 if self._matchup_blue == "vs_bot" else 0,
            "curriculum/matchup_red":  1 if self._matchup_red  == "vs_bot" else 0,
        }, step=self.update)