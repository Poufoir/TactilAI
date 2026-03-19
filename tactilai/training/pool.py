"""
tactilai/training/pool.py

Checkpoint pool for adversarial self-play.

The pool maintains a fixed-size collection of past agent snapshots.
During training, the active agent plays against a randomly sampled
checkpoint from the pool, preventing cycling and improving robustness.

Pool policy
───────────
  - When full, the oldest checkpoint is evicted (FIFO).
  - A checkpoint is saved every SAVE_EVERY PPO updates.
  - The pool always keeps at least 1 checkpoint (the initial agent).
  - Sampling is uniform across all checkpoints in the pool.

Usage
─────
  pool = CheckpointPool(max_size=20, save_dir="checkpoints/")

  # Save current agent to pool
  pool.save_checkpoint(agent, update_step=100)

  # Load a random opponent
  opponent = PPOAgent(device)
  pool.load_random(opponent)
"""

from __future__ import annotations

import random
from pathlib import Path

from tactilai.agents.ppo import PPOAgent

# ── Hyperparameters ───────────────────────────────────────────────────────────

POOL_MAX_SIZE = 20  # maximum number of checkpoints in the pool
SAVE_EVERY = 50  # save a checkpoint every N PPO updates


# ── Checkpoint pool ───────────────────────────────────────────────────────────


class CheckpointPool:
    """
    Fixed-size FIFO pool of agent checkpoints for self-play.

    Parameters
    ----------
    max_size : int      maximum number of checkpoints stored
    save_dir : Path     directory where checkpoint files are written
    seed     : int | None  random seed for reproducible sampling
    """

    def __init__(
        self,
        max_size: int = POOL_MAX_SIZE,
        save_dir: str | Path = "checkpoints/pool",
        seed: int | None = None,
    ) -> None:
        self.max_size = max_size
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._rng = random.Random(seed)

        # Ordered list of checkpoint file paths (oldest first)
        self._pool: list[Path] = []

        # Load existing checkpoints from disk if resuming
        self._reload_from_disk()

    # ── Pool management ───────────────────────────────────────────────────────

    def save_checkpoint(self, agent: PPOAgent, update_step: int) -> Path:
        """
        Saves the agent's current weights to disk and adds to pool.
        Evicts the oldest checkpoint if pool is full.

        Parameters
        ----------
        agent       : PPOAgent   agent to snapshot
        update_step : int        current update step (used in filename)

        Returns
        -------
        path : Path   path to the saved checkpoint file
        """
        filename = self.save_dir / f"checkpoint_{update_step:06d}.pt"
        agent.save(str(filename))
        self._pool.append(filename)

        # Evict oldest if over capacity
        if len(self._pool) > self.max_size:
            oldest = self._pool.pop(0)
            if oldest.exists():
                oldest.unlink()

        return filename

    def load_random(self, agent: PPOAgent) -> Path:
        """
        Loads a randomly sampled checkpoint into the agent.

        Parameters
        ----------
        agent : PPOAgent   agent whose weights will be replaced

        Returns
        -------
        path : Path   path of the loaded checkpoint

        Raises
        ------
        RuntimeError if the pool is empty
        """
        if not self._pool:
            raise RuntimeError(
                "Checkpoint pool is empty. Save at least one checkpoint first."
            )
        path = self._rng.choice(self._pool)
        agent.load(str(path))
        return path

    def load_latest(self, agent: PPOAgent) -> Path:
        """Loads the most recent checkpoint into the agent."""
        if not self._pool:
            raise RuntimeError("Checkpoint pool is empty.")
        path = self._pool[-1]
        agent.load(str(path))
        return path

    def load_oldest(self, agent: PPOAgent) -> Path:
        """Loads the oldest checkpoint into the agent."""
        if not self._pool:
            raise RuntimeError("Checkpoint pool is empty.")
        path = self._pool[0]
        agent.load(str(path))
        return path

    # ── Pool state ────────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        return len(self._pool)

    @property
    def is_empty(self) -> bool:
        return len(self._pool) == 0

    @property
    def checkpoint_names(self) -> list[str]:
        """Returns checkpoint filenames (without dir) for ELO tracking."""
        return [p.stem for p in self._pool]

    def clear(self) -> None:
        """Removes all checkpoints from disk and empties the pool."""
        for path in self._pool:
            if path.exists():
                path.unlink()
        self._pool.clear()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _reload_from_disk(self) -> None:
        """
        Reloads existing checkpoints from save_dir on init.
        Useful when resuming a training run.
        """
        existing = sorted(self.save_dir.glob("checkpoint_*.pt"))
        # Keep only the most recent max_size checkpoints
        self._pool = existing[-self.max_size :]

    def __repr__(self) -> str:
        return f"CheckpointPool(size={self.size}/{self.max_size}, dir={self.save_dir})"
