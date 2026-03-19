"""
tactilai/training/elo.py

ELO rating system for tracking agent performance across self-play episodes.

Formula
───────
  E_A = 1 / (1 + 10^((R_B - R_A) / 400))   expected score for A
  R_A += K * (S_A - E_A)                     updated rating for A

  S_A = 1.0  (win), 0.5  (draw), 0.0  (loss)

K-factor
────────
  K controls how fast ratings change per game.
  K=32  : fast adaptation (good for early training, high variance)
  K=16  : stable (good once agents have played 30+ games)
  K=8   : very stable (tournament mode)

  We use a dynamic K that decreases as n_games increases:
    K = max(K_MIN, K_START / (1 + n_games / K_DECAY))
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

# ── Hyperparameters ───────────────────────────────────────────────────────────

ELO_START = 1000  # initial rating for all agents
K_START = 32.0  # initial K-factor
K_MIN = 8.0  # minimum K-factor
K_DECAY = 100  # games played before K halves


# ── ELO tracker ───────────────────────────────────────────────────────────────


@dataclasses.dataclass
class ELOTracker:
    """
    Tracks ELO ratings for named agents (e.g. "agent_A", "checkpoint_003").

    Attributes
    ----------
    ratings  : dict[str, float]  current ELO rating per agent
    n_games  : dict[str, int]    number of games played per agent
    history  : list[dict]        full game log for analysis
    """

    ratings: dict[str, float] = dataclasses.field(default_factory=dict)
    n_games: dict[str, int] = dataclasses.field(default_factory=dict)
    history: list[dict] = dataclasses.field(default_factory=list)

    def register(self, name: str, rating: float = ELO_START) -> None:
        """Registers a new agent with an initial rating."""
        if name not in self.ratings:
            self.ratings[name] = rating
            self.n_games[name] = 0

    def _k_factor(self, name: str) -> float:
        """Dynamic K-factor that decreases as the agent plays more games."""
        n = self.n_games.get(name, 0)
        return max(K_MIN, K_START / (1.0 + n / K_DECAY))

    def expected_score(self, name_a: str, name_b: str) -> float:
        """
        Returns the expected score for agent A against agent B.
        E_A = 1 / (1 + 10^((R_B - R_A) / 400))
        """
        r_a = self.ratings[name_a]
        r_b = self.ratings[name_b]
        return 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / 400.0))

    def update(
        self,
        name_a: str,
        name_b: str,
        winner: str | None,  # name of winner, or None for draw
    ) -> tuple[float, float]:
        """
        Updates ELO ratings after a game between agent_a and agent_b.

        Parameters
        ----------
        name_a : str   name of agent A (blue team)
        name_b : str   name of agent B (red team)
        winner : str | None   name of winner, None = draw

        Returns
        -------
        delta_a : float   rating change for agent A
        delta_b : float   rating change for agent B
        """
        # Auto-register unknown agents
        self.register(name_a)
        self.register(name_b)

        # Scores
        if winner == name_a:
            s_a, s_b = 1.0, 0.0
        elif winner == name_b:
            s_a, s_b = 0.0, 1.0
        else:
            s_a, s_b = 0.5, 0.5  # draw

        e_a = self.expected_score(name_a, name_b)
        e_b = 1.0 - e_a

        k_a = self._k_factor(name_a)
        k_b = self._k_factor(name_b)

        delta_a = k_a * (s_a - e_a)
        delta_b = k_b * (s_b - e_b)

        self.ratings[name_a] += delta_a
        self.ratings[name_b] += delta_b
        self.n_games[name_a] += 1
        self.n_games[name_b] += 1

        # Log
        self.history.append(
            {
                "agent_a": name_a,
                "agent_b": name_b,
                "winner": winner,
                "delta_a": round(delta_a, 2),
                "delta_b": round(delta_b, 2),
                "elo_a": round(self.ratings[name_a], 1),
                "elo_b": round(self.ratings[name_b], 1),
            }
        )

        return delta_a, delta_b

    def rating(self, name: str) -> float:
        """Returns the current rating of an agent."""
        return self.ratings.get(name, ELO_START)

    def leaderboard(self) -> list[tuple[str, float, int]]:
        """
        Returns agents sorted by rating descending.
        Each entry: (name, rating, n_games)
        """
        return sorted(
            [(n, r, self.n_games[n]) for n, r in self.ratings.items()],
            key=lambda x: x[1],
            reverse=True,
        )

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Saves tracker state to a JSON file."""
        data = {
            "ratings": self.ratings,
            "n_games": self.n_games,
            "history": self.history,
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "ELOTracker":
        """Loads tracker state from a JSON file."""
        data = json.loads(Path(path).read_text())
        tracker = cls()
        tracker.ratings = data["ratings"]
        tracker.n_games = {k: int(v) for k, v in data["n_games"].items()}
        tracker.history = data["history"]
        return tracker

    def __repr__(self) -> str:
        board = self.leaderboard()
        lines = ["ELOTracker leaderboard:"]
        for name, rating, n in board:
            lines.append(f"  {name:20s}  {rating:7.1f}  ({n} games)")
        return "\n".join(lines)
