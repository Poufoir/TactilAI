"""
tactilai/training/curriculum.py

Progressive curriculum scheduler for self-play training.

Schedule
────────
  Phase 1 : updates   0 → 200   100% vs bot
  Phase 2 : updates 200 → 500    70% vs bot,  30% self-play
  Phase 3 : updates 500 → 1000   40% vs bot,  60% self-play
  Phase 4 : updates 1000+        10% vs bot,  90% self-play

Matchup types
─────────────
  "self"     : agent vs agent (from pool)
  "vs_bot"   : agent vs heuristic bot

The scheduler is stateless — call matchup() at each update step
and it returns the matchup type based on the current update count.
"""

from __future__ import annotations

import random


# ── Schedule definition ───────────────────────────────────────────────────────

# List of (max_update, bot_probability) phases
# bot_probability = P(agent plays vs bot this update)
CURRICULUM_SCHEDULE: list[tuple[int, float]] = [
    (200,  1.00),   # Phase 1 — 100% vs bot
    (500,  0.70),   # Phase 2 — 70% vs bot
    (1000, 0.40),   # Phase 3 — 40% vs bot
    (9999, 0.10),   # Phase 4 — 10% vs bot
]


class CurriculumScheduler:
    """
    Decides the matchup type for each training update.

    Parameters
    ----------
    schedule : list of (max_update, bot_prob) phases
    seed     : int | None
    """

    def __init__(
        self,
        schedule: list[tuple[int, float]] = CURRICULUM_SCHEDULE,
        seed:     int | None = None,
    ) -> None:
        self.schedule = schedule
        self._rng     = random.Random(seed)

    def bot_probability(self, update: int) -> float:
        """Returns the probability of playing vs bot at this update step."""
        for max_update, prob in self.schedule:
            if update < max_update:
                return prob
        return self.schedule[-1][1]

    def matchup(self, update: int) -> str:
        """
        Returns the matchup type for this update.

        Returns
        -------
        "vs_bot"  : agent plays against heuristic bot
        "self"    : agent plays against pool checkpoint
        """
        prob = self.bot_probability(update)
        return "vs_bot" if self._rng.random() < prob else "self"

    def phase(self, update: int) -> int:
        """Returns the current curriculum phase (1-4)."""
        for i, (max_update, _) in enumerate(self.schedule):
            if update < max_update:
                return i + 1
        return len(self.schedule)

    def __repr__(self) -> str:
        lines = ["CurriculumScheduler:"]
        for i, (max_upd, prob) in enumerate(self.schedule):
            lines.append(
                f"  Phase {i+1}: updates <{max_upd:5d} → "
                f"{prob*100:.0f}% vs bot, {(1-prob)*100:.0f}% self-play"
            )
        return "\n".join(lines)