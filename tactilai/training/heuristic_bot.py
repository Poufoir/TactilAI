"""
tactilai/training/heuristic_bot.py

Aggressive heuristic bot for curriculum learning.

Strategy
────────
For each unit (in order of readiness):
  1. If an enemy is already in attack range → attack the one with lowest HP
  2. Otherwise → move toward the closest enemy, then attack if in range
  3. If healer → heal the ally with lowest HP ratio instead of attacking

The bot uses the same action encoding as PPOAgent (Discrete 20 046)
so it plugs directly into the existing SelfPlayTrainer infrastructure.

Usage
─────
  bot = HeuristicBot(team=Team.RED)
  action = bot.select_action(obs, mask, grid)
"""

from __future__ import annotations

import random

import numpy as np

from tactilai.env.grid import Grid
from tactilai.env.gym_wrapper import N_ACT, N_MOVE, encode_action
from tactilai.env.unit import DamageType, Team, Unit, UnitClass


class HeuristicBot:
    """
    Aggressive heuristic bot.

    Parameters
    ----------
    team : Team     which team the bot controls
    seed : int | None
    """

    def __init__(self, team: Team, seed: int | None = None) -> None:
        self.team = team
        self._rng = random.Random(seed)

    def select_action(
        self,
        obs:  np.ndarray,   # unused — bot uses grid directly
        mask: np.ndarray,   # np.int8 (ACTION_SIZE,)
        grid: Grid,
    ) -> int:
        """
        Selects a legal action using the aggressive heuristic.

        Parameters
        ----------
        obs  : np.ndarray   observation (unused by heuristic)
        mask : np.ndarray   legal action mask (int8)
        grid : Grid         current game state

        Returns
        -------
        action : int   encoded (unit_id, move_idx, act_idx)
        """
        team_units = sorted(
            [u for u in grid.alive_units(self.team) if u.is_ready],
            key=lambda u: u.uid,
        )

        for raw_uid, unit in enumerate(
            sorted(grid.alive_units(self.team), key=lambda u: u.uid)
        ):
            if not unit.is_ready:
                continue

            # ── Healer logic ──────────────────────────────────────────────────
            if unit.weapon.damage_type == DamageType.HEAL:
                action = self._healer_action(unit, raw_uid, grid, mask)
                if action is not None:
                    return action

            # ── Offensive logic ───────────────────────────────────────────────
            action = self._offensive_action(unit, raw_uid, grid, mask)
            if action is not None:
                return action

        # Fallback — pick any legal action
        legal = np.where(mask == 1)[0]
        return int(self._rng.choice(legal)) if len(legal) > 0 else 0

    # ── Healer logic ──────────────────────────────────────────────────────────

    def _healer_action(
        self, unit: Unit, raw_uid: int, grid: Grid, mask: np.ndarray
    ) -> int | None:
        """Heals the ally with the lowest HP ratio in range."""
        allies = [
            a for a in grid.alive_units(self.team)
            if a is not unit and a.hp_ratio < 1.0
        ]
        if not allies:
            return None

        # Sort by HP ratio ascending (most injured first)
        allies.sort(key=lambda a: a.hp_ratio)

        for ally in allies:
            # Try healing from current position
            if unit.in_range(ally.pos):
                act_idx = ally.uid + 1
                action  = encode_action(raw_uid, 0, act_idx)
                if mask[action] == 1:
                    return action

            # Try moving closer then healing
            move_action = self._move_toward(unit, ally.pos, raw_uid, grid, mask)
            if move_action is not None:
                move_idx = (move_action % (N_MOVE * N_ACT)) // N_ACT
                # After moving, check if in range
                new_pos = self._idx_to_pos(move_idx - 1) if move_idx > 0 else unit.pos
                if self._chebyshev(new_pos, ally.pos) <= unit.weapon.max_range:
                    act_idx = ally.uid + 1
                    action  = encode_action(raw_uid, move_idx, act_idx)
                    if mask[action] == 1:
                        return action
                # Move only (skip heal)
                action = encode_action(raw_uid, move_idx, 0)
                if mask[action] == 1:
                    return action

        return None

    # ── Offensive logic ───────────────────────────────────────────────────────

    def _offensive_action(
        self, unit: Unit, raw_uid: int, grid: Grid, mask: np.ndarray
    ) -> int | None:
        """
        Aggressive strategy:
          1. If enemy in range → attack lowest HP enemy
          2. Else → move toward closest enemy, attack if now in range
        """
        enemies = grid.enemy_of(unit)
        if not enemies:
            return None

        # 1. Attack from current position
        in_range = [e for e in enemies if unit.in_range(e.pos)]
        if in_range:
            target  = min(in_range, key=lambda e: e.current_hp)
            act_idx = target.uid + 1
            action  = encode_action(raw_uid, 0, act_idx)
            if mask[action] == 1:
                return action

        # 2. Move toward closest enemy
        closest = min(enemies, key=lambda e: self._chebyshev(unit.pos, e.pos))
        move_action = self._move_toward(unit, closest.pos, raw_uid, grid, mask)

        if move_action is not None:
            move_idx = (move_action % (N_MOVE * N_ACT)) // N_ACT
            new_pos  = self._idx_to_pos(move_idx - 1) if move_idx > 0 else unit.pos

            # Check if any enemy is now in range after moving
            in_range_after = [
                e for e in enemies
                if self._chebyshev(new_pos, e.pos) <= unit.weapon.max_range
                and self._chebyshev(new_pos, e.pos) >= unit.weapon.min_range
            ]
            if in_range_after:
                target  = min(in_range_after, key=lambda e: e.current_hp)
                act_idx = target.uid + 1
                action  = encode_action(raw_uid, move_idx, act_idx)
                if mask[action] == 1:
                    return action

            # Move only
            action = encode_action(raw_uid, move_idx, 0)
            if mask[action] == 1:
                return action

        # Skip — stay and pass
        action = encode_action(raw_uid, 0, 0)
        if mask[action] == 1:
            return action

        return None

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _move_toward(
        self,
        unit:    Unit,
        target:  tuple[int, int],
        raw_uid: int,
        grid:    Grid,
        mask:    np.ndarray,
    ) -> int | None:
        """
        Finds the reachable tile closest to target and returns
        the encoded action (move_idx, act=0).
        """
        reachable = grid.reachable_tiles(unit)
        if not reachable:
            return None

        best_tile = min(reachable, key=lambda t: self._chebyshev(t, target))
        move_idx  = self._pos_to_idx(best_tile) + 1

        action = encode_action(raw_uid, move_idx, 0)
        if mask[action] == 1:
            return action

        return None

    @staticmethod
    def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

    @staticmethod
    def _pos_to_idx(pos: tuple[int, int], size: int = 16) -> int:
        return pos[0] * size + pos[1]

    @staticmethod
    def _idx_to_pos(idx: int, size: int = 16) -> tuple[int, int]:
        return divmod(idx, size)