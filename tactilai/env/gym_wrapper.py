"""
tactilai/env/gym_wrapper.py

Gymnasium wrapper around Grid with free-order, single Discrete action space.

Action encoding
───────────────
A single integer encodes (unit_id, move_idx, act_idx) via:

    action = unit_id * (N_MOVE * N_ACT) + move_idx * N_ACT + act_idx

Decoding:
    unit_id  = action // (N_MOVE * N_ACT)
    move_idx = (action % (N_MOVE * N_ACT)) // N_ACT
    act_idx  = action % N_ACT

Ranges:
    unit_id  ∈ [0, NUM_UNITS)   →  6  values
    move_idx ∈ [0, N_MOVE)      → 257 values  (0 = stay, 1..256 = tile index)
    act_idx  ∈ [0, N_ACT)       →  13 values  (0 = skip, 1..12 = target uid)

Total action space size: 6 × 257 × 13 = 20 046

Free-order rule
───────────────
On each step the agent picks any unit that is still ready (not yet moved
AND not yet acted this turn). Once a unit has both moved and acted it is
removed from the available set. When all units are done the env calls
end_turn() automatically and resets the ready set for the next team.

Observation (388 float32)
─────────────────────────
Same as before: grid_obs (378) + active_unit_obs (10).
"Active unit" here means the unit whose uid was selected in the last
decoded action. On reset it defaults to the first alive unit.

Reward shaping
──────────────
  +dmg_dealt / target_max_hp     offensive
  +heal_given / ally_max_hp      support
  -dmg_taken / self_max_hp       damage taken penalty
  +0.5                           kill bonus
  +1.0 / -1.0                    win / loss at episode end
  -0.005 per step                time penalty
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from tactilai.env.grid import GRID_SIZE, MAX_TURNS, NUM_UNITS, Grid
from tactilai.env.unit import DamageType, Team, Unit

# ── Action space constants ────────────────────────────────────────────────────

N_MOVE = GRID_SIZE * GRID_SIZE + 1  # 257  (0=stay, 1..256=tile)
N_ACT = NUM_UNITS * 2 + 1  # 13   (0=skip, 1..12=target uid)
ACTION_SIZE = NUM_UNITS * N_MOVE * N_ACT  # 6 × 257 × 13 = 20 046

# ── Observation space constants ───────────────────────────────────────────────

GRID_OBS_SIZE = GRID_SIZE * GRID_SIZE + NUM_UNITS * 2 * 10 + 2  # 378
UNIT_OBS_SIZE = 10
OBS_SIZE = GRID_OBS_SIZE + UNIT_OBS_SIZE  # 388


# ── Encoding / decoding ───────────────────────────────────────────────────────


def encode_action(unit_id: int, move_idx: int, act_idx: int) -> int:
    return unit_id * (N_MOVE * N_ACT) + move_idx * N_ACT + act_idx


def decode_action(action: int) -> tuple[int, int, int]:
    unit_id = action // (N_MOVE * N_ACT)
    reminder = action % (N_MOVE * N_ACT)
    move_idx = reminder // N_ACT
    act_idx = reminder % N_ACT
    return unit_id, move_idx, act_idx


def _pos_to_idx(pos: tuple[int, int], size: int = GRID_SIZE) -> int:
    return pos[0] * size + pos[1]


def _idx_to_pos(idx: int, size: int = GRID_SIZE) -> tuple[int, int]:
    return divmod(idx, size)


# ── Wrapper ───────────────────────────────────────────────────────────────────


class TactilAIEnv(gym.Env):
    """
    Gymnasium environment for TactilAI with free-order unit selection.

    Each step() call encodes a full (unit, move, act) triple.
    The agent can select any unit that has not yet acted this turn.
    The turn ends automatically when all units have acted.

    Parameters
    ----------
    team : Team
        Which team this instance controls.
    seed : int | None
        Passed to Grid for reproducible episodes.
    """

    metadata = {"render_modes": ["ascii"], "render_fps": 4}

    def __init__(self, team: Team = Team.BLUE, seed: int | None = None) -> None:
        super().__init__()
        self.controlled_team = team
        self._seed = seed
        self._grid: Grid = Grid(seed=seed)

        # ── Spaces ────────────────────────────────────────────────────────────
        self.action_space = spaces.Discrete(ACTION_SIZE)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(OBS_SIZE,),
            dtype=np.float32,
        )

        # ── Internal state ────────────────────────────────────────────────────
        self._last_active_unit: Unit | None = None
        self._episode_reward: float = 0.0
        self._ready_uids: set[int] = set()  # uids not yet done this turn
        self._init_ready_set()

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._seed = seed if seed is not None else self._seed
        self._grid = Grid(seed=self._seed)
        self._episode_reward = 0.0
        self._last_active_unit = None
        self._init_ready_set()
        return self._get_obs(), self._get_info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Decode and apply a (unit_id, move_idx, act_idx) action.

        Illegal sub-actions are handled gracefully:
          - unit not ready        → treated as no-op (stay + skip)
          - move tile unreachable → stay in place
          - target invalid        → skip action
        """
        reward = -0.005
        terminated = False
        truncated = False
        grid = self._grid

        # ── Decode ────────────────────────────────────────────────────────────
        raw_uid, move_idx, act_idx = decode_action(action)

        # Resolve unit from uid (raw_uid is an index into alive team units)
        team_units = sorted(grid.alive_units(grid.active_team), key=lambda u: u.uid)
        unit: Unit | None = None
        if raw_uid < len(team_units):
            candidate = team_units[raw_uid]
            if candidate.uid in self._ready_uids:
                unit = candidate

        # Fallback: pick first ready unit if selected unit is unavailable
        if unit is None:
            ready = [u for u in team_units if u.uid in self._ready_uids]
            unit = ready[0] if ready else None

        if unit is None:
            # No units left — should not happen if action_mask is respected
            terminated = grid.is_terminal
            return self._get_obs(), reward, terminated, False, self._get_info()

        self._last_active_unit = unit

        # ── Move sub-action ───────────────────────────────────────────────────
        if move_idx == 0:
            unit.has_moved = True  # explicit stay
        else:
            target_pos = _idx_to_pos(move_idx - 1)
            if not grid.move_unit(unit, target_pos):
                unit.has_moved = True  # illegal tile → stay

        # ── Act sub-action ────────────────────────────────────────────────────
        if act_idx == 0:
            unit.has_acted = True  # explicit skip
        else:
            target_uid = act_idx - 1
            target = next((u for u in grid.alive_units() if u.uid == target_uid), None)
            if target is not None and unit.in_range(target.pos):
                hp_before_target = target.current_hp
                hp_before_self = unit.current_hp
                results = grid.resolve_combat(unit, target)
                reward += self._compute_reward(
                    unit, target, hp_before_self, hp_before_target
                )
            else:
                unit.has_acted = True  # invalid target → skip

        # ── Update ready set ──────────────────────────────────────────────────
        if unit.has_moved and unit.has_acted:
            self._ready_uids.discard(unit.uid)

        # ── Auto end-turn when all units done ─────────────────────────────────
        if not self._ready_uids:
            grid.end_turn()
            self._init_ready_set()

        # ── Terminal check ────────────────────────────────────────────────────
        if grid.is_terminal:
            terminated = True
            reward += self._terminal_reward()
        elif grid.turn >= MAX_TURNS:
            truncated = True

        self._episode_reward += reward
        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self, mode: str = "ascii") -> None:
        print(self._grid.render_ascii())

    def close(self) -> None:
        pass

    # ── Action mask ───────────────────────────────────────────────────────────

    def action_mask(self) -> np.ndarray:
        """
        Boolean mask of shape (ACTION_SIZE,) indicating legal actions.

        A full (unit_id, move_idx, act_idx) triple is legal iff:
          - unit is alive, belongs to active team, and is in _ready_uids
          - move_idx is 0 (stay) OR the tile is reachable
          - act_idx is 0 (skip) OR the target is valid and in range
        """
        mask = np.zeros(ACTION_SIZE, dtype=np.int8)
        grid = self._grid
        team_units = sorted(grid.alive_units(grid.active_team), key=lambda u: u.uid)

        for raw_uid, unit in enumerate(team_units):
            if unit.uid not in self._ready_uids:
                continue

            # Legal move indices for this unit
            legal_moves: set[int] = {0}  # stay always legal
            if not unit.has_moved:
                for tile in grid.reachable_tiles(unit):
                    idx = _pos_to_idx(tile) + 1
                    if idx < N_MOVE:
                        legal_moves.add(idx)

            # Legal act indices for this unit
            legal_acts: set[int] = {0}  # skip always legal
            if not unit.has_acted:
                for target in grid.attackable_targets(unit):
                    a_idx = target.uid + 1
                    if a_idx < N_ACT:
                        legal_acts.add(a_idx)

            # Set mask for all legal (move, act) combinations
            for m in legal_moves:
                for a in legal_acts:
                    mask[encode_action(raw_uid, m, a)] = True

        return mask

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _init_ready_set(self) -> None:
        """Populates _ready_uids with all alive units of the active team."""
        self._ready_uids = {
            u.uid for u in self._grid.alive_units(self._grid.active_team)
        }
        # Default last active unit to first alive unit
        alive = self._grid.alive_units(self._grid.active_team)
        self._last_active_unit = alive[0] if alive else None

    def _get_obs(self) -> np.ndarray:
        grid_obs = self._grid.to_obs_array()
        unit_obs = (
            np.array(self._last_active_unit.to_obs_vector(), dtype=np.float32)
            if self._last_active_unit is not None
            else np.zeros(UNIT_OBS_SIZE, dtype=np.float32)
        )
        return np.concatenate([grid_obs, unit_obs])

    def _get_info(self) -> dict[str, Any]:
        ready_units = [
            u
            for u in self._grid.alive_units(self._grid.active_team)
            if u.uid in self._ready_uids
        ]
        return {
            "turn": self._grid.turn,
            "active_team": self._grid.active_team.name,
            "last_unit": repr(self._last_active_unit),
            "ready_units": [repr(u) for u in ready_units],
            "alive_blue": len(self._grid.alive_units(Team.BLUE)),
            "alive_red": len(self._grid.alive_units(Team.RED)),
            "winner": self._grid.winner.name if self._grid.winner is not None else None,
            "action_mask": self.action_mask(),
            "episode_reward": self._episode_reward,
        }

    def _compute_reward(
        self,
        unit: Unit,
        target: Unit,
        hp_before_self: int,
        hp_before_target: int,
    ) -> float:
        if unit.weapon.damage_type == DamageType.HEAL:
            heal_given = target.current_hp - hp_before_target
            return heal_given / target.stats.hp

        dmg_dealt = hp_before_target - target.current_hp
        dmg_taken = hp_before_self - unit.current_hp
        reward = dmg_dealt / target.stats.hp - dmg_taken / unit.stats.hp
        if not target.is_alive:
            reward += 0.5
        return reward

    def _terminal_reward(self) -> float:
        winner = self._grid.winner
        if winner is None:
            return 0.0
        return 1.0 if winner == self.controlled_team else -1.0
