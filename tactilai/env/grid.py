"""
tactilai/env/grid.py

16x16 grid managing terrain layout, unit placement, movement validation,
reachable tile computation (Dijkstra), and combat resolution.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

from tactilai.env.terrain import (
    TerrainType,
    get_avoid_bonus,
    get_movement_cost,
    is_passable,
    terrain_display_char,
)
from tactilai.env.unit import CombatResult, DamageType, Team, Unit, UnitClass

# ── Constants ─────────────────────────────────────────────────────────────────

GRID_SIZE = 16
NUM_UNITS = 6  # units per team (one per class)
MAX_TURNS = 50  # episode ends in a draw after this many turns
MAX_MOVEMENT = 7  # Cavalier a le plus grand mouvement
MAX_RANGE = 3  # Archer a la plus grande portée


# ── Spawn zones ───────────────────────────────────────────────────────────────

# Blue spawns on the left side, Red on the right side
_BLUE_SPAWN_COLS = range(0, 3)
_RED_SPAWN_COLS = range(13, 16)


# ── Grid ──────────────────────────────────────────────────────────────────────


@dataclass
class Grid:
    """
    Main environment container.

    Attributes
    ----------
    size : int
        Width and height of the square grid (default 16).
    seed : int | None
        Random seed for reproducible terrain and spawn generation.
    terrain : np.ndarray
        2D array of shape (size, size) with TerrainType int values.
    units : list[Unit]
        All units (both teams) currently in the episode.
    turn : int
        Current turn number (incremented after both teams have acted).
    active_team : Team
        Which team is currently taking its turn.
    """

    size: int = GRID_SIZE
    seed: int | None = None
    terrain: np.ndarray = field(init=False)
    units: list[Unit] = field(init=False, default_factory=list)
    turn: int = field(init=False, default=0)
    active_team: Team = field(init=False, default=Team.BLUE)
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        self.terrain = self._generate_terrain()
        self.units = self._spawn_units()

    # ── Terrain generation ────────────────────────────────────────────────────

    def _generate_terrain(self) -> np.ndarray:
        """
        Generates a terrain map with weighted random placement.
        Weights: Plain 60%, Forest 25%, Mountain 15%.
        Spawn zones (cols 0-2 and 13-15) are always Plain for fair starts.
        """
        weights = [0.60, 0.25, 0.15]
        terrain_vals = [t.value for t in TerrainType]
        grid = np.array(
            self._rng.choices(terrain_vals, weights=weights, k=self.size * self.size),
            dtype=np.int8,
        ).reshape(self.size, self.size)

        # Force spawn zones to Plain
        grid[:, list(_BLUE_SPAWN_COLS)] = TerrainType.PLAIN.value
        grid[:, list(_RED_SPAWN_COLS)] = TerrainType.PLAIN.value

        return grid

    # ── Unit spawning ─────────────────────────────────────────────────────────

    def _spawn_units(self) -> list[Unit]:
        """
        Places one unit of each class per team on their respective spawn zones.
        Positions are randomly chosen without overlap.
        """
        units: list[Unit] = []
        uid = 0

        for team, col_range in (
            (Team.BLUE, _BLUE_SPAWN_COLS),
            (Team.RED, _RED_SPAWN_COLS),
        ):
            # Generate all candidate positions in the spawn zone
            candidates = [(row, col) for row in range(self.size) for col in col_range]
            self._rng.shuffle(candidates)

            for unit_class in UnitClass:
                pos = candidates.pop()
                units.append(Unit(uid=uid, unit_class=unit_class, team=team, pos=pos))
                uid += 1

        return units

    # ── Unit accessors ────────────────────────────────────────────────────────

    def get_unit_at(self, pos: tuple[int, int]) -> Unit | None:
        """Returns the alive unit at pos, or None."""
        for unit in self.units:
            if unit.is_alive and unit.pos == pos:
                return unit
        return None

    def alive_units(self, team: Team | None = None) -> list[Unit]:
        """Returns all alive units, optionally filtered by team."""
        return [
            u for u in self.units if u.is_alive and (team is None or u.team == team)
        ]

    def enemy_of(self, unit: Unit) -> list[Unit]:
        """Returns all alive enemies of a given unit."""
        return self.alive_units(Team(1 - unit.team.value))

    # ── Terrain helpers ───────────────────────────────────────────────────────

    def terrain_at(self, pos: tuple[int, int]) -> TerrainType:
        return TerrainType(self.terrain[pos[0], pos[1]])

    def in_bounds(self, pos: tuple[int, int]) -> bool:
        r, c = pos
        return 0 <= r < self.size and 0 <= c < self.size

    def _neighbors(self, pos: tuple[int, int]) -> Iterator[tuple[int, int]]:
        r, c = pos
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nb = (r + dr, c + dc)
            if self.in_bounds(nb):
                yield nb

    # ── Movement ─────────────────────────────────────────────────────────────

    def reachable_tiles(self, unit: Unit) -> set[tuple[int, int]]:
        """
        Returns all tiles a unit can move to this turn using Dijkstra.
        A tile is reachable if:
          - it is in bounds and passable
          - cumulative movement cost <= unit.movement
          - it is not occupied by another unit (friendly or enemy)
        """
        occupied = {u.pos for u in self.alive_units() if u is not unit}

        # Dijkstra: dist[pos] = minimum movement points spent to reach pos
        dist: dict[tuple[int, int], int] = {unit.pos: 0}
        queue: deque[tuple[int, int]] = deque([unit.pos])

        while queue:
            current = queue.popleft()
            current_cost = dist[current]

            for nb in self._neighbors(current):
                terrain = self.terrain_at(nb)
                if not is_passable(terrain):
                    continue
                step_cost = get_movement_cost(terrain, unit.unit_class)
                new_cost = current_cost + step_cost

                if new_cost <= unit.movement and new_cost < dist.get(nb, 999):
                    dist[nb] = new_cost
                    queue.append(nb)

        # Remove starting tile and occupied tiles (can't stop on them)
        reachable = set(dist.keys()) - {unit.pos} - occupied
        return reachable

    def move_unit(self, unit: Unit, target_pos: tuple[int, int]) -> bool:
        """
        Moves a unit to target_pos if valid.
        Returns True on success, False if the move is illegal.
        """
        if unit.has_moved:
            return False
        if target_pos not in self.reachable_tiles(unit):
            return False
        unit.pos = target_pos
        unit.has_moved = True
        return True

    # ── Combat resolution ─────────────────────────────────────────────────────

    def attackable_targets(self, unit: Unit) -> list[Unit]:
        """
        Returns all valid targets for the unit's action:
        - Enemies in range for offensive units
        - Allies in range (excluding self) for healers
        """
        if unit.weapon.damage_type == DamageType.HEAL:
            return [
                ally
                for ally in self.alive_units(unit.team)
                if ally is not unit and unit.in_range(ally.pos)
            ]
        return [enemy for enemy in self.enemy_of(unit) if unit.in_range(enemy.pos)]

    def resolve_combat(self, attacker: Unit, defender: Unit) -> list[CombatResult]:
        """
        Resolves a full combat exchange between attacker and defender.

        Fire Emblem rules:
          1. Attacker strikes first.
          2. Defender counter-attacks if defender is still alive and in range.
          3. If attacker SPD >= defender SPD + 4, attacker strikes again (double).
          4. Healer action always succeeds (no counter-attack).

        Returns a list of CombatResult, one per strike.
        """
        results: list[CombatResult] = []
        terrain_avoid = get_avoid_bonus(self.terrain_at(defender.pos))

        # ── Heal action ───────────────────────────────────────────────────────
        if attacker.weapon.damage_type == DamageType.HEAL:
            heal_amount = attacker.compute_damage(defender)
            defender.receive_heal(heal_amount)
            attacker.has_acted = True
            results.append(
                CombatResult(
                    hit=True,
                    critical=False,
                    damage=heal_amount,
                    is_heal=True,
                    attacker_id=attacker.uid,
                    defender_id=defender.uid,
                )
            )
            return results

        def _strike(atk: Unit, dfn: Unit, avoid: int) -> CombatResult:
            hit_roll = self._rng.randint(0, 99)
            hit_rate = atk.compute_hit_rate(dfn, terrain_avoid=avoid)
            did_hit = hit_roll < hit_rate

            crit_rate = atk.compute_crit_rate(dfn) if did_hit else 0
            crit_roll = self._rng.randint(0, 99)
            did_crit = did_hit and (crit_roll < crit_rate)

            damage = atk.compute_damage(dfn, critical=did_crit) if did_hit else 0
            if did_hit:
                dfn.take_damage(damage)

            return CombatResult(
                hit=did_hit,
                critical=did_crit,
                damage=damage,
                is_heal=False,
                attacker_id=atk.uid,
                defender_id=dfn.uid,
            )

        # 1. Attacker strikes
        results.append(_strike(attacker, defender, terrain_avoid))

        # 2. Counter-attack (if defender alive and in range)
        counter_avoid = get_avoid_bonus(self.terrain_at(attacker.pos))
        if defender.is_alive and defender.in_range(attacker.pos):
            results.append(_strike(defender, attacker, counter_avoid))

        # 3. Double attack (attacker)
        if attacker.is_alive and attacker.stats.spd >= defender.stats.spd + 4:
            results.append(_strike(attacker, defender, terrain_avoid))

        # 4. Double attack (defender counter, if still alive)
        if (
            defender.is_alive
            and defender.in_range(attacker.pos)
            and defender.stats.spd >= attacker.stats.spd + 4
        ):
            results.append(_strike(defender, attacker, counter_avoid))

        attacker.has_acted = True
        return results

    # ── Turn management ───────────────────────────────────────────────────────

    def end_turn(self) -> None:
        """
        Ends the current team's turn and switches to the other team.
        Increments the global turn counter after both teams have acted.
        """
        if self.active_team == Team.RED:
            self.turn += 1
        self.active_team = Team(1 - self.active_team.value)

        # Reset action flags for the newly active team
        for unit in self.alive_units(self.active_team):
            unit.reset_turn()

    # ── Terminal conditions ───────────────────────────────────────────────────

    @property
    def is_terminal(self) -> bool:
        return self.winner is not None or self.turn >= MAX_TURNS

    @property
    def winner(self) -> Team | None:
        """Returns the winning Team, or None if the game is still ongoing."""
        blue_alive = bool(self.alive_units(Team.BLUE))
        red_alive = bool(self.alive_units(Team.RED))
        if not blue_alive and not red_alive:
            return None  # simultaneous wipe → draw
        if not red_alive:
            return Team.BLUE
        if not blue_alive:
            return Team.RED
        return None

    # ── Observation ───────────────────────────────────────────────────────────

    def to_obs_array(self) -> np.ndarray:
        """
        Builds a flat float32 observation array for the RL agents.

        Layout:
          - terrain : (size * size,)  int values normalised to [0, 1]
          - units   : (NUM_UNITS * 2 * 10,)  — 10 features per unit, 0-padded if dead
          - meta    : [turn / MAX_TURNS, active_team]

        Total length: 256 + 120 + 2 = 378 features.
        """
        # Terrain (256,)
        terrain_obs = self.terrain.flatten().astype(np.float32) / len(TerrainType)

        # Units (120,) — fixed order: blue units first, then red, sorted by uid
        unit_obs: list[float] = []
        for team in (Team.BLUE, Team.RED):
            team_units = sorted(
                [u for u in self.units if u.team == team],
                key=lambda u: u.uid,
            )
            for u in team_units:
                if u.is_alive:
                    unit_obs.extend(u.to_obs_vector())
                else:
                    unit_obs.extend([0.0] * 10)  # dead unit → zero vector

        # Meta (2,)
        meta_obs = [self.turn / MAX_TURNS, float(self.active_team)]

        return np.array(terrain_obs.tolist() + unit_obs + meta_obs, dtype=np.float32)

    # ── Debug rendering ───────────────────────────────────────────────────────

    def render_ascii(self) -> str:
        """
        Returns a string representation of the grid for terminal debugging.
        Units are shown as their class initial + team letter (e.g. 'KB' = Knight Blue).
        """
        unit_map = {u.pos: u for u in self.alive_units()}
        rows: list[str] = []

        header = "   " + " ".join(f"{c:2}" for c in range(self.size))
        rows.append(header)

        for r in range(self.size):
            row_str = f"{r:2} "
            for c in range(self.size):
                pos = (r, c)
                if pos in unit_map:
                    u = unit_map[pos]
                    team_char = "B" if u.team == Team.BLUE else "R"
                    row_str += f"{u.unit_class.name[0]}{team_char} "
                else:
                    terrain_char = terrain_display_char(self.terrain_at(pos))
                    row_str += f" {terrain_char} "
            rows.append(row_str)

        rows.append(f"\nTurn {self.turn} | Active: {self.active_team.name}")
        return "\n".join(rows)
