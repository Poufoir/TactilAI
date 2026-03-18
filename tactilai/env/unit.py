"""
forge_rl/env/unit.py

Defines unit classes, base stats, and the Fire Emblem-style combat system.
Combat formula reference:
  - Hit%    = weapon_hit + (SKL * 2) + LCK - (enemy_avoid)
  - Avoid%  = (SPD * 2) + LCK + terrain_avoid
  - Crit%   = weapon_crit + (SKL // 2) - (enemy_lck)
  - Damage  = ATK - enemy_DEF  (physical) | ATK - enemy_RES  (magical)
  - ATK     = STR + weapon_might  (physical) | MAG + weapon_might  (magical)
"""

from __future__ import annotations

import dataclasses
from enum import IntEnum, auto
from typing import NamedTuple

# ── Enumerations ──────────────────────────────────────────────────────────────


class UnitClass(IntEnum):
    KNIGHT = 0
    ARCHER = 1
    MAGE = 2
    HEALER = 3
    CAVALIER = 4
    BARBARIAN = 5


class DamageType(IntEnum):
    PHYSICAL = auto()
    MAGICAL = auto()
    HEAL = auto()


class Team(IntEnum):
    BLUE = 0
    RED = 1


# ── Weapon profile ────────────────────────────────────────────────────────────


class WeaponProfile(NamedTuple):
    """Stat bonuses provided by the unit's default weapon."""

    might: int  # base damage added to STR or MAG
    hit: int  # base hit rate contribution
    crit: int  # base crit rate contribution
    min_range: int
    max_range: int
    damage_type: DamageType


# Default weapon profiles per class
_WEAPON_PROFILES: dict[UnitClass, WeaponProfile] = {
    UnitClass.KNIGHT: WeaponProfile(
        might=8,
        hit=80,
        crit=0,
        min_range=1,
        max_range=1,
        damage_type=DamageType.PHYSICAL,
    ),
    UnitClass.ARCHER: WeaponProfile(
        might=6,
        hit=85,
        crit=5,
        min_range=2,
        max_range=3,
        damage_type=DamageType.PHYSICAL,
    ),
    UnitClass.MAGE: WeaponProfile(
        might=7,
        hit=80,
        crit=5,
        min_range=1,
        max_range=2,
        damage_type=DamageType.MAGICAL,
    ),
    UnitClass.HEALER: WeaponProfile(
        might=10, hit=100, crit=0, min_range=1, max_range=1, damage_type=DamageType.HEAL
    ),
    UnitClass.CAVALIER: WeaponProfile(
        might=7,
        hit=80,
        crit=5,
        min_range=1,
        max_range=1,
        damage_type=DamageType.PHYSICAL,
    ),
    UnitClass.BARBARIAN: WeaponProfile(
        might=10,
        hit=70,
        crit=5,
        min_range=1,
        max_range=1,
        damage_type=DamageType.PHYSICAL,
    ),
}


# ── Base stats per class ──────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class BaseStats:
    hp: int  # max hit points
    str: int  # physical attack
    mag: int  # magical attack
    skl: int  # skill  → hit rate, crit rate
    spd: int  # speed  → avoid, double attack
    lck: int  # luck   → hit, avoid, crit-avoid
    defense: (
        int  # physical damage reduction  (named 'defense' to avoid shadowing builtins)
    )
    res: int  # magical damage reduction


# Balanced for a 16x16 grid with ~4 units per team
_BASE_STATS: dict[UnitClass, BaseStats] = {
    UnitClass.KNIGHT: BaseStats(
        hp=40, str=12, mag=0, skl=7, spd=4, lck=4, defense=14, res=4
    ),
    UnitClass.ARCHER: BaseStats(
        hp=28, str=10, mag=0, skl=12, spd=9, lck=7, defense=6, res=5
    ),
    UnitClass.MAGE: BaseStats(
        hp=24, str=0, mag=13, skl=10, spd=8, lck=6, defense=3, res=12
    ),
    UnitClass.HEALER: BaseStats(
        hp=22, str=0, mag=10, skl=8, spd=7, lck=8, defense=2, res=10
    ),
    UnitClass.CAVALIER: BaseStats(
        hp=32, str=11, mag=0, skl=9, spd=10, lck=6, defense=8, res=5
    ),
    UnitClass.BARBARIAN: BaseStats(
        hp=36, str=14, mag=0, skl=6, spd=6, lck=3, defense=7, res=10
    ),
}

# Movement range per class (number of tiles per turn)
MOVEMENT_RANGE: dict[UnitClass, int] = {
    UnitClass.KNIGHT: 4,
    UnitClass.ARCHER: 5,
    UnitClass.MAGE: 5,
    UnitClass.HEALER: 5,
    UnitClass.CAVALIER: 7,
    UnitClass.BARBARIAN: 5,
}


# ── Combat result ─────────────────────────────────────────────────────────────


@dataclasses.dataclass
class CombatResult:
    """Outcome of a single combat exchange (attacker → defender)."""

    hit: bool
    critical: bool
    damage: int  # 0 if miss
    is_heal: bool  # True when healer targets an ally
    attacker_id: int
    defender_id: int

    def __repr__(self) -> str:
        if self.is_heal:
            return f"CombatResult(heal={self.damage}, target={self.defender_id})"
        status = "CRIT" if self.critical else ("HIT" if self.hit else "MISS")
        return f"CombatResult({status}, dmg={self.damage}, {self.attacker_id}→{self.defender_id})"


# ── Unit ──────────────────────────────────────────────────────────────────────


@dataclasses.dataclass
class Unit:
    """
    A single combatant on the grid.

    Attributes
    ----------
    uid : int
        Unique identifier within the episode.
    unit_class : UnitClass
        Determines base stats, weapon profile, and movement range.
    team : Team
        BLUE (agent A) or RED (agent B).
    pos : tuple[int, int]
        Current (row, col) position on the grid.
    """

    uid: int
    unit_class: UnitClass
    team: Team
    pos: tuple[int, int]

    # Runtime state (mutable)
    current_hp: int = dataclasses.field(init=False)
    has_moved: bool = dataclasses.field(default=False, init=False)
    has_acted: bool = dataclasses.field(default=False, init=False)

    def __post_init__(self) -> None:
        self.current_hp = self.stats.hp

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def stats(self) -> BaseStats:
        return _BASE_STATS[self.unit_class]

    @property
    def weapon(self) -> WeaponProfile:
        return _WEAPON_PROFILES[self.unit_class]

    @property
    def movement(self) -> int:
        return MOVEMENT_RANGE[self.unit_class]

    @property
    def is_alive(self) -> bool:
        return self.current_hp > 0

    @property
    def is_ready(self) -> bool:
        """True if the unit has neither moved nor acted this turn."""
        return not self.has_moved and not self.has_acted

    @property
    def hp_ratio(self) -> float:
        return self.current_hp / self.stats.hp

    # ── Turn management ───────────────────────────────────────────────────────

    def reset_turn(self) -> None:
        """Called at the start of each team's turn."""
        self.has_moved = False
        self.has_acted = False

    # ── Combat ────────────────────────────────────────────────────────────────

    def compute_hit_rate(self, defender: "Unit", terrain_avoid: int = 0) -> int:
        """
        Returns the effective hit rate (0–100) of self attacking defender.
        Hit% = (weapon_hit + SKL*2 + LCK) - (SPD*2 + LCK + terrain_avoid)
        Clamped to [0, 100].
        """
        attacker_hit = self.weapon.hit + self.stats.skl * 2 + self.stats.lck
        defender_avoid = defender.stats.spd * 2 + defender.stats.lck + terrain_avoid
        return max(0, min(100, attacker_hit - defender_avoid))

    def compute_crit_rate(self, defender: "Unit") -> int:
        """
        Returns the effective crit rate (0–100).
        Crit% = (weapon_crit + SKL//2) - defender.LCK
        Clamped to [0, 100].
        """
        raw = self.weapon.crit + self.stats.skl // 2 - defender.stats.lck
        return max(0, min(100, raw))

    def compute_damage(self, defender: "Unit", critical: bool = False) -> int:
        """
        Returns the raw damage dealt (before application to HP).
        Physical : ATK = STR + might,  reduced by DEF
        Magical   : ATK = MAG + might,  reduced by RES
        Heal      : amount = MAG + might (always positive, targets ally HP)
        Crit multiplies damage by 3 (classic FE formula).
        """
        if self.weapon.damage_type == DamageType.HEAL:
            return self.stats.mag + self.weapon.might

        if self.weapon.damage_type == DamageType.PHYSICAL:
            atk = self.stats.str + self.weapon.might
            reduction = defender.stats.defense
        else:  # MAGICAL
            atk = self.stats.mag + self.weapon.might
            reduction = defender.stats.res

        raw = max(1, atk - reduction)  # minimum 1 damage on hit
        return raw * 3 if critical else raw

    def in_range(self, target_pos: tuple[int, int]) -> bool:
        """True if target_pos is within this unit's weapon range (Chebyshev distance)."""
        dr = abs(self.pos[0] - target_pos[0])
        dc = abs(self.pos[1] - target_pos[1])
        dist = max(dr, dc)
        return self.weapon.min_range <= dist <= self.weapon.max_range

    # ── Apply results ─────────────────────────────────────────────────────────

    def take_damage(self, amount: int) -> None:
        self.current_hp = max(0, self.current_hp - amount)

    def receive_heal(self, amount: int) -> None:
        self.current_hp = min(self.stats.hp, self.current_hp + amount)

    # ── Observation vector ────────────────────────────────────────────────────

    def to_obs_vector(self) -> list[float]:
        """
        Compact float representation for the RL observation space.
        All values normalised to [0, 1].
        10 features: [class_id, team_id, row, col, hp_ratio,
                    has_moved, has_acted, movement, min_range, max_range]
        """
        from tactilai.env.grid import GRID_SIZE, MAX_MOVEMENT, MAX_RANGE

        return [
            float(self.unit_class) / (len(UnitClass) - 1),
            float(self.team),
            float(self.pos[0]) / (GRID_SIZE - 1),
            float(self.pos[1]) / (GRID_SIZE - 1),
            self.hp_ratio,
            float(self.has_moved),
            float(self.has_acted),
            float(self.movement) / MAX_MOVEMENT,
            float(self.weapon.min_range) / MAX_RANGE,
            float(self.weapon.max_range) / MAX_RANGE,
        ]

    def __repr__(self) -> str:
        team_str = "B" if self.team == Team.BLUE else "R"
        return (
            f"{self.unit_class.name[:3]}{team_str}"
            f"(uid={self.uid}, hp={self.current_hp}/{self.stats.hp}, pos={self.pos})"
        )
