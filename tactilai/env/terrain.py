"""
tactilai/env/terrain.py

Defines terrain types and their effect on movement and combat.

Terrain effects (classic Fire Emblem):
  - movement_cost : how many movement points it costs to enter this tile
  - defense_bonus : flat DEF/RES bonus while standing on this tile
  - avoid_bonus   : bonus to avoid% while standing on this tile
"""

from __future__ import annotations

import dataclasses
from enum import IntEnum

# ── Terrain types ─────────────────────────────────────────────────────────────


class TerrainType(IntEnum):
    PLAIN = 0
    FOREST = 1
    MOUNTAIN = 2


# ── Terrain profile ───────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class TerrainProfile:
    name: str
    movement_cost: int  # movement points consumed to enter
    defense_bonus: int  # added to DEF/RES for damage calculation
    avoid_bonus: int  # added to avoid% in hit rate calculation
    passable: bool  # False = impassable for all units


# Movement cost overrides per class (some terrain is easier for specific units)
# Format: {TerrainType: {UnitClass: cost}}  — missing entries use the default
from tactilai.env.unit import UnitClass  # noqa: E402  (local import to avoid circular)

_CLASS_MOVEMENT_OVERRIDE: dict[TerrainType, dict[UnitClass, int]] = {
    TerrainType.MOUNTAIN: {
        UnitClass.CAVALIER: 4,  # cavalry struggles in mountains
        UnitClass.BARBARIAN: 2,  # barbarians are mountain-adapted
    },
    TerrainType.FOREST: {
        UnitClass.CAVALIER: 3,  # cavalry is slowed in forests
        UnitClass.ARCHER: 1,  # archers move freely in forests
    },
}


# ── Terrain profiles ──────────────────────────────────────────────────────────

TERRAIN_PROFILES: dict[TerrainType, TerrainProfile] = {
    TerrainType.PLAIN: TerrainProfile(
        name="Plain",
        movement_cost=1,
        defense_bonus=0,
        avoid_bonus=0,
        passable=True,
    ),
    TerrainType.FOREST: TerrainProfile(
        name="Forest",
        movement_cost=2,
        defense_bonus=1,
        avoid_bonus=20,
        passable=True,
    ),
    TerrainType.MOUNTAIN: TerrainProfile(
        name="Mountain",
        movement_cost=3,
        defense_bonus=2,
        avoid_bonus=30,
        passable=True,
    ),
}


# ── Helper functions ──────────────────────────────────────────────────────────


def get_movement_cost(terrain: TerrainType, unit_class: UnitClass) -> int:
    """
    Returns the movement cost for a given unit class entering a terrain tile.
    Falls back to the terrain's default cost if no class override exists.
    """
    override = _CLASS_MOVEMENT_OVERRIDE.get(terrain, {})
    return override.get(unit_class, TERRAIN_PROFILES[terrain].movement_cost)


def get_defense_bonus(terrain: TerrainType) -> int:
    return TERRAIN_PROFILES[terrain].defense_bonus


def get_avoid_bonus(terrain: TerrainType) -> int:
    return TERRAIN_PROFILES[terrain].avoid_bonus


def is_passable(terrain: TerrainType) -> bool:
    return TERRAIN_PROFILES[terrain].passable


def terrain_display_char(terrain: TerrainType) -> str:
    """Single character for ASCII debug rendering."""
    return {
        TerrainType.PLAIN: ".",
        TerrainType.FOREST: "T",
        TerrainType.MOUNTAIN: "^",
    }[terrain]
