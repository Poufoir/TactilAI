"""
tests/env/test_env.py

Unit tests for tactilai/env/ — Grid, Unit, Terrain logic.
Run with : pytest tests/env/test_env.py -v
"""

import numpy as np
import pytest

from tactilai.env.grid import GRID_SIZE, NUM_UNITS, Grid
from tactilai.env.terrain import (
    TerrainType,
    get_avoid_bonus,
    get_movement_cost,
    is_passable,
)
from tactilai.env.unit import DamageType, Team, Unit, UnitClass

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def grid() -> Grid:
    """Deterministic grid for all tests."""
    return Grid(seed=42)


@pytest.fixture
def knight(grid: Grid) -> Unit:
    return next(
        u for u in grid.alive_units(Team.BLUE) if u.unit_class == UnitClass.KNIGHT
    )


@pytest.fixture
def barbarian(grid: Grid) -> Unit:
    return next(
        u for u in grid.alive_units(Team.RED) if u.unit_class == UnitClass.BARBARIAN
    )


@pytest.fixture
def healer(grid: Grid) -> Unit:
    return next(
        u for u in grid.alive_units(Team.BLUE) if u.unit_class == UnitClass.HEALER
    )


# ── Unit tests ────────────────────────────────────────────────────────────────


class TestUnit:
    def test_initial_hp_equals_max(self, grid: Grid) -> None:
        for unit in grid.alive_units():
            assert unit.current_hp == unit.stats.hp

    def test_take_damage_reduces_hp(self, knight: Unit) -> None:
        knight.take_damage(10)
        assert knight.current_hp == knight.stats.hp - 10

    def test_take_damage_cannot_go_below_zero(self, knight: Unit) -> None:
        knight.take_damage(9999)
        assert knight.current_hp == 0

    def test_receive_heal_cannot_exceed_max_hp(self, knight: Unit) -> None:
        knight.take_damage(5)
        knight.receive_heal(9999)
        assert knight.current_hp == knight.stats.hp

    def test_is_alive(self, knight: Unit) -> None:
        assert knight.is_alive
        knight.take_damage(9999)
        assert not knight.is_alive

    def test_reset_turn_clears_flags(self, knight: Unit) -> None:
        knight.has_moved = True
        knight.has_acted = True
        knight.reset_turn()
        assert not knight.has_moved
        assert not knight.has_acted

    def test_hp_ratio(self, knight: Unit) -> None:
        knight.take_damage(knight.stats.hp // 2)
        assert pytest.approx(knight.hp_ratio, abs=0.05) == 0.5

    def test_obs_vector_length(self, knight: Unit) -> None:
        assert len(knight.to_obs_vector()) == 10

    def test_obs_vector_values_in_range(self, knight: Unit) -> None:
        obs = knight.to_obs_vector()
        # hp_ratio (index 4) must be in [0, 1]
        assert 0.0 <= obs[4] <= 1.0

    def test_hit_rate_clamped(self, knight: Unit, barbarian: Unit) -> None:
        hit = knight.compute_hit_rate(barbarian)
        assert 0 <= hit <= 100

    def test_crit_rate_clamped(self, knight: Unit, barbarian: Unit) -> None:
        crit = knight.compute_crit_rate(barbarian)
        assert 0 <= crit <= 100

    def test_physical_damage_positive(self, knight: Unit, barbarian: Unit) -> None:
        dmg = knight.compute_damage(barbarian)
        assert dmg >= 1

    def test_magical_damage_uses_res(self, grid: Grid) -> None:
        mage = next(
            u for u in grid.alive_units(Team.BLUE) if u.unit_class == UnitClass.MAGE
        )
        barbarian = next(
            u for u in grid.alive_units(Team.RED) if u.unit_class == UnitClass.BARBARIAN
        )
        dmg = mage.compute_damage(barbarian)
        expected = max(1, mage.stats.mag + mage.weapon.might - barbarian.stats.res)
        assert dmg == expected

    def test_crit_triples_damage(self, knight: Unit, barbarian: Unit) -> None:
        normal = knight.compute_damage(barbarian, critical=False)
        crit = knight.compute_damage(barbarian, critical=True)
        assert crit == normal * 3

    def test_healer_damage_type(self, healer: Unit) -> None:
        assert healer.weapon.damage_type == DamageType.HEAL

    def test_in_range_melee(self, knight: Unit) -> None:
        knight.pos = (5, 5)
        assert knight.in_range((5, 6))
        assert knight.in_range((6, 5))
        assert not knight.in_range((5, 8))

    def test_archer_range(self, grid: Grid) -> None:
        archer = next(
            u for u in grid.alive_units(Team.BLUE) if u.unit_class == UnitClass.ARCHER
        )
        archer.pos = (5, 5)
        assert not archer.in_range((5, 6))  # too close (min_range=2)
        assert archer.in_range((5, 7))
        assert archer.in_range((5, 8))
        assert not archer.in_range((5, 9))  # too far (max_range=3)


# ── Terrain tests ─────────────────────────────────────────────────────────────


class TestTerrain:
    def test_plain_movement_cost(self) -> None:
        assert get_movement_cost(TerrainType.PLAIN, UnitClass.KNIGHT) == 1

    def test_forest_slows_cavalry(self) -> None:
        assert get_movement_cost(TerrainType.FOREST, UnitClass.CAVALIER) == 3

    def test_archer_ignores_forest(self) -> None:
        assert get_movement_cost(TerrainType.FOREST, UnitClass.ARCHER) == 1

    def test_barbarian_adapted_to_mountain(self) -> None:
        cost_bar = get_movement_cost(TerrainType.MOUNTAIN, UnitClass.BARBARIAN)
        cost_kni = get_movement_cost(TerrainType.MOUNTAIN, UnitClass.KNIGHT)
        assert cost_bar < cost_kni

    def test_avoid_bonus_increases_with_cover(self) -> None:
        assert get_avoid_bonus(TerrainType.PLAIN) < get_avoid_bonus(TerrainType.FOREST)
        assert get_avoid_bonus(TerrainType.FOREST) < get_avoid_bonus(
            TerrainType.MOUNTAIN
        )

    def test_all_terrain_passable(self) -> None:
        for t in TerrainType:
            assert is_passable(t)


# ── Grid tests ────────────────────────────────────────────────────────────────


class TestGrid:
    def test_grid_size(self, grid: Grid) -> None:
        assert grid.terrain.shape == (GRID_SIZE, GRID_SIZE)

    def test_spawn_zones_are_plain(self, grid: Grid) -> None:
        for row in range(GRID_SIZE):
            assert grid.terrain[row, 0] == TerrainType.PLAIN.value
            assert grid.terrain[row, 15] == TerrainType.PLAIN.value

    def test_correct_number_of_units(self, grid: Grid) -> None:
        assert len(grid.alive_units(Team.BLUE)) == NUM_UNITS
        assert len(grid.alive_units(Team.RED)) == NUM_UNITS

    def test_no_unit_overlap_on_spawn(self, grid: Grid) -> None:
        positions = [u.pos for u in grid.alive_units()]
        assert len(positions) == len(set(positions))

    def test_blue_spawns_left(self, grid: Grid) -> None:
        for unit in grid.alive_units(Team.BLUE):
            assert unit.pos[1] < 3

    def test_red_spawns_right(self, grid: Grid) -> None:
        for unit in grid.alive_units(Team.RED):
            assert unit.pos[1] >= 13

    def test_obs_array_shape(self, grid: Grid) -> None:
        obs = grid.to_obs_array()
        expected = GRID_SIZE * GRID_SIZE + NUM_UNITS * 2 * 10 + 2
        assert obs.shape == (expected,)
        assert obs.dtype == np.float32

    def test_obs_array_in_range(self, grid: Grid) -> None:
        obs = grid.to_obs_array()
        assert obs.min() >= 0.0
        assert obs.max() <= 1.0

    def test_get_unit_at(self, grid: Grid, knight: Unit) -> None:
        assert grid.get_unit_at(knight.pos) is knight

    def test_get_unit_at_empty(self, grid: Grid) -> None:
        assert grid.get_unit_at((7, 7)) is None

    def test_reachable_tiles_not_empty(self, grid: Grid, knight: Unit) -> None:
        tiles = grid.reachable_tiles(knight)
        assert len(tiles) > 0

    def test_reachable_tiles_within_movement(self, grid: Grid, knight: Unit) -> None:
        tiles = grid.reachable_tiles(knight)
        for tile in tiles:
            assert grid.in_bounds(tile)

    def test_move_unit_success(self, grid: Grid, knight: Unit) -> None:
        reachable = grid.reachable_tiles(knight)
        target = next(iter(reachable))
        assert grid.move_unit(knight, target)
        assert knight.pos == target
        assert knight.has_moved

    def test_move_unit_cannot_move_twice(self, grid: Grid, knight: Unit) -> None:
        reachable = list(grid.reachable_tiles(knight))
        grid.move_unit(knight, reachable[0])
        assert not grid.move_unit(
            knight, reachable[1] if len(reachable) > 1 else reachable[0]
        )

    def test_combat_reduces_hp(self, grid: Grid, knight: Unit, barbarian: Unit) -> None:
        knight.pos = (7, 7)
        barbarian.pos = (7, 8)
        hp_before = barbarian.current_hp
        grid.resolve_combat(knight, barbarian)
        # At least one hit should land over multiple seeds — deterministic with seed=42
        assert barbarian.current_hp <= hp_before

    def test_healer_increases_ally_hp(
        self, grid: Grid, healer: Unit, knight: Unit
    ) -> None:
        knight.take_damage(10)
        healer.pos = (7, 7)
        knight.pos = (7, 8)
        hp_before = knight.current_hp
        grid.resolve_combat(healer, knight)
        assert knight.current_hp >= hp_before

    def test_end_turn_switches_team(self, grid: Grid) -> None:
        assert grid.active_team == Team.BLUE
        grid.end_turn()
        assert grid.active_team == Team.RED

    def test_turn_increments_after_both_teams(self, grid: Grid) -> None:
        assert grid.turn == 0
        grid.end_turn()  # RED plays
        grid.end_turn()  # back to BLUE → turn += 1
        assert grid.turn == 1

    def test_winner_none_at_start(self, grid: Grid) -> None:
        assert grid.winner is None

    def test_winner_blue_when_red_wiped(self, grid: Grid) -> None:
        for unit in grid.alive_units(Team.RED):
            unit.take_damage(9999)
        assert grid.winner == Team.BLUE

    def test_is_terminal_on_wipe(self, grid: Grid) -> None:
        for unit in grid.alive_units(Team.RED):
            unit.take_damage(9999)
        assert grid.is_terminal
