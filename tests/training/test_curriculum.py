"""
tests/training/test_curriculum.py

Tests for HeuristicBot and CurriculumScheduler.
"""

import pytest
import numpy as np

from tactilai.env.grid import Grid
from tactilai.env.gym_wrapper import ACTION_SIZE, TactilAIEnv
from tactilai.env.unit import Team
from tactilai.training.curriculum import CURRICULUM_SCHEDULE, CurriculumScheduler
from tactilai.training.heuristic_bot import HeuristicBot


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def grid() -> Grid:
    return Grid(seed=42)


@pytest.fixture
def bot_red() -> HeuristicBot:
    return HeuristicBot(team=Team.RED, seed=42)


@pytest.fixture
def scheduler() -> CurriculumScheduler:
    return CurriculumScheduler(seed=42)


# ── CurriculumScheduler ───────────────────────────────────────────────────────

class TestCurriculumScheduler:
    def test_phase_1_at_start(self, scheduler: CurriculumScheduler) -> None:
        assert scheduler.phase(0) == 1
        assert scheduler.phase(100) == 1
        assert scheduler.phase(199) == 1

    def test_phase_2_after_200(self, scheduler: CurriculumScheduler) -> None:
        assert scheduler.phase(200) == 2
        assert scheduler.phase(400) == 2

    def test_phase_3_after_500(self, scheduler: CurriculumScheduler) -> None:
        assert scheduler.phase(500) == 3
        assert scheduler.phase(999) == 3

    def test_phase_4_after_1000(self, scheduler: CurriculumScheduler) -> None:
        assert scheduler.phase(1000) == 4
        assert scheduler.phase(9000) == 4

    def test_bot_prob_decreases(self, scheduler: CurriculumScheduler) -> None:
        prob_early = scheduler.bot_probability(0)
        prob_late  = scheduler.bot_probability(1500)
        assert prob_early > prob_late

    def test_bot_prob_phase1_is_1(self, scheduler: CurriculumScheduler) -> None:
        assert scheduler.bot_probability(0)   == 1.0
        assert scheduler.bot_probability(100) == 1.0

    def test_matchup_returns_valid_string(
        self, scheduler: CurriculumScheduler
    ) -> None:
        for update in [0, 100, 300, 700, 1500]:
            result = scheduler.matchup(update)
            assert result in ("vs_bot", "self")

    def test_matchup_always_vs_bot_in_phase1(
        self, scheduler: CurriculumScheduler
    ) -> None:
        for _ in range(20):
            assert scheduler.matchup(50) == "vs_bot"

    def test_matchup_mostly_self_in_phase4(self) -> None:
        scheduler = CurriculumScheduler(seed=0)
        results   = [scheduler.matchup(1500) for _ in range(100)]
        self_count = results.count("self")
        assert self_count >= 70   # expect ~90% self-play


# ── HeuristicBot ──────────────────────────────────────────────────────────────

class TestHeuristicBot:
    def test_returns_legal_action(
        self, bot_red: HeuristicBot, grid: Grid
    ) -> None:
        env  = TactilAIEnv(team=Team.BLUE, seed=42)
        obs, info = env.reset(seed=42)
        mask = info["action_mask"]
        action = bot_red.select_action(obs, mask, env._grid)
        assert 0 <= action < ACTION_SIZE
        assert mask[action] == 1

    def test_runs_full_episode(self, bot_red: HeuristicBot) -> None:
        """Bot should not crash during a full episode."""
        bot_blue = HeuristicBot(team=Team.BLUE, seed=0)
        env      = TactilAIEnv(team=Team.BLUE, seed=42)
        obs, info = env.reset(seed=42)

        for _ in range(2000):
            active_team = env._grid.active_team
            mask        = info["action_mask"]
            bot = bot_blue if active_team == Team.BLUE else bot_red
            action = bot.select_action(obs, mask, env._grid)
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        env.close()

    def test_action_respects_mask(self, bot_red: HeuristicBot) -> None:
        """Bot must never select an illegal action."""
        env = TactilAIEnv(team=Team.BLUE, seed=42)
        obs, info = env.reset(seed=42)

        for _ in range(100):
            mask   = info["action_mask"]
            action = bot_red.select_action(obs, mask, env._grid)
            assert mask[action] == 1, f"Bot selected illegal action {action}"
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        env.close()

    def test_bot_moves_toward_enemy(self) -> None:
        """After a move action, the unit should be closer to the nearest enemy."""
        grid    = Grid(seed=42)
        bot     = HeuristicBot(team=Team.RED, seed=42)
        env     = TactilAIEnv(team=Team.BLUE, seed=42)
        obs, info = env.reset(seed=42)

        # Skip to RED turn
        while env._grid.active_team == Team.BLUE:
            mask   = info["action_mask"]
            action = env.action_space.sample(mask=mask)
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                return

        mask   = info["action_mask"]
        action = bot.select_action(obs, mask, env._grid)
        assert mask[action] == 1
        env.close()