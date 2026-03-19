"""
tests/training/test_training.py

Unit tests for tactilai/training/ — ELO, CheckpointPool, SelfPlayTrainer.
Run with : pytest tests/training/test_training.py -v
"""

from pathlib import Path

import pytest
import torch

from tactilai.agents.ppo import PPOAgent
from tactilai.training.elo import ELO_START, ELOTracker
from tactilai.training.pool import CheckpointPool

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def elo() -> ELOTracker:
    tracker = ELOTracker()
    tracker.register("alice")
    tracker.register("bob")
    return tracker


@pytest.fixture
def pool(tmp_path: Path, device: torch.device) -> CheckpointPool:
    return CheckpointPool(max_size=5, save_dir=tmp_path / "pool", seed=42)


@pytest.fixture
def agent(device: torch.device) -> PPOAgent:
    return PPOAgent(device=device)


# ── ELOTracker ────────────────────────────────────────────────────────────────


class TestELOTracker:
    def test_initial_rating(self, elo: ELOTracker) -> None:
        assert elo.rating("alice") == ELO_START
        assert elo.rating("bob") == ELO_START

    def test_winner_gains_rating(self, elo: ELOTracker) -> None:
        elo.update("alice", "bob", winner="alice")
        assert elo.rating("alice") > ELO_START

    def test_loser_loses_rating(self, elo: ELOTracker) -> None:
        elo.update("alice", "bob", winner="alice")
        assert elo.rating("bob") < ELO_START

    def test_ratings_sum_conserved(self, elo: ELOTracker) -> None:
        """ELO is zero-sum: total rating change sums to zero."""
        before = elo.rating("alice") + elo.rating("bob")
        elo.update("alice", "bob", winner="alice")
        after = elo.rating("alice") + elo.rating("bob")
        assert abs(after - before) < 1e-6

    def test_draw_minimal_change(self, elo: ELOTracker) -> None:
        """Equal players drawing should produce minimal rating change."""
        elo.update("alice", "bob", winner=None)
        assert abs(elo.rating("alice") - ELO_START) < 1.0
        assert abs(elo.rating("bob") - ELO_START) < 1.0

    def test_expected_score_equal_players(self, elo: ELOTracker) -> None:
        assert elo.expected_score("alice", "bob") == pytest.approx(0.5)

    def test_expected_score_stronger_player(self, elo: ELOTracker) -> None:
        elo.ratings["alice"] = 1400
        elo.ratings["bob"] = 1000
        assert elo.expected_score("alice", "bob") > 0.9

    def test_auto_register_unknown(self, elo: ELOTracker) -> None:
        elo.update("alice", "carol", winner="alice")
        assert "carol" in elo.ratings

    def test_history_recorded(self, elo: ELOTracker) -> None:
        elo.update("alice", "bob", winner="alice")
        assert len(elo.history) == 1
        assert elo.history[0]["winner"] == "alice"

    def test_leaderboard_sorted(self, elo: ELOTracker) -> None:
        elo.ratings["alice"] = 1200
        elo.ratings["bob"] = 800
        board = elo.leaderboard()
        assert board[0][0] == "alice"
        assert board[1][0] == "bob"

    def test_n_games_incremented(self, elo: ELOTracker) -> None:
        elo.update("alice", "bob", winner="alice")
        assert elo.n_games["alice"] == 1
        assert elo.n_games["bob"] == 1

    def test_k_factor_decreases(self, elo: ELOTracker) -> None:
        k_early = elo._k_factor("alice")
        elo.n_games["alice"] = 1000
        k_late = elo._k_factor("alice")
        assert k_late < k_early

    def test_save_and_load(self, elo: ELOTracker, tmp_path: Path) -> None:
        elo.update("alice", "bob", winner="alice")
        path = tmp_path / "elo.json"
        elo.save(path)
        loaded = ELOTracker.load(path)
        assert loaded.rating("alice") == pytest.approx(elo.rating("alice"))
        assert loaded.rating("bob") == pytest.approx(elo.rating("bob"))
        assert len(loaded.history) == len(elo.history)


# ── CheckpointPool ────────────────────────────────────────────────────────────


class TestCheckpointPool:
    def test_initially_empty(self, pool: CheckpointPool) -> None:
        assert pool.is_empty
        assert pool.size == 0

    def test_save_increases_size(self, pool: CheckpointPool, agent: PPOAgent) -> None:
        pool.save_checkpoint(agent, update_step=1)
        assert pool.size == 1

    def test_save_creates_file(
        self, pool: CheckpointPool, agent: PPOAgent, tmp_path: Path
    ) -> None:
        path = pool.save_checkpoint(agent, update_step=1)
        assert path.exists()

    def test_evicts_oldest_when_full(
        self, pool: CheckpointPool, agent: PPOAgent
    ) -> None:
        for i in range(6):  # max_size=5, so 6th evicts the 1st
            pool.save_checkpoint(agent, update_step=i)
        assert pool.size == 5

    def test_load_random_restores_weights(
        self, pool: CheckpointPool, agent: PPOAgent, device: torch.device
    ) -> None:
        pool.save_checkpoint(agent, update_step=1)
        agent2 = PPOAgent(device=device)
        pool.load_random(agent2)
        for (_, p1), (_, p2) in zip(
            agent.actor_critic.named_parameters(),
            agent2.actor_critic.named_parameters(),
        ):
            assert torch.allclose(p1, p2)

    def test_load_random_raises_when_empty(
        self, pool: CheckpointPool, agent: PPOAgent
    ) -> None:
        with pytest.raises(RuntimeError):
            pool.load_random(agent)

    def test_load_latest(
        self, pool: CheckpointPool, agent: PPOAgent, device: torch.device
    ) -> None:
        pool.save_checkpoint(agent, update_step=1)
        pool.save_checkpoint(agent, update_step=2)
        agent2 = PPOAgent(device=device)
        path = pool.load_latest(agent2)
        assert "000002" in path.stem

    def test_load_oldest(
        self, pool: CheckpointPool, agent: PPOAgent, device: torch.device
    ) -> None:
        pool.save_checkpoint(agent, update_step=1)
        pool.save_checkpoint(agent, update_step=2)
        agent2 = PPOAgent(device=device)
        path = pool.load_oldest(agent2)
        assert "000001" in path.stem

    def test_clear_removes_files(self, pool: CheckpointPool, agent: PPOAgent) -> None:
        paths = [pool.save_checkpoint(agent, i) for i in range(3)]
        pool.clear()
        assert pool.is_empty
        for p in paths:
            assert not p.exists()

    def test_checkpoint_names(self, pool: CheckpointPool, agent: PPOAgent) -> None:
        pool.save_checkpoint(agent, update_step=42)
        assert "checkpoint_000042" in pool.checkpoint_names

    def test_reload_from_disk(
        self,
        pool: CheckpointPool,
        agent: PPOAgent,
        tmp_path: Path,
        device: torch.device,
    ) -> None:
        """A new pool instance should reload existing checkpoints."""
        pool.save_checkpoint(agent, update_step=1)
        pool.save_checkpoint(agent, update_step=2)
        pool2 = CheckpointPool(max_size=5, save_dir=tmp_path / "pool", seed=42)
        assert pool2.size == 2
