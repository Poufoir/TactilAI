"""
tests/env/test_wrapper.py

Integration tests for the TactilAIEnv Gymnasium wrapper.
Run with : pytest tests/env/test_wrapper.py -v
"""

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from tactilai.env.gym_wrapper import (
    ACTION_SIZE,
    OBS_SIZE,
    TactilAIEnv,
    decode_action,
    encode_action,
)
from tactilai.env.unit import Team

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def env() -> TactilAIEnv:
    e = TactilAIEnv(team=Team.BLUE, seed=42)
    yield e
    e.close()


# ── Encoding ──────────────────────────────────────────────────────────────────


class TestEncoding:
    def test_encode_decode_roundtrip(self) -> None:
        for uid in range(6):
            for move in [0, 1, 50, 256]:
                for act in range(13):
                    encoded = encode_action(uid, move, act)
                    assert decode_action(encoded) == (uid, move, act)

    def test_encoded_within_action_space(self) -> None:
        for uid in range(6):
            encoded = encode_action(uid, 0, 0)
            assert 0 <= encoded < ACTION_SIZE


# ── Spaces ────────────────────────────────────────────────────────────────────


class TestSpaces:
    def test_action_space_size(self, env: TactilAIEnv) -> None:
        assert env.action_space.n == ACTION_SIZE

    def test_observation_space_shape(self, env: TactilAIEnv) -> None:
        assert env.observation_space.shape == (OBS_SIZE,)

    def test_obs_dtype(self, env: TactilAIEnv) -> None:
        obs, _ = env.reset()
        assert obs.dtype == np.float32

    def test_obs_shape_after_reset(self, env: TactilAIEnv) -> None:
        obs, _ = env.reset()
        assert obs.shape == (OBS_SIZE,)

    def test_obs_in_bounds(self, env: TactilAIEnv) -> None:
        obs, _ = env.reset()
        assert obs.min() >= 0.0
        assert obs.max() <= 1.0


# ── Action mask ───────────────────────────────────────────────────────────────


class TestActionMask:
    def test_mask_dtype(self, env: TactilAIEnv) -> None:
        _, info = env.reset()
        assert info["action_mask"].dtype == np.int8

    def test_mask_shape(self, env: TactilAIEnv) -> None:
        _, info = env.reset()
        assert info["action_mask"].shape == (ACTION_SIZE,)

    def test_mask_has_legal_actions(self, env: TactilAIEnv) -> None:
        _, info = env.reset()
        assert info["action_mask"].sum() > 0

    def test_masked_sample_in_action_space(self, env: TactilAIEnv) -> None:
        _, info = env.reset()
        mask = info["action_mask"]
        action = env.action_space.sample(mask=mask)
        assert 0 <= action < ACTION_SIZE

    def test_masked_action_decodes_to_valid_unit(self, env: TactilAIEnv) -> None:
        _, info = env.reset()
        mask = info["action_mask"]
        action = env.action_space.sample(mask=mask)
        uid, _, _ = decode_action(action)
        assert 0 <= uid < 6


# ── Step ──────────────────────────────────────────────────────────────────────


class TestStep:
    def test_step_returns_correct_types(self, env: TactilAIEnv) -> None:
        _, info = env.reset()
        mask = info["action_mask"]
        action = env.action_space.sample(mask=mask)
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_obs_shape_after_step(self, env: TactilAIEnv) -> None:
        _, info = env.reset()
        action = env.action_space.sample(mask=info["action_mask"])
        obs, *_ = env.step(action)
        assert obs.shape == (OBS_SIZE,)

    def test_time_penalty_applied(self, env: TactilAIEnv) -> None:
        _, info = env.reset()
        action = encode_action(0, 0, 0)  # stay + skip
        _, reward, _, _, _ = env.step(action)
        assert reward < 0  # at minimum the time penalty

    def test_reset_resets_turn(self, env: TactilAIEnv) -> None:
        _, info = env.reset()
        for _ in range(10):
            action = env.action_space.sample(mask=info["action_mask"])
            _, _, _, _, info = env.step(action)
        env.reset()
        assert env._grid.turn == 0

    def test_episode_completes(self, env: TactilAIEnv) -> None:
        """Runs a full random episode and checks it terminates."""
        _, info = env.reset(seed=0)
        for _ in range(5000):
            action = env.action_space.sample(mask=info["action_mask"])
            _, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                assert info["winner"] is not None or truncated
                return
        pytest.fail("Episode did not terminate within 5000 steps")

    def test_winner_is_valid(self, env: TactilAIEnv) -> None:
        _, info = env.reset(seed=1)
        for _ in range(5000):
            action = env.action_space.sample(mask=info["action_mask"])
            _, _, terminated, truncated, info = env.step(action)
            if terminated:
                assert info["winner"] in ("BLUE", "RED", None)
                return


# ── Gymnasium compliance ──────────────────────────────────────────────────────


class TestGymnasiumCompliance:
    def test_check_env(self) -> None:
        """Official Gymnasium environment checker."""
        env = TactilAIEnv(team=Team.BLUE, seed=42)
        check_env(env, warn=True)
        env.close()
