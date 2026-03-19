"""
tests/agents/test_ppo.py

Unit tests for tactilai/agents/ppo.py
Run with : pytest tests/agents/test_ppo.py -v
"""

import numpy as np
import pytest
import torch

from tactilai.agents.ppo import ROLLOUT_STEPS, PPOAgent, compute_gae
from tactilai.env.gym_wrapper import ACTION_SIZE, OBS_SIZE

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def agent(device: torch.device) -> PPOAgent:
    return PPOAgent(device=device)


@pytest.fixture
def dummy_transition() -> dict:
    """A single random transition."""
    return {
        "obs": torch.rand(OBS_SIZE),
        "obs_next": torch.rand(OBS_SIZE),
        "action": torch.tensor(0),
        "log_prob": torch.tensor(-3.0),
        "reward": 1.0,
        "value": torch.tensor(0.5),
        "done": False,
        "mask": torch.ones(ACTION_SIZE, dtype=torch.int8),
    }


def fill_buffer(agent: PPOAgent, n: int = ROLLOUT_STEPS) -> None:
    """Fills the agent's buffer with n random transitions."""
    for _ in range(n):
        agent.buffer.add(
            obs=torch.rand(OBS_SIZE),
            obs_next=torch.rand(OBS_SIZE),
            action=torch.tensor(0),
            log_prob=torch.tensor(-3.0),
            reward=1.0,
            value=torch.tensor(0.5),
            done=False,
            mask=torch.ones(ACTION_SIZE, dtype=torch.int8),
        )


# ── RolloutBuffer ─────────────────────────────────────────────────────────────


class TestRolloutBuffer:
    def test_add_increases_length(
        self, agent: PPOAgent, dummy_transition: dict
    ) -> None:
        t = dummy_transition
        agent.buffer.add(
            t["obs"],
            t["obs_next"],
            t["action"],
            t["log_prob"],
            t["reward"],
            t["value"],
            t["done"],
            t["mask"],
        )
        assert len(agent.buffer) == 1

    def test_is_full(self, agent: PPOAgent) -> None:
        fill_buffer(agent, ROLLOUT_STEPS)
        assert agent.buffer.is_full

    def test_clear_resets_buffer(self, agent: PPOAgent) -> None:
        fill_buffer(agent, 10)
        agent.buffer.clear()
        assert len(agent.buffer) == 0

    def test_as_tensors_shapes(self, agent: PPOAgent) -> None:
        fill_buffer(agent, 16)
        data = agent.buffer.as_tensors()
        assert data["obs"].shape == (16, OBS_SIZE)
        assert data["obs_next"].shape == (16, OBS_SIZE)
        assert data["actions"].shape == (16,)
        assert data["log_probs"].shape == (16,)
        assert data["rewards"].shape == (16,)
        assert data["values"].shape == (16,)
        assert data["dones"].shape == (16,)
        assert data["masks"].shape == (16, ACTION_SIZE)

    def test_as_tensors_on_cpu(self, agent: PPOAgent) -> None:
        fill_buffer(agent, 8)
        data = agent.buffer.as_tensors()
        for v in data.values():
            assert v.device.type == "cpu"


# ── GAE ───────────────────────────────────────────────────────────────────────


class TestComputeGAE:
    def test_output_shapes(self) -> None:
        T = 16
        rewards = torch.ones(T)
        values = torch.zeros(T)
        dones = torch.zeros(T)
        last_value = torch.tensor(0.0)
        adv, ret = compute_gae(rewards, values, dones, last_value)
        assert adv.shape == (T,)
        assert ret.shape == (T,)

    def test_returns_equal_advantages_plus_values(self) -> None:
        T = 8
        rewards = torch.rand(T)
        values = torch.rand(T)
        dones = torch.zeros(T)
        last_value = torch.tensor(0.0)
        adv, ret = compute_gae(rewards, values, dones, last_value)
        assert torch.allclose(ret, adv + values, atol=1e-5)

    def test_done_resets_bootstrap(self) -> None:
        """After a done=1, the next value should not bootstrap."""
        rewards = torch.tensor([1.0, 1.0])
        values = torch.tensor([0.0, 0.0])
        dones = torch.tensor([1.0, 0.0])  # episode ends at t=0
        last_value = torch.tensor(10.0)
        adv, _ = compute_gae(rewards, values, dones, last_value)
        # At t=0, done=1 → next value is 0, not bootstrapped from values[1]
        assert adv[0].item() == pytest.approx(1.0, abs=1e-4)

    def test_no_nan_in_output(self) -> None:
        T = 32
        rewards = torch.rand(T)
        values = torch.rand(T)
        dones = torch.zeros(T)
        last_value = torch.tensor(0.5)
        adv, ret = compute_gae(rewards, values, dones, last_value)
        assert not torch.isnan(adv).any()
        assert not torch.isnan(ret).any()

    def test_zero_reward_zero_advantage(self) -> None:
        """With all-zero rewards and values, advantages should be zero."""
        T = 8
        rewards = torch.zeros(T)
        values = torch.zeros(T)
        dones = torch.zeros(T)
        last_value = torch.tensor(0.0)
        adv, _ = compute_gae(rewards, values, dones, last_value)
        assert torch.allclose(adv, torch.zeros(T), atol=1e-5)


# ── PPOAgent ──────────────────────────────────────────────────────────────────


class TestPPOAgent:
    def test_select_action_returns_valid_action(self, agent: PPOAgent) -> None:
        obs = np.random.rand(OBS_SIZE).astype(np.float32)
        mask = np.ones(ACTION_SIZE, dtype=np.int8)
        action, log_prob, value = agent.select_action(obs, mask)
        assert 0 <= action < ACTION_SIZE
        assert isinstance(log_prob, torch.Tensor)
        assert isinstance(value, torch.Tensor)

    def test_select_action_respects_mask(self, agent: PPOAgent) -> None:
        """With only action 0 legal, must always return 0."""
        obs = np.random.rand(OBS_SIZE).astype(np.float32)
        mask = np.zeros(ACTION_SIZE, dtype=np.int8)
        mask[0] = 1
        for _ in range(10):
            action, _, _ = agent.select_action(obs, mask)
            assert action == 0

    def test_select_action_log_prob_is_finite(self, agent: PPOAgent) -> None:
        obs = np.random.rand(OBS_SIZE).astype(np.float32)
        mask = np.ones(ACTION_SIZE, dtype=np.int8)
        _, log_prob, _ = agent.select_action(obs, mask)
        assert torch.isfinite(log_prob)

    def test_update_returns_metrics(self, agent: PPOAgent) -> None:
        fill_buffer(agent, ROLLOUT_STEPS)
        last_obs = np.random.rand(OBS_SIZE).astype(np.float32)
        metrics = agent.update(last_obs)
        assert isinstance(metrics, dict)
        assert "loss/policy" in metrics
        assert "loss/value" in metrics
        assert "loss/entropy" in metrics
        assert "loss/icm" in metrics

    def test_update_clears_buffer(self, agent: PPOAgent) -> None:
        fill_buffer(agent, ROLLOUT_STEPS)
        last_obs = np.random.rand(OBS_SIZE).astype(np.float32)
        agent.update(last_obs)
        assert len(agent.buffer) == 0

    def test_update_metrics_are_finite(self, agent: PPOAgent) -> None:
        fill_buffer(agent, ROLLOUT_STEPS)
        last_obs = np.random.rand(OBS_SIZE).astype(np.float32)
        metrics = agent.update(last_obs)
        for k, v in metrics.items():
            assert np.isfinite(v), f"Non-finite metric: {k} = {v}"

    def test_save_and_load(
        self, agent: PPOAgent, tmp_path: pytest.TempPathFactory
    ) -> None:
        path = str(tmp_path / "checkpoint.pt")
        agent.save(path)
        # Load into a fresh agent and check weights match
        agent2 = PPOAgent(device=agent.device)
        agent2.load(path)
        for (n1, p1), (n2, p2) in zip(
            agent.actor_critic.named_parameters(),
            agent2.actor_critic.named_parameters(),
        ):
            assert torch.allclose(p1, p2), f"Mismatch in {n1}"

    def test_buffer_not_full_before_update(self, agent: PPOAgent) -> None:
        fill_buffer(agent, ROLLOUT_STEPS // 2)
        assert not agent.buffer.is_full
