"""
tests/agents/test_network.py

Unit tests for tactilai/agents/network.py
Run with : pytest tests/agents/test_network.py -v
"""

import pytest
import torch

from tactilai.agents.network import (
    CNN_CHANNELS,
    CNN_FEATURES,
    ActorCritic,
    CNNEncoder,
    preprocess_obs,
)
from tactilai.env.grid import GRID_SIZE
from tactilai.env.gym_wrapper import ACTION_SIZE, OBS_SIZE

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def net(device: torch.device) -> ActorCritic:
    return ActorCritic(device=device)


@pytest.fixture
def obs_batch(device: torch.device) -> torch.Tensor:
    """Random normalised observation batch of size 4."""
    return torch.rand(4, OBS_SIZE, device=device)


@pytest.fixture
def mask_all(device: torch.device) -> torch.Tensor:
    """All-legal mask (all actions allowed)."""
    return torch.ones(4, ACTION_SIZE, dtype=torch.bool, device=device)


@pytest.fixture
def mask_single(device: torch.device) -> torch.Tensor:
    """Only the first action is legal."""
    mask = torch.zeros(4, ACTION_SIZE, dtype=torch.bool, device=device)
    mask[:, 0] = True
    return mask


# ── preprocess_obs ────────────────────────────────────────────────────────────


class TestPreprocessObs:
    def test_output_shapes(self, obs_batch: torch.Tensor, device: torch.device) -> None:
        grid, unit = preprocess_obs(obs_batch)
        assert grid.shape == (4, CNN_CHANNELS, GRID_SIZE, GRID_SIZE)
        assert unit.shape == (4, 10)

    def test_grid_values_in_range(self, obs_batch: torch.Tensor) -> None:
        grid, _ = preprocess_obs(obs_batch)
        assert grid.min() >= 0.0
        assert grid.max() <= 1.0

    def test_unit_values_in_range(self, obs_batch: torch.Tensor) -> None:
        _, unit = preprocess_obs(obs_batch)
        assert unit.min() >= 0.0
        assert unit.max() <= 1.0

    def test_output_on_correct_device(
        self, obs_batch: torch.Tensor, device: torch.device
    ) -> None:
        grid, unit = preprocess_obs(obs_batch)
        assert grid.device.type == device.type
        assert unit.device.type == device.type

    def test_single_obs(self, device: torch.device) -> None:
        """Batch size of 1 must work."""
        obs = torch.rand(1, OBS_SIZE, device=device)
        grid, unit = preprocess_obs(obs)
        assert grid.shape == (1, CNN_CHANNELS, GRID_SIZE, GRID_SIZE)
        assert unit.shape == (1, 10)


# ── CNNEncoder ────────────────────────────────────────────────────────────────


class TestCNNEncoder:
    def test_output_shape(self, device: torch.device) -> None:
        encoder = CNNEncoder().to(device)
        grid = torch.rand(4, CNN_CHANNELS, GRID_SIZE, GRID_SIZE, device=device)
        out = encoder(grid)
        assert out.shape == (4, CNN_FEATURES)

    def test_output_single_batch(self, device: torch.device) -> None:
        encoder = CNNEncoder().to(device)
        grid = torch.rand(1, CNN_CHANNELS, GRID_SIZE, GRID_SIZE, device=device)
        out = encoder(grid)
        assert out.shape == (1, CNN_FEATURES)

    def test_no_nan_in_output(self, device: torch.device) -> None:
        encoder = CNNEncoder().to(device)
        grid = torch.rand(4, CNN_CHANNELS, GRID_SIZE, GRID_SIZE, device=device)
        out = encoder(grid)
        assert not torch.isnan(out).any()

    def test_gradients_flow(self, device: torch.device) -> None:
        encoder = CNNEncoder().to(device)
        grid = torch.rand(
            4, CNN_CHANNELS, GRID_SIZE, GRID_SIZE, device=device, requires_grad=True
        )
        out = encoder(grid)
        loss = out.sum()
        loss.backward()
        assert grid.grad is not None


# ── ActorCritic ───────────────────────────────────────────────────────────────


class TestActorCritic:
    def test_forward_output_shapes(
        self, net: ActorCritic, obs_batch: torch.Tensor
    ) -> None:
        logits, value = net(obs_batch)
        assert logits.shape == (4, ACTION_SIZE)
        assert value.shape == (4, 1)

    def test_forward_single_obs(self, net: ActorCritic, device: torch.device) -> None:
        obs = torch.rand(1, OBS_SIZE, device=device)
        logits, value = net(obs)
        assert logits.shape == (1, ACTION_SIZE)
        assert value.shape == (1, 1)

    def test_no_nan_in_forward(self, net: ActorCritic, obs_batch: torch.Tensor) -> None:
        logits, value = net(obs_batch)
        assert not torch.isnan(logits).any()
        assert not torch.isnan(value).any()

    def test_get_value_shape(self, net: ActorCritic, obs_batch: torch.Tensor) -> None:
        value = net.get_value(obs_batch)
        assert value.shape == (4,)

    def test_masked_distribution_all_legal(
        self,
        net: ActorCritic,
        obs_batch: torch.Tensor,
        mask_all: torch.Tensor,
    ) -> None:
        dist = net.masked_distribution(obs_batch, mask_all)
        assert isinstance(dist, torch.distributions.Categorical)

    def test_masked_distribution_single_action(
        self,
        net: ActorCritic,
        obs_batch: torch.Tensor,
        mask_single: torch.Tensor,
    ) -> None:
        """When only action 0 is legal, all samples must be 0."""
        dist = net.masked_distribution(obs_batch, mask_single)
        action = dist.sample()
        assert (action == 0).all()

    def test_sample_shape(
        self,
        net: ActorCritic,
        obs_batch: torch.Tensor,
        mask_all: torch.Tensor,
    ) -> None:
        dist = net.masked_distribution(obs_batch, mask_all)
        action = dist.sample()
        assert action.shape == (4,)

    def test_log_prob_shape(
        self,
        net: ActorCritic,
        obs_batch: torch.Tensor,
        mask_all: torch.Tensor,
    ) -> None:
        dist = net.masked_distribution(obs_batch, mask_all)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        assert log_prob.shape == (4,)

    def test_log_prob_is_finite(
        self,
        net: ActorCritic,
        obs_batch: torch.Tensor,
        mask_all: torch.Tensor,
    ) -> None:
        dist = net.masked_distribution(obs_batch, mask_all)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        assert torch.isfinite(log_prob).all()

    def test_illegal_actions_have_zero_prob(
        self,
        net: ActorCritic,
        obs_batch: torch.Tensor,
        mask_single: torch.Tensor,
    ) -> None:
        """Masked-out actions must have probability 0."""
        dist = net.masked_distribution(obs_batch, mask_single)
        probs = dist.probs  # (4, ACTION_SIZE)
        # All actions except index 0 must have prob ≈ 0
        assert probs[:, 1:].sum() < 1e-6

    def test_orthogonal_init_actor_head(self, net: ActorCritic) -> None:
        """Actor head weights must have small norm (gain=0.01)."""
        weight_norm = net.actor_head.weight.norm().item()
        # With orthogonal init gain=0.01, norm should be much smaller than 1
        assert weight_norm < 5.0

    def test_network_on_correct_device(
        self, net: ActorCritic, device: torch.device
    ) -> None:
        for param in net.parameters():
            assert param.device.type == device.type

    def test_gradients_flow_through_full_network(
        self,
        net: ActorCritic,
        obs_batch: torch.Tensor,
        mask_all: torch.Tensor,
    ) -> None:
        """End-to-end gradient check."""
        logits, value = net(obs_batch)
        loss = logits.sum() + value.sum()
        loss.backward()
        for name, param in net.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_parameter_count_reasonable(self, net: ActorCritic) -> None:
        """Network should have between 1M and 20M parameters."""
        n_params = sum(p.numel() for p in net.parameters())
        assert 1_000_000 < n_params < 20_000_000, (
            f"Unexpected param count: {n_params:,}"
        )
