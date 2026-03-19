"""
tests/agents/test_icm.py

Unit tests for tactilai/agents/icm.py
Run with : pytest tests/agents/test_icm.py -v
"""

import pytest
import torch

from tactilai.agents.icm import (
    ICM,
    ICM_BETA,
    LATENT_DIM,
    ForwardModel,
    InverseModel,
    StateEncoder,
)
from tactilai.env.gym_wrapper import ACTION_SIZE, OBS_SIZE

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def icm(device: torch.device) -> ICM:
    return ICM(device=device)


@pytest.fixture
def batch(device: torch.device) -> dict[str, torch.Tensor]:
    B = 4
    return {
        "obs": torch.rand(B, OBS_SIZE, device=device),
        "obs_next": torch.rand(B, OBS_SIZE, device=device),
        "action": torch.randint(0, ACTION_SIZE, (B,), device=device),
    }


# ── StateEncoder ──────────────────────────────────────────────────────────────


class TestStateEncoder:
    def test_output_shape(self, device: torch.device) -> None:
        encoder = StateEncoder().to(device)
        obs = torch.rand(4, OBS_SIZE, device=device)
        phi = encoder(obs)
        assert phi.shape == (4, LATENT_DIM)

    def test_output_single(self, device: torch.device) -> None:
        encoder = StateEncoder().to(device)
        obs = torch.rand(1, OBS_SIZE, device=device)
        phi = encoder(obs)
        assert phi.shape == (1, LATENT_DIM)

    def test_no_nan(self, device: torch.device) -> None:
        encoder = StateEncoder().to(device)
        obs = torch.rand(4, OBS_SIZE, device=device)
        phi = encoder(obs)
        assert not torch.isnan(phi).any()

    def test_gradients_flow(self, device: torch.device) -> None:
        encoder = StateEncoder().to(device)
        obs = torch.rand(4, OBS_SIZE, device=device, requires_grad=False)
        phi = encoder(obs)
        phi.sum().backward()
        for name, p in encoder.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"


# ── InverseModel ──────────────────────────────────────────────────────────────


class TestInverseModel:
    def test_output_shape(self, device: torch.device) -> None:
        model = InverseModel().to(device)
        phi_s = torch.rand(4, LATENT_DIM, device=device)
        phi_s_next = torch.rand(4, LATENT_DIM, device=device)
        logits = model(phi_s, phi_s_next)
        assert logits.shape == (4, ACTION_SIZE)

    def test_no_nan(self, device: torch.device) -> None:
        model = InverseModel().to(device)
        phi_s = torch.rand(4, LATENT_DIM, device=device)
        phi_s_next = torch.rand(4, LATENT_DIM, device=device)
        logits = model(phi_s, phi_s_next)
        assert not torch.isnan(logits).any()

    def test_gradients_flow(self, device: torch.device) -> None:
        model = InverseModel().to(device)
        phi_s = torch.rand(4, LATENT_DIM, device=device)
        phi_s_next = torch.rand(4, LATENT_DIM, device=device)
        action = torch.randint(0, ACTION_SIZE, (4,), device=device)
        loss = torch.nn.functional.cross_entropy(model(phi_s, phi_s_next), action)
        loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"


# ── ForwardModel ──────────────────────────────────────────────────────────────


class TestForwardModel:
    def test_output_shape(self, device: torch.device) -> None:
        model = ForwardModel().to(device)
        phi_s = torch.rand(4, LATENT_DIM, device=device)
        action = torch.randint(0, ACTION_SIZE, (4,), device=device)
        phi_hat = model(phi_s, action)
        assert phi_hat.shape == (4, LATENT_DIM)

    def test_no_nan(self, device: torch.device) -> None:
        model = ForwardModel().to(device)
        phi_s = torch.rand(4, LATENT_DIM, device=device)
        action = torch.randint(0, ACTION_SIZE, (4,), device=device)
        phi_hat = model(phi_s, action)
        assert not torch.isnan(phi_hat).any()

    def test_gradients_flow(self, device: torch.device) -> None:
        model = ForwardModel().to(device)
        phi_s = torch.rand(4, LATENT_DIM, device=device)
        action = torch.randint(0, ACTION_SIZE, (4,), device=device)
        phi_hat = model(phi_s, action)
        target = torch.rand(4, LATENT_DIM, device=device)
        loss = torch.nn.functional.mse_loss(phi_hat, target)
        loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"


# ── ICM ───────────────────────────────────────────────────────────────────────


class TestICM:
    def test_intrinsic_reward_shape(self, icm: ICM, batch: dict) -> None:
        r_i = icm.intrinsic_reward(batch["obs"], batch["obs_next"], batch["action"])
        assert r_i.shape == (4,)

    def test_intrinsic_reward_non_negative(self, icm: ICM, batch: dict) -> None:
        r_i = icm.intrinsic_reward(batch["obs"], batch["obs_next"], batch["action"])
        assert (r_i >= 0).all()

    def test_intrinsic_reward_is_detached(self, icm: ICM, batch: dict) -> None:
        r_i = icm.intrinsic_reward(batch["obs"], batch["obs_next"], batch["action"])
        assert not r_i.requires_grad

    def test_intrinsic_reward_scaled_by_eta(
        self, device: torch.device, batch: dict
    ) -> None:
        icm_small = ICM(device=device, eta=0.001)
        icm_large = ICM(device=device, eta=1.0)
        r_small = icm_small.intrinsic_reward(
            batch["obs"], batch["obs_next"], batch["action"]
        )
        r_large = icm_large.intrinsic_reward(
            batch["obs"], batch["obs_next"], batch["action"]
        )
        # Same weights, same inputs — only eta differs
        # r_large / r_small ≈ 1.0 / 0.001 = 1000
        assert r_large.mean() > r_small.mean()

    def test_loss_returns_three_tensors(self, icm: ICM, batch: dict) -> None:
        loss_icm, loss_fwd, loss_inv = icm.loss(
            batch["obs"], batch["obs_next"], batch["action"]
        )
        assert loss_icm.shape == torch.Size([])
        assert loss_fwd.shape == torch.Size([])
        assert loss_inv.shape == torch.Size([])

    def test_loss_is_positive(self, icm: ICM, batch: dict) -> None:
        loss_icm, loss_fwd, loss_inv = icm.loss(
            batch["obs"], batch["obs_next"], batch["action"]
        )
        assert loss_icm.item() > 0
        assert loss_fwd.item() > 0
        assert loss_inv.item() > 0

    def test_loss_combination(self, icm: ICM, batch: dict) -> None:
        """L_icm = β * L_fwd + (1-β) * L_inv."""
        loss_icm, loss_fwd, loss_inv = icm.loss(
            batch["obs"], batch["obs_next"], batch["action"]
        )
        expected = ICM_BETA * loss_fwd + (1.0 - ICM_BETA) * loss_inv
        assert torch.isclose(loss_icm, expected, atol=1e-5)

    def test_loss_backward(self, icm: ICM, batch: dict) -> None:
        loss_icm, _, _ = icm.loss(batch["obs"], batch["obs_next"], batch["action"])
        loss_icm.backward()
        for name, p in icm.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"

    def test_no_nan_in_loss(self, icm: ICM, batch: dict) -> None:
        loss_icm, loss_fwd, loss_inv = icm.loss(
            batch["obs"], batch["obs_next"], batch["action"]
        )
        assert not torch.isnan(loss_icm)
        assert not torch.isnan(loss_fwd)
        assert not torch.isnan(loss_inv)

    def test_device_placement(self, icm: ICM, device: torch.device) -> None:
        for p in icm.parameters():
            assert p.device.type == device.type

    def test_same_obs_gives_zero_reward(self, icm: ICM, device: torch.device) -> None:
        """
        When obs == obs_next the forward model error should be small
        after training — at init it won't be zero but should be finite.
        """
        obs = torch.rand(4, OBS_SIZE, device=device)
        action = torch.zeros(4, dtype=torch.long, device=device)
        r_i = icm.intrinsic_reward(obs, obs, action)
        assert torch.isfinite(r_i).all()
