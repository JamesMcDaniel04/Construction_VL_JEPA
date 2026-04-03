"""Tests for the VL-JEPA model core."""

import torch
import pytest

from construction_vl_jepa.config import ModelConfig
from construction_vl_jepa.model.jepa import VLJEPA
from construction_vl_jepa.model.encoders import SensorPatchEncoder, PatchEmbedding
from construction_vl_jepa.model.predictor import JEPAPredictor


@pytest.fixture
def small_model_cfg():
    return ModelConfig(
        latent_dim=32,
        context_window=120,
        target_window=60,
        predictor_depth=2,
        encoder_depth=2,
        n_heads=4,
        dropout=0.0,
    )


@pytest.fixture
def small_model(small_model_cfg):
    return VLJEPA(small_model_cfg, n_sensors=4, sample_rate_hz=1.0)


class TestPatchEmbedding:
    def test_output_shape(self):
        embed = PatchEmbedding(patch_size=60, n_sensors=4, latent_dim=32)
        x = torch.randn(2, 3, 60, 4)  # (batch, n_patches, patch_size, n_sensors)
        out = embed(x)
        assert out.shape == (2, 3, 32)


class TestSensorPatchEncoder:
    def test_output_shape(self):
        encoder = SensorPatchEncoder(
            patch_size=60, n_sensors=4, latent_dim=32, depth=2, n_heads=4
        )
        x = torch.randn(2, 3, 60, 4)
        out = encoder(x)
        # Output includes CLS token: (batch, n_patches + 1, latent_dim)
        assert out.shape == (2, 4, 32)

    def test_with_mask(self):
        encoder = SensorPatchEncoder(
            patch_size=60, n_sensors=4, latent_dim=32, depth=2, n_heads=4
        )
        x = torch.randn(2, 3, 60, 4)
        mask = torch.tensor([[True, True, False], [True, False, True]])
        out = encoder(x, mask=mask)
        assert out.shape == (2, 4, 32)


class TestJEPAPredictor:
    def test_output_shape(self):
        predictor = JEPAPredictor(
            latent_dim=32, depth=2, n_heads=4, n_target_tokens=2
        )
        context = torch.randn(2, 4, 32)  # (batch, n_ctx_tokens, dim)
        out = predictor(context)
        assert out.shape == (2, 2, 32)


class TestVLJEPA:
    def test_forward_returns_loss(self, small_model):
        context = torch.randn(2, 2, 60, 4)  # 2 patches of 60 steps, 4 sensors
        target = torch.randn(2, 1, 60, 4)   # 1 target patch
        result = small_model(context, target)
        assert "loss" in result
        assert "prediction_error" in result
        assert result["loss"].shape == ()
        assert result["prediction_error"].shape == (2,)

    def test_loss_is_finite(self, small_model):
        context = torch.randn(2, 2, 60, 4)
        target = torch.randn(2, 1, 60, 4)
        result = small_model(context, target)
        assert torch.isfinite(result["loss"])

    def test_encode(self, small_model):
        x = torch.randn(2, 2, 60, 4)
        latent = small_model.encode(x)
        assert latent.shape == (2, 3, 32)  # 2 patches + CLS

    def test_predict_and_score(self, small_model):
        context = torch.randn(2, 2, 60, 4)
        target = torch.randn(2, 1, 60, 4)
        result = small_model.predict_and_score(context, target)
        assert "anomaly_score" in result
        assert "cls_embedding" in result
        assert result["anomaly_score"].shape == (2,)
        assert result["cls_embedding"].shape == (2, 32)

    def test_ema_update(self, small_model):
        # Check that EMA update changes target encoder params
        ctx_params_before = [p.clone() for p in small_model.target_encoder.parameters()]
        # Modify context encoder
        for p in small_model.context_encoder.parameters():
            p.data.add_(torch.randn_like(p) * 0.1)
        small_model.update_target_encoder(0.99)
        for p_before, p_after in zip(ctx_params_before, small_model.target_encoder.parameters()):
            assert not torch.allclose(p_before, p_after)

    def test_gradient_only_through_context_encoder(self, small_model):
        context = torch.randn(2, 2, 60, 4)
        target = torch.randn(2, 1, 60, 4)
        result = small_model(context, target)
        result["loss"].backward()

        # Context encoder should have gradients
        assert any(p.grad is not None for p in small_model.context_encoder.parameters())
        # Target encoder should NOT have gradients
        assert all(p.grad is None for p in small_model.target_encoder.parameters())
