import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.optim.swa_utils import AveragedModel

from src.core import config
from src.modeling.mil import build_timm_attention_mil
from src.modeling.model import AutoConfig, WindowPool, build_model, load_checkpoint


@pytest.fixture
def tiny_dino():
    torch.manual_seed(42)
    model_config = {
        "model_type": "dinov2",
        "variant": "dinov2-base",
        "pretrained": False,
        "backbone_config": AutoConfig.for_model(
            "dinov2", hidden_size=16, num_hidden_layers=1,
            num_attention_heads=4, image_size=16, patch_size=8, mlp_ratio=2,
        ).to_dict(),
        "use_cross_slot": True,
        "unfreeze_last": 1,
        "lora_rank": 0,
        "grad_checkpoint": False,
        "encode_chunk_size": 8,
    }
    return build_model(**model_config).eval(), model_config


def test_dino_temperature_round_trip_and_best_reload(tmp_path, tiny_dino):
    model, model_config = tiny_dino
    images = torch.randint(
        0, 256, (2, config.N_SLOTS, 3, 3, 16, 16), dtype=torch.uint8
    )
    slot_mask = torch.zeros(2, config.N_SLOTS)
    slot_mask[0, :2] = 1
    window_mask = torch.zeros(2, config.N_SLOTS, 3)
    window_mask[0, :2, :2] = 1
    model.wpool.set_temperature(0.4)
    with torch.no_grad():
        expected = model(images, slot_mask, window_mask).sigmoid()
    assert torch.isfinite(expected).all()  # Includes a fully masked study.

    checkpoint = tmp_path / "best.pt"
    torch.save({"model": model.state_dict(), "model_config": model_config}, checkpoint)
    restored = load_checkpoint(str(checkpoint))
    assert restored.wpool.temperature.item() == pytest.approx(0.4)
    with torch.no_grad():
        actual = restored(images, slot_mask, window_mask).sigmoid()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    # Reproduce the training/OOF flow: a later epoch changes temperature and
    # weights, then loads the selected best state into the same model.
    model.wpool.set_temperature(1.0)
    with torch.no_grad():
        other_temperature = model(images, slot_mask, window_mask).sigmoid()
        model.head.out.bias.add_(1.0)
    assert (other_temperature - expected).abs().max().item() > 1e-6
    best = torch.load(checkpoint, weights_only=False)
    model.load_state_dict(best["model"], strict=True)
    assert model.wpool.temperature.item() == pytest.approx(0.4)
    with torch.no_grad():
        reloaded = model(images, slot_mask, window_mask).sigmoid()
        changed_padding = images.clone()
        changed_padding[window_mask == 0] = 255
        masked = model(changed_padding, slot_mask, window_mask).sigmoid()
    torch.testing.assert_close(reloaded, expected, rtol=0, atol=0)
    torch.testing.assert_close(masked, expected, rtol=0, atol=0)


def test_legacy_temperature_warns_and_defaults_without_allowing_other_missing_keys(
    tmp_path, tiny_dino
):
    model, model_config = tiny_dino
    state = model.state_dict()
    state.pop("wpool.temperature")
    path = tmp_path / "legacy.pt"
    torch.save({"model": state, "model_config": model_config}, path)
    with pytest.warns(UserWarning, match="Exact old prediction reproduction cannot be guaranteed"):
        restored = load_checkpoint(str(path))
    assert restored.wpool.temperature.item() == 1.0

    state.pop("wpool.attention_weights.weight")
    torch.save({"model": state, "model_config": model_config}, path)
    with pytest.warns(UserWarning, match="wpool.temperature"):
        with pytest.raises(RuntimeError, match="wpool.attention_weights.weight"):
            load_checkpoint(str(path))


def test_temperature_setter_preserves_buffer_and_swa_copies_latest():
    pool = WindowPool(8)
    buffer = pool.temperature
    assert buffer.shape == torch.Size([])
    assert buffer.item() == 1.0
    swa = AveragedModel(pool, use_buffers=False)
    for temperature in (0.8, 0.4):
        pool.set_temperature(temperature)
        assert pool.temperature is buffer
        swa.update_parameters(pool)
        assert swa.module.temperature.item() == pytest.approx(temperature)
        assert swa.module.state_dict()["temperature"].item() == pytest.approx(temperature)
    assert swa.module.temperature is not pool.temperature


@pytest.mark.parametrize("temperature", [0, -1, float("nan"), float("inf")])
def test_temperature_setter_rejects_invalid_values(temperature):
    pool = WindowPool(8)
    with pytest.raises(ValueError, match="finite and positive"):
        pool.set_temperature(temperature)
    assert pool.temperature.item() == 1.0


@pytest.mark.parametrize("grad_checkpoint", [True, False])
def test_timm_factory_receives_grad_checkpoint_flag(monkeypatch, grad_checkpoint):
    class Backbone(nn.Module):
        num_features = 8

        def set_grad_checkpointing(self, enabled):
            self.checkpointing = enabled

    monkeypatch.setitem(sys.modules, "timm", SimpleNamespace(
        create_model=lambda *args, **kwargs: Backbone()
    ))
    model = build_model(
        variant="tiny", model_type="timm_mil", pretrained=False,
        grad_checkpoint=grad_checkpoint,
    )
    assert model.backbone.checkpointing is grad_checkpoint
    # Existing direct callers retain the previous checkpointing default.
    assert build_timm_attention_mil("tiny").backbone.checkpointing is True
