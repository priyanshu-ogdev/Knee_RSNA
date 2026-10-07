import inspect

import torch
from torch import nn

import src.core.config as config
import src.inference.inference as inference
import src.modeling.model as model_module


def test_build_model_default_lora_rank_matches_training_config():
    assert inspect.signature(model_module.build_model).parameters["lora_rank"].default == config.LORA_RANK


def test_load_checkpoint_infers_legacy_lora_rank_and_retains_preprocessing_config(
    tmp_path, monkeypatch
):
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.adapter = nn.Module()
            self.adapter.register_parameter("A", nn.Parameter(torch.ones(3, 4)))

    saved_cfg = {
        "name": "v2",
        "crop_mm": 130.0,
        "img_size": 336,
        "pad_mode": "border_median",
        "resize_mode": "area",
        "wide_center": True,
        "wide_ratio": 1.3,
        "norm_lo": 1.0,
        "norm_hi": 99.5,
        "clip_negative": True,
        "order_mode": "position",
        "z_mode": "mm",
        "stack_depth": 24,
        "z_step_mm": 4.0,
        "band": [0.2, 0.8],
        "group": 3,
        "win_stride": 2,
        "reorient": True,
        "lat_canon": True,
        "slot_prefer_2d": True,
        "slot_fs_priority": "csv",
        "slot_csv_fallback": True,
    }
    checkpoint = tmp_path / "legacy.pt"
    torch.save({"model": TinyModel().state_dict(), "cfg": saved_cfg}, checkpoint)
    received = {}

    def fake_build_model(**kwargs):
        received.update(kwargs)
        return TinyModel()

    monkeypatch.setattr(model_module, "build_model", fake_build_model)
    loaded = model_module.load_checkpoint(str(checkpoint))

    assert received["lora_rank"] == 4
    assert loaded._rsna_preprocessing_config == saved_cfg


def test_resolve_preprocessing_config_uses_and_checks_checkpoint_metadata():
    class Model:
        _rsna_preprocessing_config = {
            "name": "v2",
            "crop_mm": 130.0,
            "img_size": 336,
            "pad_mode": "border_median",
            "resize_mode": "area",
            "wide_center": True,
            "wide_ratio": 1.3,
            "norm_lo": 1.0,
            "norm_hi": 99.5,
            "clip_negative": True,
            "order_mode": "position",
            "z_mode": "mm",
            "stack_depth": 24,
            "z_step_mm": 4.0,
            "band": [0.2, 0.8],
            "group": 3,
            "win_stride": 2,
            "reorient": True,
            "lat_canon": True,
            "slot_prefer_2d": True,
            "slot_fs_priority": "csv",
            "slot_csv_fallback": True,
        }

    model = Model()
    expected = config.cfg_from_dict(model._rsna_preprocessing_config)
    assert inference.resolve_preprocessing_config([model], None) == expected
    try:
        inference.resolve_preprocessing_config([model], config.get_cfg("v2"))
    except ValueError as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError("mismatched explicit preprocessing config was accepted")
