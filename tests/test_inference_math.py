import numpy as np
import pytest
import torch
from torch import nn
from pytest import param

from src.core import config
from src.inference import inference


class _ConstantModel(nn.Module):
    def __init__(self, expected_device="cpu"):
        super().__init__()
        self.logits = nn.Parameter(torch.linspace(-1.0, 1.0, len(config.TARGETS)))
        self.expected_device = expected_device

    def forward(self, images, slot_mask, window_mask):
        assert images.device.type == self.expected_device
        return self.logits.unsqueeze(0).expand(images.shape[0], -1)


@pytest.mark.parametrize(
    ("device", "expected_device"),
    [("cpu", "cpu"), param("meta", "meta", id="non-cpu-transfer")],
)
def test_predict_chunk_transfers_samples_to_requested_device(
    monkeypatch, device, expected_device
):
    model = _ConstantModel(expected_device)

    def make_sample(cache, index, cfg, train=False, n_use=None):
        return (
            np.zeros((config.N_SLOTS, 1, 3, 4, 4), dtype=np.uint8),
            np.ones(config.N_SLOTS, dtype=np.uint8),
            np.ones((config.N_SLOTS, 1), dtype=bool),
        )

    monkeypatch.setattr(inference.loader, "make_sample", make_sample)
    predictions = inference.predict_chunk(
        models=[model],
        cache=object(),
        a=0,
        b=3,
        cfg=config.get_cfg("v2"),
        device=torch.device(device),
        batch=2,
    )

    assert predictions.shape == (3, len(config.TARGETS))
    assert np.isfinite(predictions).all()
    np.testing.assert_allclose(
        predictions[0],
        torch.sigmoid(model.logits.detach()).numpy(),
    )


def test_prediction_options_rejects_invalid_temperatures():
    model = _ConstantModel()

    with pytest.raises(ValueError, match="finite and positive"):
        inference._validate_prediction_options([model], [0.0], batch=1)

    with pytest.raises(ValueError, match="shape"):
        inference._validate_prediction_options([model], [np.ones(2)], batch=1)
