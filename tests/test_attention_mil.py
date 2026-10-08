import pytest
import torch

from src.core import config
from src.modeling.model import build_model, load_checkpoint


def test_timm_attention_mil_forward_backward_and_checkpoint_round_trip(tmp_path):
    model = build_model(
        variant="resnet18",
        model_type="coatnet_mil",
        input_size=32,
        encode_chunk_size=2,
        pretrained=False,
    )
    model.train()
    images = torch.randint(0, 256, (2, 6, 2, 3, 40, 40), dtype=torch.uint8)
    slot_mask = torch.zeros((2, 6))
    slot_mask[0, :2] = 1
    window_mask = torch.zeros((2, 6, 2))
    window_mask[0, :2, :] = 1

    logits = model(images, slot_mask, window_mask)
    assert logits.shape == (2, len(config.TARGETS))
    assert torch.isfinite(logits).all()
    logits.square().mean().backward()
    assert any(parameter.grad is not None for parameter in model.backbone.parameters())
    assert model.classifier.grad is not None
    model.eval()
    with torch.no_grad():
        expected_logits = model(images, slot_mask, window_mask)

    checkpoint = tmp_path / "mil.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "model_type": "coatnet_mil",
            "model_config": {
                "model_type": "coatnet_mil",
                "variant": "resnet18",
                "input_size": 32,
                "encode_chunk_size": 2,
                "pretrained": False,
            },
            "cfg": config.get_cfg("v2").__dict__,
        },
        checkpoint,
    )
    loaded = load_checkpoint(str(checkpoint))
    loaded.eval()
    with torch.no_grad():
        restored_logits = loaded(images, slot_mask, window_mask)
    assert loaded.pooling_mode == "flat"
    assert restored_logits.shape == expected_logits.shape
    torch.testing.assert_close(restored_logits, expected_logits, rtol=1e-4, atol=1e-5)


def test_hierarchical_timm_mil_masks_slots_and_round_trips_checkpoint(tmp_path):
    model = build_model(
        variant="resnet18",
        model_type="timm_mil",
        input_size=32,
        encode_chunk_size=2,
        timm_pooling="hierarchical",
        pretrained=False,
    )
    model.train()
    images = torch.randint(0, 256, (2, 6, 2, 3, 40, 40), dtype=torch.uint8)
    slot_mask = torch.zeros((2, 6))
    slot_mask[0, :2] = 1
    window_mask = torch.zeros((2, 6, 2))
    window_mask[0, :2, :] = 1

    logits = model(images, slot_mask, window_mask)
    assert logits.shape == (2, len(config.TARGETS))
    assert torch.isfinite(logits).all()
    logits.square().mean().backward()
    for name in ("slot_embedding", "slot_query", "slot_bias"):
        assert getattr(model, name).grad is not None

    model.eval()
    with torch.no_grad():
        expected_logits = model(images, slot_mask, window_mask)
        padded_images = torch.cat(
            (images, torch.randint(0, 256, (2, 6, 1, 3, 40, 40), dtype=torch.uint8)),
            dim=2,
        )
        padded_window_mask = torch.cat(
            (window_mask, torch.zeros((2, 6, 1))), dim=2
        )
        padded_logits = model(padded_images, slot_mask, padded_window_mask)
    torch.testing.assert_close(padded_logits, expected_logits, rtol=1e-4, atol=1e-5)

    checkpoint = tmp_path / "hierarchical_mil.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "model_type": "timm_mil",
            "model_config": {
                "model_type": "timm_mil",
                "variant": "resnet18",
                "input_size": 32,
                "encode_chunk_size": 2,
                "timm_pooling": "hierarchical",
                "pretrained": False,
            },
            "cfg": config.get_cfg("v2").__dict__,
        },
        checkpoint,
    )
    loaded = load_checkpoint(str(checkpoint))
    with torch.no_grad():
        restored_logits = loaded(images, slot_mask, window_mask)
    assert loaded.pooling_mode == "hierarchical"
    torch.testing.assert_close(restored_logits, expected_logits, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("pooling_mode", ["flat", "hierarchical"])
def test_timm_mil_returns_bias_for_study_with_no_valid_windows(pooling_mode):
    model = build_model(
        variant="resnet18",
        model_type="timm_mil",
        input_size=32,
        encode_chunk_size=2,
        timm_pooling=pooling_mode,
        pretrained=False,
    )
    images = torch.randint(0, 256, (1, 6, 2, 3, 40, 40), dtype=torch.uint8)
    slot_mask = torch.zeros((1, 6))
    window_mask = torch.zeros((1, 6, 2))

    with torch.no_grad():
        logits = model(images, slot_mask, window_mask)

    torch.testing.assert_close(
        logits, model.bias.unsqueeze(0).expand_as(logits)
    )


def test_cross_slot_transformer_handles_study_with_no_valid_slots():
    from src.modeling.model import CrossSlotTransformer

    model = CrossSlotTransformer(dim=32, nhead=4)
    features = torch.randn(2, config.N_SLOTS, 32)
    slot_mask = torch.zeros((2, config.N_SLOTS))
    slot_mask[1, 0] = 1

    output = model(features, slot_mask)

    assert torch.isfinite(output).all()
    assert torch.count_nonzero(output[0]) == 0
