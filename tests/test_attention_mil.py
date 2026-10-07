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
    assert restored_logits.shape == expected_logits.shape
    torch.testing.assert_close(restored_logits, expected_logits, rtol=1e-4, atol=1e-5)
