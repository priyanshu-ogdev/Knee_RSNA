import torch

from src.modeling.losses import AsymmetricLoss


def test_asymmetric_loss_uses_shifted_negative_probability_for_focusing():
    p = torch.tensor([[0.2]], dtype=torch.float32)
    logits = torch.logit(p)
    targets = torch.zeros_like(p)
    clip = 0.05
    gamma_neg = 2.0

    loss = AsymmetricLoss(gamma_neg=gamma_neg, clip=clip)(logits, targets)

    shifted_negative_probability = min(1.0, 1.0 - p.item() + clip)
    expected = -torch.log(torch.tensor(shifted_negative_probability)) * (
        1.0 - shifted_negative_probability
    ) ** gamma_neg
    torch.testing.assert_close(loss, expected)


def test_asymmetric_loss_masks_unlabeled_entries_and_has_finite_gradients():
    logits = torch.tensor([[0.2, -0.4], [1.0, -1.0]], requires_grad=True)
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    weights = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    loss = AsymmetricLoss(gamma_neg=[2.0, 3.0])(logits, targets, weights)
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, 1] == 0
    assert logits.grad[1, 0] == 0
