import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from src.modeling.losses import AUCAsymmetricLoss, PairwiseAUC, build_loss


def test_pairwise_auc_masks_unknown_entries_and_has_finite_gradients():
    logits = torch.tensor([[2.0, 0.0], [-1.0, 1.0], [0.5, -0.5]], requires_grad=True)
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0], [float("nan"), 0.0]])
    weights = torch.tensor([[1.0, 1.0], [1.0, 1.0], [0.0, 1.0]])
    loss = PairwiseAUC()(logits, targets, weights)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_auc_asl_is_registered_and_combines_objectives():
    criterion = build_loss("auc_asl", rank_weight=0.2)
    assert isinstance(criterion, AUCAsymmetricLoss)
    logits = torch.zeros(4, 2, requires_grad=True)
    targets = torch.tensor([[1., 0.], [0., 1.], [1., 0.], [0., 1.]])
    weights = torch.ones_like(targets)
    value = criterion(logits, targets, weights)
    assert torch.isfinite(value)
    value.backward()
    assert torch.isfinite(logits.grad).all()


if __name__ == "__main__":
    test_pairwise_auc_masks_unknown_entries_and_has_finite_gradients()
    test_auc_asl_is_registered_and_combines_objectives()
    print("All AUC loss tests passed successfully!")
