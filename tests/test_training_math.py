import torch
from torch import nn

from src.training import train as train_module


class _ScalarStudyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.logit = nn.Parameter(torch.tensor(0.0))

    def forward(self, images, slot_mask, window_mask):
        return self.logit.expand(images.shape[0], 1)


class _WeightedSquaredError(nn.Module):
    def forward(self, logits, targets, weights):
        return (((logits - targets) ** 2) * weights).mean()


class _NoOpScheduler:
    def step(self):
        pass


class _OverflowScaler:
    def __init__(self):
        self.scale_value = 8.0

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        pass

    def step(self, optimizer):
        pass

    def update(self):
        self.scale_value /= 2

    def get_scale(self):
        return self.scale_value

    def is_enabled(self):
        return True


class _CountingScheduler:
    def __init__(self):
        self.steps = 0

    def step(self):
        self.steps += 1


def test_partial_gradient_accumulation_uses_actual_group_size(monkeypatch):
    monkeypatch.setattr(train_module, "check_memory_circuit_breaker", lambda **kwargs: None)
    model = _ScalarStudyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    batch = (
        torch.zeros((1, 1)),
        torch.ones((1, 1)),
        torch.ones((1, 1, 1)),
        torch.full((1, 1), 0.1),
        torch.ones((1, 1)),
    )

    train_module.train_epoch(
        model=model,
        dataloader=[batch, batch],
        optimizer=optimizer,
        scaler=scaler,
        scheduler=_NoOpScheduler(),
        device=torch.device("cpu"),
        criterion=_WeightedSquaredError(),
        grad_accum=3,
        gold_weight_mult=1.0,
    )

    torch.testing.assert_close(model.logit.detach(), torch.tensor(0.02))


def test_scheduler_does_not_advance_when_grad_scaler_skips_optimizer_step(monkeypatch):
    monkeypatch.setattr(train_module, "check_memory_circuit_breaker", lambda **kwargs: None)
    model = _ScalarStudyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = _CountingScheduler()
    batch = (
        torch.zeros((1, 1)),
        torch.ones((1, 1)),
        torch.ones((1, 1, 1)),
        torch.full((1, 1), 0.1),
        torch.ones((1, 1)),
    )

    train_module.train_epoch(
        model=model,
        dataloader=[batch],
        optimizer=optimizer,
        scaler=_OverflowScaler(),
        scheduler=scheduler,
        device=torch.device("cpu"),
        criterion=_WeightedSquaredError(),
    )

    assert scheduler.steps == 0
    torch.testing.assert_close(model.logit.detach(), torch.tensor(0.0))
