import json

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.core import config
from src.main import run_oof_and_checkpoint_verification
from src.training.train import evaluate


class _ImageScoreModel(nn.Module):
    def forward(self, images, slot_mask, window_mask):
        return images[:, :1].expand(-1, len(config.TARGETS))


def test_epoch_evaluation_never_falls_back_to_pseudo_labels(monkeypatch):
    from src.training import train as train_module

    monkeypatch.setattr(train_module, "check_memory_circuit_breaker", lambda **kwargs: None)
    labels = torch.zeros((4, len(config.TARGETS)))
    weights = torch.zeros_like(labels)
    labels[:2, 0] = torch.tensor([0.0, 1.0])
    weights[:2, 0] = 1.0
    labels[:2, 1] = 1.0
    weights[:2, 1] = 1.0
    labels[2:, 1] = torch.tensor([0.0, 1.0])
    weights[2:, 1] = 0.25
    dataset = TensorDataset(
        torch.tensor([[0.1], [0.9], [0.2], [0.8]]),
        torch.ones((4, 1)),
        torch.ones((4, 1)),
        labels,
        weights,
    )

    _, per_target = evaluate(_ImageScoreModel(), DataLoader(dataset, batch_size=4), torch.device("cpu"))

    assert per_target == {"ACL": 1.0}


def test_oof_report_uses_gold_only_and_best_checkpoints(tmp_path):
    model_dir = tmp_path / "models_fold0"
    model_dir.mkdir()
    best_checkpoint = model_dir / "fold0_best.pt"
    best_checkpoint.write_bytes(b"checkpoint")

    oof = pd.DataFrame({"StudyInstanceUID": ["a", "b"], "fold": [0, 0]})
    labels = pd.DataFrame({"StudyInstanceUID": ["a", "b"], "source": ["gold", "gold"]})
    for target in config.TARGETS:
        oof[f"pred_{target}"] = [0.1, 0.9]
        oof[f"target_{target}"] = [0.0, 0.0]
        oof[f"weight_{target}"] = [1.0, 1.0]
        labels[target] = [0.0, 0.0]
        labels[f"{target}_weight"] = [0.0, 0.0]
    oof["target_ACL"] = [0.0, 1.0]
    labels["ACL"] = [0.0, 1.0]
    labels["ACL_weight"] = [1.0, 1.0]
    oof.to_csv(model_dir / "fold0_oof.csv", index=False)
    labels.to_csv(tmp_path / "train_labels_v2.csv", index=False)
    folds = pd.DataFrame({"StudyInstanceUID": ["a", "b"], "fold": [0, 0]})
    folds_path = tmp_path / "folds.csv"
    folds.to_csv(folds_path, index=False)

    checkpoints, temperatures = run_oof_and_checkpoint_verification(
        str(tmp_path),
        str(tmp_path / "train_labels_v2.csv"),
        str(folds_path),
        [0],
        {0: 1.0},
    )

    assert checkpoints == [str(best_checkpoint)]
    assert temperatures == [1.0]
    metrics = json.loads((tmp_path / "oof_metrics.json").read_text())
    assert metrics["metric"] == "gold-only fold-held-out macro ROC-AUC"
    assert metrics["macro_auc"] == 1.0
    assert list(metrics["per_target"]) == ["ACL"]
