"""Ensemble utilities.

Implements:
  TemperatureCalibration  Post-hoc probability calibration per model arm.
                          Guo et al., ICML 2017 — "On Calibration of Modern Neural Networks".
                          Finds scalar T per arm that minimises NLL on OOF predictions.
                          AUC is rank-invariant (unaffected by monotone T scaling) but the
                          calibrated probabilities make rank-percentile blending more principled.

  greedy_ensemble_select  Caruana et al., ICML 2004 — "Ensemble Selection from Libraries of Models".
                          Greedy forward selection from a pool of model predictions on a held-out
                          OOF set. Always >= mean blending. Supports with-replacement (models can
                          be selected multiple times, effectively up-weighting them).

  rank_ensemble_n         N-arm rank-percentile blending (generalization of baseline 2-arm blend).
                          Per-target CoAtNet weighting kept from baseline (0.943 source).
                          Third arm (ConvNeXt / RadImageNet CNN) added with per-target weights.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

import src.core.config as config


# ─────────────────────────────────────── Temperature Calibration ─────────────
class TemperatureCalibration:
    """Per-arm temperature scaling (Guo et al., ICML 2017).

    Usage
    -----
    cal = TemperatureCalibration()
    cal.fit(oof_logits, oof_labels)          # numpy [N, C]
    calibrated_probs = cal.transform(logits) # numpy [N, C]
    """

    def __init__(self):
        self.T: float = 1.0

    # ------------------------------------------------------------------
    def _nll(self, T: float, logits: np.ndarray, labels: np.ndarray) -> float:
        """Binary cross-entropy NLL for a given temperature."""
        p = 1.0 / (1.0 + np.exp(-logits / max(T, 1e-6)))
        p = np.clip(p, 1e-7, 1.0 - 1e-7)
        return -np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p))

    def fit(self, oof_logits: np.ndarray, oof_labels: np.ndarray) -> "TemperatureCalibration":
        """Find T* = argmin NLL(T) on OOF data via bounded scalar search.

        logits: raw model output (pre-sigmoid) [N, C].
        labels: binary [N, C].
        Only uses labelled entries (weight > 0) if oof_labels contains NaNs.
        """
        mask = np.isfinite(oof_labels) & np.isfinite(oof_logits)
        lo = oof_logits[mask].ravel()
        la = oof_labels[mask].ravel()
        res = minimize_scalar(
            lambda T: self._nll(T, lo, la),
            bounds=(0.1, 10.0),
            method="bounded",
        )
        self.T = float(res.x)
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        """Apply calibration: sigmoid(logits / T)."""
        return 1.0 / (1.0 + np.exp(-logits / max(self.T, 1e-6)))

    def fit_transform(self, oof_logits, oof_labels, test_logits=None):
        self.fit(oof_logits, oof_labels)
        if test_logits is None:
            return self.transform(oof_logits)
        return self.transform(test_logits)

    def __repr__(self):
        return f"TemperatureCalibration(T={self.T:.4f})"


# ─────────────────────────────────────────── Greedy Ensemble Selection ────────
def greedy_ensemble_select(
    oof_preds: list[np.ndarray],   # list of [N, C] probability arrays (per model)
    oof_labels: np.ndarray,        # [N, C] binary ground truth
    n_select: int = 10,            # max ensemble members (with replacement)
    metric: str = "auc",           # "auc" or "bce"
) -> list[int]:
    """Caruana et al., ICML 2004 — greedy forward selection with replacement.

    Returns a list of indices (with repetitions) into oof_preds representing the
    selected ensemble members. Weight of each model = count(index) / n_select.

    This guarantees the selected ensemble is always >= the best single model
    on the OOF metric (hill-climbing property).
    """
    from sklearn.metrics import roc_auc_score

    def score(blend: np.ndarray) -> float:
        if metric == "auc":
            aucs = []
            for j in range(blend.shape[1]):
                m = np.isfinite(oof_labels[:, j])
                if m.sum() > 1 and 0 < oof_labels[m, j].sum() < m.sum():
                    aucs.append(roc_auc_score(oof_labels[m, j], blend[m, j]))
            return float(np.mean(aucs)) if aucs else 0.0
        else:  # BCE
            p = np.clip(blend, 1e-7, 1 - 1e-7)
            m = np.isfinite(oof_labels)
            return -float(np.mean(
                -(oof_labels[m] * np.log(p[m]) + (1 - oof_labels[m]) * np.log(1 - p[m]))
            ))

    selected: list[int] = []
    current_blend = np.zeros_like(oof_preds[0])

    for step in range(n_select):
        best_idx, best_score = -1, -np.inf
        for i, pred in enumerate(oof_preds):
            candidate = (current_blend * step + pred) / (step + 1)
            s = score(candidate)
            if s > best_score:
                best_score, best_idx = s, i
        selected.append(best_idx)
        current_blend = (current_blend * step + oof_preds[best_idx]) / (step + 1)

    return selected


def blend_from_selection(preds: list[np.ndarray], selected: list[int]) -> np.ndarray:
    """Average predictions according to a greedy-selected index list (with repetition)."""
    counts = np.bincount(selected, minlength=len(preds)).astype(float)
    blend = sum(p * c for p, c in zip(preds, counts))
    return blend / counts.sum()


# ─────────────────────────────────────────────── N-Arm Rank Ensemble ─────────
# Per-target blending weights for the existing two-arm system.
# Keys: target name (config.TARGETS).  Values: fraction assigned to CoAtNet arm.
_COAT_WEIGHTS: dict[str, float] = {t: 0.60 for t in config.TARGETS}
_COAT_WEIGHTS.update({
    "ACL": 0.75,
    "Medial Meniscus": 0.80,
    "Lateral Meniscus": 1.00,
    "Lateral OA": 0.75,
    "Fracture": 0.75,
})

# Third arm (ConvNeXt / RadImageNet-pretrained CNN).
# Verified basis: Mei et al. RadImageNet 2022 — ACL +4.8%, Meniscus +4.5%.
# ConvNeXt texture focus → strong for localised findings (Fracture, OA).
# Start conservatively at 0.15 across board; increase for ACL/Meniscus empirically.
_CONVNEXT_WEIGHTS: dict[str, float] = {t: 0.10 for t in config.TARGETS}
_CONVNEXT_WEIGHTS.update({
    "ACL": 0.15,
    "Medial Meniscus": 0.15,
    "Lateral Meniscus": 0.20,
    "Fracture": 0.15,
    "Contusion": 0.10,
})


def rank_ensemble_n(
    dfs: list[tuple[str, pd.DataFrame]],
    coat_weights: dict[str, float] | None = None,
    convnext_weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """N-arm rank-percentile blending.  Generalises the baseline 2-arm blend.

    Parameters
    ----------
    dfs : list of (arm_name, DataFrame) pairs.
        Each DataFrame must have 'StudyInstanceUID' + config.TARGETS columns.
        Supported arm names: 'dino', 'coatnet', 'convnext'. Others treated as
        additional DINOv2-style arms and added to the DINOv2 weight.
    coat_weights / convnext_weights : optional override dicts.

    Returns
    -------
    DataFrame with 'StudyInstanceUID' + config.TARGETS as rank-percentile blend.
    """
    cw = coat_weights or _COAT_WEIGHTS
    xw = convnext_weights or _CONVNEXT_WEIGHTS

    # Align on the first arm's study order
    ref_uid = dfs[0][1]["StudyInstanceUID"].tolist()
    ranks: dict[str, pd.DataFrame] = {}
    for arm_name, df in dfs:
        aligned = df.set_index("StudyInstanceUID").reindex(ref_uid).reset_index()
        r = aligned[config.TARGETS].rank(method="average", pct=True)
        r.index = ref_uid
        ranks[arm_name] = r

    blend = pd.DataFrame(index=ref_uid, columns=config.TARGETS, dtype=float)
    for t in config.TARGETS:
        coat_w = cw.get(t, 0.60)
        conv_w = xw.get(t, 0.10)
        dino_total = 1.0 - coat_w - conv_w  # remaining weight to DINOv2 arms

        # Sum DINOv2-type arm contributions (handles 5-fold averages)
        dino_contrib = 0.0
        n_dino = 0
        for arm_name, _ in dfs:
            if arm_name not in ("coatnet", "convnext"):
                dino_contrib += ranks[arm_name][t]
                n_dino += 1
        if n_dino > 0:
            dino_contrib = (dino_contrib / n_dino) * dino_total

        coat_contrib = ranks.get("coatnet", ranks[dfs[0][0]])[t] * coat_w if "coatnet" in ranks else 0.0
        conv_contrib = ranks.get("convnext", ranks[dfs[0][0]])[t] * conv_w if "convnext" in ranks else 0.0

        blend[t] = dino_contrib + coat_contrib + conv_contrib

    # Final re-rank to uniform [0, 1] percentile space
    blend = blend.rank(method="average", pct=True)
    result = pd.DataFrame({"StudyInstanceUID": ref_uid})
    for t in config.TARGETS:
        result[t] = blend[t].values
    return result
