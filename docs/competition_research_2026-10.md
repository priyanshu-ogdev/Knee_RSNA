# RSNA Knee competition research and upgrade decision

## Evidence reviewed

The competition page confirms a notebook-based, offline inference task with
study-level predictions for twelve findings and macro ROC-AUC. Public material
reviewed on 2026-10-11 included:

- [Public DINO/RadImageNet rank ensemble](https://www.kaggle.com/code/tonylica/rsna-knee-dino-radimagenet-rank-ensemble)
- [Public DINO + dual CoAtNet notebook](https://www.kaggle.com/code/anthonytherrien/rsna-knee-dino-radimagenet-dual-coatnet)
- [Public CoAtNet fine-tune blend](https://www.kaggle.com/code/paiky1995/rsna-knee-0-926-lb-coatnet-fine-tune-blend)
- [Public DINO weak-label notebook](https://www.kaggle.com/code/renta0426/rsna-knee-0-937-weak-label-dinov2-meniscus-resid)
- [Public ensemble notebook](https://www.kaggle.com/code/pranjalverma08/rsna-knee-climbing-up-the-hill-the-ensemble-way)
- [Independent reproduction and experiment ledger](https://github.com/TranBaDat2607/RSNA-Knee-Abnormality-Detection)
- [Public frontier reproduction with pinned evidence](https://github.com/willmurray9/kaggle-rsna-knee-abnormality-detection)

Public repositories report a reproducible frontier around 0.939--0.943 from
rank ensembles combining public CoAtNet readers, DINOv2, RadImageNet and
Raptor-style arms. Individual public CoAtNet arms are reported around 0.917--
0.920, while weak-label DINO-only experiments are much lower on the 58-study
gold set. These are useful architectural signals, not proof of a 0.965 score:
public notebook scores, hidden weights, and leaderboard probing are not
independently reproducible from this checkout.

## Dataset implications

This repository's audit records 4,407 studies, 24,371 series, 819,078 slices,
and only 58 expert-labelled studies. Reports are available for training but
not test. The task is therefore primarily weak-label learning followed by
image-only test inference. The audit also found 21 site proxies, 57
cross-study identical-image groups, large per-series intensity variation,
non-square images, and 238 metadata-poor studies.

## Architecture decision

The existing DINOv2/slot-MIL and CoAtNet/timm families are retained. The
highest-confidence improvements are ensemble diversity and evaluation
integrity, not an unverified wholesale backbone replacement. In particular:

1. Keep independent DINOv2 and CoAtNet arms and compare flat versus
   hierarchical pooling on identical folds.
2. Use rank blending only after selecting arms/weights on held-out OOF data;
   do not import D4 weights as if they were validated for this graph.
3. Treat report-derived labels as weak supervision, never as validation truth.
4. Preserve unknown `-1` report states as masked labels rather than converting
   them using a global prevalence (which leaks label statistics across folds).
5. Report both site-stratified CV (competition-matched) and site-held-out CV
   (shortcut detection), plus per-target gold counts and bootstrap intervals.

## Changes made in this pass

- Pairwise AUC loss now masks NaN targets and uses confidence-weighted pairs.
- Label generation now marks mixed gold/pseudo rows as `mixed` and keeps `-1`
  uncertain report states unlabeled.
- Inference rank blending rejects non-finite predictions instead of silently
  assigning them the lowest rank.
- Prefetch worker exceptions are propagated to the consumer instead of causing
  an indefinite inference hang.

## Submission gate

Do not claim 0.966 from local validation alone. Before submission, run the
full preparation manifest, train independent seeds/families, export OOF,
select blend weights without reusing the final evaluation slice, run the
offline Kaggle notebook smoke test, and verify the final CSV schema and
finite predictions. The score target remains a hypothesis until a Kaggle
submission receipt verifies it.
