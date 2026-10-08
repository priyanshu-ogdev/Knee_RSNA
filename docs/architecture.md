# RSNA Knee Classification Architecture

This describes the implementation in `src/`. Training details and experiment
limitations are documented in [`ml_model_design.md`](ml_model_design.md).

## Data and labels

The pipeline routes each study into six anatomical MRI slots. The `v2` cache
uses 518-pixel images, stack depth 32, and windows built from three adjacent
grayscale slices as the model's three input channels. Training samples up to
five windows per slot; validation and inference use all available windows.
Only valid windows are encoded.

Structured competition labels take precedence target-by-target. Report-derived
targets fill only missing labels and carry extractor confidence weights.
Unlabeled targets are masked from the loss, not treated as negatives. Training
validation and checkpoint selection use Gold-labeled targets only.

## Model families

### DINOv2 MIL

`dinov2` is the default. A pretrained DINOv2 backbone encodes each window
using the CLS token, mean patch embedding, and high-magnitude patch features.
Window attention pools features within each slot; an optional cross-slot
transformer models interactions, then a target-specific slot head produces
the twelve logits. The anatomical attention prior is configurable with
`--no_slot_prior`. New checkpoints store the backbone configuration and
target order; failed pretrained-weight loading is an error rather than a
silent random-initialization fallback.

### timm attention MIL

`timm_mil` and `coatnet_mil` use timm image encoders and target-specific
attention pooling. New runs use hierarchical pooling by default: pool windows
within each of six fixed slots, then attend over valid slot summaries per
target. Use `--timm_pooling flat` for the legacy all-windows-together ablation.
Older checkpoints without pooling metadata retain flat pooling for state-dict
compatibility. `--random_init` is supported only for these timm-based models.

Hierarchical pooling preserves slot identity and remains permutation-invariant
to windows within a slot. It avoids imposing a recurrent sequence length on
the variable train/evaluation window sets, but has not been shown to improve
AUC; compare it with flat pooling on matched folds before treating it as a
gain. Both model families require task-specific training of the classification
head. Generic ImageNet or DINOv2 weights alone are not knee classifiers.

## Objective, optimization, and evaluation

The default objective is asymmetric loss with a shared negative focusing
exponent. Confidence weights scale each labeled entry; each target loss is
normalized by its labeled-entry count before the twelve target losses are
averaged. Gold-only entries receive the configured training upweight;
pseudo-label entries retain their confidence weights. Weighted BCE is
available for controlled comparisons. DINOv2 fine-tunes its final backbone
blocks with layer-wise learning-rate decay; timm currently fine-tunes the
backbone at a lower rate than its MIL head. Both use AdamW, warmup/cosine
scheduling, mixed precision, gradient accumulation, and clipping.

Each fold selects its best raw checkpoint using Gold validation AUC, then uses
that checkpoint for held-out-fold predictions. OOF output is required to cover
the requested studies and folds. Since each fold's best epoch is selected on
the same validation labels later used in the aggregate OOF score, that score
is selection-biased, especially with the small Gold cohort. It is internal
comparison evidence, not an unbiased generalization estimate.

## Inference and ensemble

Inference reconstructs the model and preprocessing configuration from each
checkpoint and rejects incompatible preprocessing metadata. It averages
fold probabilities within each model family, converts family predictions to
per-target average-tie percentile ranks, then averages families equally.
`--d4_target_weights` opts into a D4-derived target-weight schedule; those
weights were tuned for another model graph and are not presumed optimal for
local models. Positive temperature scaling can affect calibration but cannot
change per-target ROC-AUC ranking.

The Kaggle submission notebook embeds a snapshot of the source modules. Its
current checkpoint discovery supports DINOv2, CoAtNet, and `timm_mil` families,
including per-variant grouping for timm models. It does not automatically track
future `src/` changes, so regenerate and validate the embedded source when the
pipeline changes. The local pipeline is not a
reproduction of the D4 inference graph: its external Raptor, A5, and
RadImageNet task-trained checkpoints and preprocessing are not included.
The copied D4 notebook has no saved execution outputs in this checkout, so
its reported score and ensemble contribution cannot be reproduced here.

## Hardware and measured limits

The DGX Spark memory target and ceiling are safeguards, not a promise that the
process will allocate a fixed amount of unified memory. Cache storage is
separate from system memory; the default image cache is approximately 227 GB
decimal before metadata and scratch space. No leaderboard improvement,
throughput, or 0.965 score is claimed until measured on the target hardware and
competition validation data.
