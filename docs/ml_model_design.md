# ML Architecture: Design, Inputs, and Outputs

This document describes the implementation in `src/modeling/model.py` and
`src/core/config.py`. DINOv2 remains the default; a timm-based, target-attention
MIL model is also available for CoAtNet and compatible timm encoders.

## 1. Input tensor

The preprocessing cache stores six anatomical slots per study. Each slot has a
fixed-depth image stack; the loader selects 2.5D windows of adjacent slices.
The model accepts images shaped `[B, S, K, G, H, W]` (or a legacy
`[B, S, G, H, W]` input), where `S=6`, `G=3`, and the current `v2` preset uses
`H=W=518`, depth 32, and stride 2. Training selects up to five stratified
windows per slot; evaluation uses all available windows unless configured
otherwise.

The three adjacent grayscale slices are passed as the three image channels.
Only valid windows are sent through the image backbone.

## 2. DINOv2 feature extraction

`build_model()` uses a Hugging Face DINOv2 backbone (Base by default). Training
freezes the backbone initially, then applies the configured LoRA adapters and
unfreezes the final transformer blocks. The actual defaults are defined in
`src/core/config.py`; checkpoints record the model configuration so inference
can reconstruct the same architecture.

For each window, the feature vector concatenates the CLS token, mean patch
embedding, and mean embedding of the highest-magnitude patch tokens. A
masked `WindowPool` attention layer aggregates window features into one vector
per anatomical slot. An optional `CrossSlotTransformer` models interactions
between the six slots, followed by a target-query `SlotHead` that returns one
logit per finding.

## 3. Training objective

The default objective is Asymmetric Loss. With `p = sigmoid(logit)` and
`p_neg = min(1, 1 - p + clip)`, the negative log term is
`(1 - target) * log(p_neg)`. Focusing uses `p` for positive entries and
`p_neg` for negative entries; unlabeled entries are excluded using zero
weights. `WeightedBCE` remains available for controlled ablations.

## 4. CoAtNet attention-MIL model

Select `--model_type coatnet_mil` and a timm architecture such as
`coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k`. This arm uses the same six-slot cache
and label weights as DINOv2. It resizes each valid cached window to 384 pixels,
applies ImageNet normalization, encodes windows in bounded chunks, and pools
the resulting window features with a separate attention distribution per
finding. Empty slots/studies are masked. Pretrained timm weights are used by
default; `--random_init` disables that initialization.

For a memory-conscious DGX Spark starting run:

```bash
python src/main.py --model_type coatnet_mil --variant coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k --batch_size 4 --grad_accum 4 --n_windows_train 2
```

This is a starting configuration, not a measured throughput or score
recommendation; tune only after observing unified-memory use on the host.

## 5. Outputs and checkpoint contract

The model returns logits shaped `[B, 12]`, ordered by `config.TARGETS`.
Sigmoid probabilities are computed during evaluation and inference.
Checkpoints include model architecture settings and the preprocessing config.
Inference restores that preprocessing config from the checkpoint and rejects
an incompatible config or a mixed-config checkpoint ensemble.

## 6. Report-derived labels and validation limits

`src/data/labels.py` gives structured competition labels precedence over report
extractions. Extra targets must be probabilities in `[0, 1]`; their training
weights are the extractor confidence multiplied by `extra_weight` (currently
`0.5` in `src/main.py`). The NLP extraction cache is considered complete only
when every required non-gold study has all twelve in-range targets. Invalid or
partial rows are regenerated; extraction failures now stop the pipeline rather
than silently switching to gold-only training.

Fold checkpoint selection and the aggregate OOF report use only gold targets
with both classes. Each fold writes predictions from its selected best
checkpoint for that fold's held-out studies; the aggregate gold-only metric is
written to `oof_metrics.json`. Pseudo-label-only AUC is not used to select an
epoch or reported as validation performance. Since each fold's best epoch is
selected on that same fold's gold validation labels, the aggregate is
selection-biased and is for internal comparison, not an unbiased external
estimate. No temperature calibrator is fitted; inference uses the identity
temperature.

The copied d4 notebook is an inference/ensemble graph, not a training recipe.
Its code depends on separately published Kaggle model assets and pinned
checkpoints. The copied notebook has no executed cells or saved outputs, so its
reported leaderboard score and its new blend contribution cannot be
independently reproduced from this repository. The new CoAtNet attention-MIL
arm is a `src/`-native complementary experiment, not a reproduction of the d4
CoAtNet/Raptor/RadImageNet/DINO fleet. Exact reproduction still requires the
original training recipes, source artifacts, and leakage-safe predictions for
each member family.

For live inference, pass trained DINOv2 and CoAtNet checkpoints together:

```bash
python -m src.inference.inference \
  --root /kaggle/input/rsna-knee-abnormality-detection \
  --checkpoints /kaggle/input/rsna-models/dino/fold0_best.pt \
               /kaggle/input/rsna-models/dino/fold1_best.pt \
               /kaggle/input/rsna-models/coatnet/fold0_best.pt \
               /kaggle/input/rsna-models/coatnet/fold1_best.pt \
  --out /kaggle/working/submission.csv
```

The inference process loads the checkpoint models and runs them on the
competition test data. When both families are present, it averages
probabilities within each family, converts each family output to average-tie
percentile ranks per target, applies the d4-derived CoAtNet target weights,
then rank-normalizes the blend. CoAtNet has weight `0.60` by default, `0.75`
for ACL, Lateral OA, and Fracture, `0.80` for Medial Meniscus, and `1.00` for
Lateral Meniscus. `src/main.py --ensemble_checkpoints ...` supports the same
live-checkpoint ensemble alongside a family trained by that run.

This is a live DINOv2/CoAtNet blend using the d4 notebook's target-weight
schedule, not an exact reproduction of its complete blend graph: the notebook's
other source is a Raptor/CoAtNet hybrid, while this repository does not run the
external Raptor, A5, RadImageNet, or auxiliary CoAtNet arms or their distinct
preprocessing. The copied d4 notebook also has no saved execution outputs, so
its reported leaderboard score remains unverified here.
