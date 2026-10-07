# RSNA Knee Classification Architecture

## Current models

The pipeline trains twelve study-level findings from six routed MRI series
slots. Cached images are sampled as 2.5D windows of three adjacent slices.
Training and inference share the same preprocessing/cache contract.

### DINOv2 MIL

The default `dinov2` model uses a pretrained DINOv2 image encoder. It combines
CLS, mean-patch, and high-magnitude patch features for each window, pools
windows within each slot, optionally applies cross-slot attention, and routes
slot features to target-specific outputs. It is task-trained using the
competition labels; the generic DINOv2 encoder weights alone do not predict the
twelve findings. LoRA is disabled by default. The anatomical slot prior is a
heuristic that still needs a matched ablation.

### timm attention MIL

`timm_mil` uses a pretrained timm image backbone and target-specific attention
over valid windows. It currently pools the windows from all slots together;
unlike DINOv2, it does not explicitly encode slot identity or add a cross-slot
stage. This is architectural diversity, but also a concrete ablation
opportunity: compare flat pooling with hierarchical window-within-slot and
slot-within-target pooling using the same folds, labels, and training budget.
Do not promote the more complex head without repeatable out-of-fold benefit.

## Pretraining, task fitting, and calibration

Pretrained weights provide an image representation, not a knee-abnormality
classifier. For a generic encoder, fit at least a task-specific classification
or MIL head; then compare a frozen-encoder probe with partial fine-tuning.
An already-trained RSNA checkpoint can be loaded directly only when its output
head, target order, preprocessing, and validation provenance match.

Positive scalar temperature scaling preserves per-target prediction order and
therefore cannot improve ROC-AUC by itself. It can improve probability
calibration for a probability-sensitive metric. In a rank-based ensemble it
does not affect ranks. With 58 Gold studies, fitting many target-specific blend
weights or nonlinear stackers to the same predictions is prone to overfitting.

## D4 comparison

The copied D4 notebook declares an inference ensemble made from external,
task-trained checkpoints: 20 DINO members, five A5 folds, RadImageNet E10/E13/E11
heads, four Raptor views, four CoAtNet readers, and optional student fleets.
Its code pins and validates external artifacts; it does not train those models
from generic weights during inference. This repository does not reproduce
those model families, checkpoints, or all of their preprocessing.

The notebook in `tmp/d4_blend/` has no executed cells or saved outputs in this
checkout. Its documented 0.946 leaderboard score and the contribution of its
full current graph are therefore not independently verifiable here. The
D4-derived target-specific DINO/CoAtNet weights are opt-in only; equal
architecture-family rank blending is the local default because the local
CoAtNet model is not the D4 Raptor/CoAtNet system.

## Recommended experiment order

1. Establish matched-fold OOF baselines for DINOv2 and one genuinely different
   CNN backbone such as ConvNeXt-Small. Measure per-target AUC and rank
   correlation, not only the aggregate.
2. Test frozen-backbone/head-only training against the existing partial
   fine-tuning recipe. Keep the validation studies Gold-only and prevent
   pseudo-labels from becoming validation targets.
3. If the CNN contributes complementary errors, test slot-aware hierarchical
   MIL against flat pooling. Reuse the same folds, augmentation, epochs, and
   label policy so the head is the only intended change.
4. Compare equal family-rank blending with the D4-weighted blend on predictions
   not used to select epochs or weights. Retain extra arms only when the
   improvement is consistent and checkpoint/preprocessing provenance is
   complete.

No architecture, ensemble, memory, or leaderboard gain is claimed until these
experiments are run on the actual dataset and DGX environment.
