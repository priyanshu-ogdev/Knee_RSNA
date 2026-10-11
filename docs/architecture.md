# RSNA Knee Classification Architecture

This document describes the upgraded production architecture implemented in `src/`.

## 1. Data, Slots, and Weak Supervision

The pipeline routes each study into six canonical anatomical MRI slots:
- Slot 0: Sagittal T1 / Proton Density
- Slot 1: Coronal T1 / Proton Density
- Slot 2: Axial / Patellofemoral
- Slot 3: Sagittal Fluid-Sensitive (T2 / PDFS / STIR)
- Slot 4: Coronal Fluid-Sensitive
- Slot 5: Miscellaneous / Secondary Sagittal

The canonical `v2` cache builds 2.5D windows composed of three spatially adjacent grayscale slices mapped to the model's RGB input channels:
- Training: Samples up to 5 depth-stratified windows per slot per epoch (`N_WINDOWS_TRAIN = 5`).
- Validation & Inference: Encodes all available valid windows deterministically.
- Missing slots and invalid windows are strictly masked, never fabricated.

Supervision and label provenance:
- Target-by-target gold precedence: 58 expert-labeled studies form the clean gold validation set.
- Multilingual NLP weak supervision: Reports in 8 languages (Croatian, Spanish, Turkish, Dutch, Greek, Russian/Bulgarian, English, German) are extracted using the v5 prompt with strict JSON schema constraints and target-specific lexicons.
- Extracted findings receive calibrated probabilities and confidence weights (`high` = 1.0, `medium` = 0.65, `low` = 0.85/reduced).
- Unaddressed and uncertain findings (`not_stated` / `-1`) remain strictly masked (`NaN`) from the loss, never silently imputed as negative. Mixed studies are marked explicitly.

## 2. Model Families

The production candidate ensemble consists of 10 checkpoints (5 folds each across two diverse model families):

### Family 1: CoAtNet / timm Hierarchical Attention MIL (`coatnet_mil`)
- Backbone: `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k`.
- Pooling: Hierarchical slot-aware MIL. Windows are first pooled within each anatomical slot via learned content attention, producing slot summaries.
- Learnable Gated Anatomical Priors:
  $$\text{slot\_scores} = \frac{\text{slot\_features} \cdot \text{slot\_query}}{\sqrt{d}} + \text{slot\_bias} + \sigma(\text{prior\_gate}) \odot \text{anatomical\_prior}$$
  The anatomical prior encodes clinical heuristics (e.g. ACL in sagittal fluid-sensitive slots, PF OA in axial, MCL in coronal). The gate is initialized weakly ($\text{prior\_gate} = -2.0 \implies \sigma(-2) \approx 0.12$), allowing the network to start image-driven while learning target-specific prior amplification or suppression.
- Backward compatibility: Checkpoint loading seamlessly supports legacy checkpoints lacking prior buffers.

### Family 2: DINOv2 Cross-Slot MIL (`dinov2`)
- Backbone: `dinov2-base` (518×518 input resolution).
- Features: CLS token concatenated with mean patch embedding and high-magnitude patch tokens.
- Cross-Slot Transformer: Models anatomical interactions across all six slots before applying target-specific classification heads.

## 3. Objective, Optimization, and Safety Guards

### Macro-AUC Aligned Loss (`auc_asl`)
The competition metric is mean ROC-AUC across all twelve targets. The default training loss combines Asymmetric Loss (ASL) with Pairwise Ranking AUC:
$$\mathcal{L}_{\text{total}} = (1 - \lambda_{\text{rank}}) \mathcal{L}_{\text{ASL}} + \lambda_{\text{rank}} \mathcal{L}_{\text{PairwiseAUC}}$$
- Default $\lambda_{\text{rank}} = 0.15$ (`AUC_RANK_WEIGHT`).
- Pairwise AUC computes softplus ranking penalties between positive and negative studies per target, weighted by label confidence.
- NaN targets and zero-weight entries are strictly excluded from ranking pools.
- An auxiliary `LocalAttentionSmoothness` module penalizes sharp attention jumps across adjacent valid windows within the same series.

### Host-Memory and VRAM Safety
- Effective batch size: Physical batch 8 with gradient accumulation 3 (`GRAD_ACCUM = 3`), producing an effective batch of ~24 studies without large VRAM spikes.
- Dynamic DataLoader budgeting: Available host RAM and queued image payloads are evaluated before worker instantiation. `num_workers` is scaled dynamically, falling back to 0 workers under memory pressure to avoid prefetch-driven OOM.
- Bounded Cache Forge: Preprocessing tasks are bounded at `2 × workers` in-flight to prevent executor future bloat.

### Checkpoint and Validation Integrity
- Gold validation macro-AUC: `fold*_best.pt` is saved strictly when gold validation score improves.
- Later sub-optimal epochs cannot overwrite the best model. `fold*_last.pt` is retained separately for recovery and diagnostics.
- Out-of-fold (OOF) predictions and ensemble weights are strictly generated from the verified best checkpoints.

## 4. Kaggle Offline Inference (30 GiB RAM / 2×16 GiB T4)

- Model loading: Weights are loaded once and cast to native mixed precision (BF16 if supported, FP16 on T4).
- Multi-GPU sharding: `torch.nn.DataParallel` activates when `batch >= torch.cuda.device_count()`, allowing a batch size of 2 to utilize both T4 GPUs.
- Resource bounds: Cache worker count set to 1, chunk size 8, and batch size 2 to operate safely within 30 GiB host RAM.
- Deterministic path: TTA disabled (`n_tta = 1`), `torch.compile` disabled by default (`RSNA_COMPILE = 0`) to eliminate compile warm-up latency.
- Ensembling: Fold predictions are averaged within each model family, converted to percentile ranks, and blended equally across families.
- Embedded snapshot: `notebooks/kaggle_submission.ipynb` maintains a synchronized base64 snapshot of all current `src/` modules for offline execution.
