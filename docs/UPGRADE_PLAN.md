# RSNA Knee pipeline upgrade plan

## Objective

Optimize the competition objective (macro ROC-AUC over 12 study-level targets)
without contaminating the 58 expert studies or risking unified-memory OOM.

## Execution order

1. **Data contract and provenance**
   - Freeze `train.csv`, `train_series.csv`, report hashes, duplicate groups, and
     grouped five-fold assignments before model selection.
   - Keep expert labels, report labels, and image pseudo-labels in separate columns
     with per-target weights. Unknown is masked, never silently negative.
2. **NLP supervision**
   - Run structured, schema-constrained vLLM extraction in resumable chunks (v5 prompt).
   - Bound concurrency from live RAM headroom and cap it at 64 requests.
   - Validate every row against UID, report hash, prompt hash, model ID, and extractor
     version (`clinical-report-labels-v5`); audit report-label AUC on the expert set before use.
3. **Preprocessing**
   - Build one canonical DICOM index, verify spatial ordering, laterality, geometry,
     series coverage, and duplicate groups.
   - Cache uint8 2.5D windows with the exact preprocessing config in the manifest.
   - Bound in-flight tasks (`2 × workers`) to eliminate future-queue overhead.
4. **Training**
   - Use the DINOv2/CoAtNet MIL families with grouped folds and masked labels.
   - Default to `auc_asl` (0.85 ASL + 0.15 PairwiseAUC) because the official metric is
     ROC-AUC rather than thresholded accuracy.
   - Employ learnable gated anatomical priors in hierarchical MIL, initialized near zero.
   - Use AMP, gradient accumulation (`GRAD_ACCUM = 3`, batch 8 -> effective batch 24),
     gradient clipping, layer-wise learning rates, early stopping, and OOF predictions.
   - Derive DataLoader worker count dynamically from live RAM and queued payload before spawning
     workers.
5. **Model selection**
   - Save `fold*_best.pt` strictly on gold validation macro-AUC improvement; preserve `fold*_last.pt`.
   - Select epochs and ensemble weights using fold-held-out OOF predictions only.
   - Compare per-target AUC and macro AUC; never tune target weights from the public
     leaderboard.
6. **Efficiency-track inference**
   - Load checkpoints once, share the DICOM cache, batch valid windows only, use
     inference mode and native BF16/FP16, and keep TTA bounded (`n_tta = 1`, `RSNA_COMPILE = 0`).
   - Enable DataParallel when `batch >= n_gpus` to fully utilize 2× T4 GPUs.
   - Rank-blend diverse families after validating identical preprocessing contracts.
   - Verify UID order, finite probabilities, exact target columns, and row count.

## Recommended full run

```bash
python src/main.py \
  --folds 0,1,2,3,4 \
  --model_type coatnet_mil \
  --variant coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k \
  --timm_pooling hierarchical \
  --epochs 15 \
  --batch_size 8 \
  --grad_accum 3 \
  --n_windows_train 5 \
  --num_workers 8 \
  --prefetch_factor 1 \
  --inference_batch_size 2
```

The runtime safety layer may reduce worker count or NLP concurrency based on live
headroom. This is intentional: throughput is preferable to a host-wide OOM that
corrupts a fold or invalidates an efficiency-track submission.

## Required evidence before claiming an improvement

- Per-target and macro OOF AUC on the same frozen folds.
- Gold-only and weak-label-only metrics reported separately.
- Peak RAM/VRAM, samples/sec, and inference wall-clock time.
- Exact checkpoint, preprocessing, label-manifest, and dependency hashes.
- At least one ablation isolating the proposed change.
