# Final training and data-forge specification

## Model count

The production candidate set is **10 checkpoints**:

- 5 grouped folds of `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k`, hierarchical MIL.
- 5 grouped folds of `dinov2-base`, cross-slot attention enabled.

This is the smallest ensemble that gives both fold averaging and architecture
diversity. Do not train more models until this 10-checkpoint set has complete OOF
predictions and a measured gain. Each fold is independently checkpointed; inference
averages within family, then rank-blends the two families.

## Convergence recipe

### Stage A: smoke/convergence gate

Run fold 0 only for 2 epochs using 10% of the cache. The loss must decrease, all
12 logits must remain finite, and at least 8 targets must have both classes in the
gold validation set. This is a pipeline gate, not a model score.

### Stage B: full candidates

Use the following fixed starting recipe for all ten checkpoints:

| Setting | CoAtNet MIL | DINOv2 |
|---|---:|---:|
| epochs | 15 | 12 |
| batch size | 8 | 8 |
| gradient accumulation | 3 | 3 |
| effective batch | 24 | 24 |
| windows/slot | 5 | 5 |
| head LR | 1e-3 | 2e-3 |
| backbone LR | 3e-5 | 1e-5 |
| weight decay | 0.02 | 0.05 |
| warmup | 8% | 8% |
| scheduler | cosine | cosine |
| loss | `auc_asl` | `auc_asl` |
| pairwise weight | 0.15 | 0.15 |
| clip grad norm | 0.5 | 0.5 |
| AMP | BF16, FP16 fallback | BF16, FP16 fallback |

The smaller physical batch plus accumulation is intentional: it bounds GPU memory
while preserving the optimizer statistics of a 24-study batch.

### Stage C: constrained tuning

Only tune these three factors, one at a time, using identical folds:

1. `rank_weight ∈ {0.0, 0.10, 0.15, 0.20}`.
2. `n_windows_train ∈ {3, 5, 8}`.
3. backbone LR multiplier `∈ {0.5, 1.0, 2.0}`.

That is at most 12 controlled experiments per family. Select using pooled OOF
macro-AUC, with per-target AUC reported separately. Never tune on the public
leaderboard. Because only 58 studies are expert-labeled, retain the simpler recipe
unless the improvement is consistent across at least 3 folds and at least 8 targets.

## NLP: rerun or continue?

The NLP extractor has now changed to version `clinical-report-labels-v5`:

- adaptive memory-safe vLLM chunk sizing;
- explicit handling of medium confidence;
- the same provenance checks and report hashing.

Therefore, **do not blindly consume an old NLP output**. Run the extractor again with
the same output path. It will resume valid rows only if the manifest matches; the
version change intentionally invalidates prior rows and regenerates them under v5.
This is the correct “resume with provenance” behavior, not an untracked full rerun.

Before training, require:

- manifest status `complete`;
- source `train.csv` hash matches;
- all required UIDs present;
- prompt/model/extractor provenance matches;
- gold-report audit completed;
- uncertain/not-stated entries remain masked.

If the previous v4 run had no audited disagreement report, rerun gold evaluation
before merging pseudo-labels into training.

## Data forge

The cache is a canonical uint8 memmap with six anatomical slots and 2.5D windows.
The builder now uses:

- spatially canonicalized DICOM ordering;
- resumable `done` flags;
- LPT study scheduling;
- bounded in-flight process batches (`2 × workers`, minimum 8);
- direct shared-memmap writes rather than pickling image arrays;
- cache fingerprints for source DICOM/CSV/config parity.

This bounded scheduler is important: submitting all studies as futures retains a
large task graph and can create avoidable host-memory pressure. The builder now only
keeps a bounded batch of decode tasks active.

Recommended forge settings:

```bash
python -m src.main --prepare_only \
  --preprocess_workers 8 \
  --prefetch_factor 1
```

Acceptance gates must pass before training: zero unreadable series, zero mixed-shape
series, canonical geometry, laterality unresolved below 5%, complete study coverage,
and no missing expected selected slots.

## Production run

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
  --prefetch_factor 1
```

Run the DINOv2 family separately into a fresh output directory, then combine only
after both families pass OOF verification. Do not mix stale fold artifacts.
