# RSNA Knee upgrade audit and execution plan

**Status:** design finalized on 2026-10-14; the active training run is not
modified or invalidated by this document. The new cache contract is opt-in.

## Executive verdict

The repository has a serious model-selection defect: training checkpoints were
selected on the weak Qwen/report labels (`weight >= 0.49`) while the logs and
documentation described the score as gold validation. The code now selects on
gold-only AUC whenever a fold contains at least one target with both gold
classes, and keeps a clearly labelled weak-label fallback only for synthetic
smoke tests. This is the highest-confidence immediate correction.

The current DINOv2 + CoAtNet design is technically coherent, but it is not yet
the strongest demonstrated competition recipe. The main missing image-side
experiment is an independent CoAtNet-class view with dense 2.5-D windows. The
new `coatnet44` preprocessing preset is isolated from the running `v2` cache:

| Contract | Current `v2` | New `coatnet44` arm |
|---|---:|---:|
| image size | 518 | 384 |
| cached depth | 24 | 44 |
| through-plane span | physical-mm grid | 15--85% index span |
| adjacent windows | stride 2 | stride 1, 42 available windows |
| crop | 130 mm | 140 mm |
| robust intensity | p1--p99.5 | p2--p98 |
| side/reorientation | enabled | disabled to match the raw public view |

This is an experiment, not a guaranteed gain. It follows the strongest public
evidence that the CoAtNet/Raptor-style 384px dense-window pipeline is much
stronger than a DINO-only arm and that pipeline diversity matters more than
adding near-duplicate backbones.

No honest local audit can guarantee 0.970. The public evidence available in this
checkout shows a reproducible public-only frontier around 0.943--0.945, public
external-data blends around 0.950, and a leaderboard top near 0.965. The final
0.970 target therefore requires a measured new model plus a leaderboard
submission; local gold-58 AUC is a regression guard, not proof of leaderboard
performance.

## Competition and dataset facts

- The task predicts 12 study-level targets: ACL, MCL, medial/lateral meniscus,
  medial/lateral/PF OA, effusion, synovitis, Baker's cyst, contusion and
  fracture.
- The official metric is the unweighted mean of the 12 ROC-AUC values. Scores
  must be continuous rankings; thresholded diagnoses are not appropriate.
- The data snapshot has 4,407 studies, 24,371 series and 819,078 slices; only
  58 studies have expert labels and 4,349 have reports but no expert labels.
  Reports are absent at test time.
- The audit found 21 language/vendor site proxies, 238 metadata-poor studies,
  57 cross-study single-image hash groups, and one corroborated duplicate pair.
- 51.89% of series have materially varying DICOM window tags; per-series
  percentile normalization is therefore required.
- Pixel scale varies by roughly 22.5x across scanners, 7.3% of series are
  non-square, and 72.4% have a meaningful non-zero border pedestal. The current
  physical crop, robust normalization and border-median padding are justified.
- `Fluid_Sensitive` and `Fat_Suppression` are identical in the training CSV but
  must not be hard-wired equal for hidden test data.

The local audit sources are `docs/data_analysis_report.md`,
`docs/deep_report.md`, `docs/architecture.md`, and the compact tables under
`docs/tables/`.

## Research synthesis

The following public material was reviewed. Scores are reported as evidence from
the authors' ledgers, not as independently reproduced results in this checkout.

1. **Tony Lica — DINO/RadImageNet rank ensemble** and **Anthony Therrien —
   DINO + RadImageNet + dual CoAtNet**: useful evidence for rank blending and
   architectural diversity, but the public 0.943 stack contains several
   pretrained/public arms and is not a clean estimate of this repository's
   models.
2. **Renta0426 — weak-label DINOv2** and **Paiky1995 — CoAtNet fine-tune**:
   support report-derived supervision and CoAtNet-style 2.5-D MIL, but single
   arms are below the strongest ensemble.
3. **TranBaDat2607/RSNA-Knee-Abnormality-Detection**: the most useful public
   reproducibility ledger. It reports public CoAtNet arms around 0.917--0.920,
   a clean CoAtNet-family submission around 0.939, and that a different
   pipeline/teacher helps more than another near-duplicate checkpoint.
4. **willmurray9/kaggle-rsna-knee-abnormality-detection**: reports a public
   stack around 0.943--0.944 and an independently trained ConvNeXt reader. Its
   phase-3 notes report an OAI-trained public CoAtNet blend around 0.950 and
   explicitly state that the top leaderboard values depend on data/weights not
   fully reproducible from the public notebooks.
5. The current repository's `docs/competition_research_2026-10.md` correctly
   warns that public rank weights and hidden checkpoints are not evidence that a
   local pipeline will reach 0.965.

The strongest consistent findings are:

- CoAtNet/Raptor-style MIL is the strongest public single-family starting point.
- DINO/RadImageNet/ConvNeXt add value mainly when preprocessing, labels or
  training sources are genuinely different; same-view backbone swaps are highly
  correlated.
- More TTA is not automatically useful. Public ablations found highly
  correlated multi-scale views and no reliable gain.
- Report-label quality matters. The current Qwen output must be gold-audited
  per target before its confidence weights are trusted. Do not assume that
  self-training or OOF pseudo-label mixing helps: the cited public ablations
  found cases where it reduced leaderboard performance.
- Public per-target weights were often selected from leaderboard feedback. They
  may be used as a hypothesis, never as validated local truth.

## Current pipeline audit

### What is sound

- DICOM indexing checks geometry, series coverage, unreadable slices, mixed
  shapes, laterality and duplicate candidates before training.
- The cache stores the preprocessing contract in its metadata and checkpoints
  carry the same configuration for inference parity.
- Missing report findings are represented as masked entries rather than being
  silently converted to global-prevalence negatives.
- Gold labels take precedence over extra labels target by target.
- The DINOv2 model uses masked window pooling, cross-slot interaction and a
  target-specific head. The timm arm supports hierarchical slot/window MIL.
- Training has AMP, gradient accumulation, gradient clipping, dynamic worker
  budgeting, atomic checkpoints, OOF export and memory circuit breakers.
- Inference checks finite outputs, test-ID order, checkpoint preprocessing
  parity and target schema. Family rank blending avoids incompatible probability
  scales.

### Blocking or high-risk issues

1. **Fixed in code: wrong checkpoint-selection source.** Before this upgrade,
   `evaluate()` used weak labels for selection and only printed gold AUC as a
   diagnostic. The selection path is now gold-first and emits `[AUC SOURCE]`.
2. **Qwen labels are not yet a demonstrated teacher.** The repository has a
   strict extraction/provenance contract, but it does not have a checked-in,
   completed gold-agreement report for the current Qwen run. v6 also corrects
   the confidence ordering so low-confidence entries are down-weighted below
   medium-confidence entries. Run
   `scripts/evaluate_nlp_gold.py` with the exact model and manifest before
   treating the output as production supervision.
3. **The default v2 view is not the public CoAtNet view.** It is a reasonable
   physical-mm, 518px, 24-depth representation, but it limits through-plane
   coverage and available adjacent windows. `coatnet44` supplies the missing
   controlled arm without touching v2 artifacts.
4. **Static D4 weights are not locally selected.** `--d4_target_weights`
   currently applies public-derived per-target weights. The default comparison
   should be equal family rank blending; target weights must be fitted only on a
   development slice of weak-label OOF data and frozen before final evaluation.
5. **Gold-58 is small.** A fold's selected epoch and a target-specific blend
   can overfit it. Report per-target counts and paired bootstrap intervals; do
   not call a 0.003 local gain decisive.
6. **TTA defaults were inconsistent.** The master pipeline used four random
   augmented views while the standalone inference CLI defaulted to one. The
   master now exposes `--n_tta` and defaults to one; promote more views only
   after an OOF ablation.

### Things deliberately not changed

- The active run's cache, labels, folds and process are not overwritten.
- No external/private checkpoint is copied into the repository.
- No pseudo-label is used for validation when gold labels are available.
- No OAI or other external-data model is silently added. Use a separately
  documented, license/rules-checked track if the team elects to do so.

## Upgrade stages and gates

### Stage 0 — preserve the active run

Let the already-started run finish. Record its exact git commit, cache metadata,
label manifest, command line, model family, fold, peak RAM/VRAM and all OOF
files. Do not rebuild `pipeline_out/cache/v2` in place and do not mix its
checkpoints with a new preprocessing contract.

### Stage 1 — label audit and controls (CPU / local)

1. Verify Qwen v6 manifest: source CSV hash, report hash, prompt hash, model ID,
   extractor version, row count, all 12 target columns and masked `not_stated`.
   Existing v5 rows can be passed through `scripts/migrate_qwen_v5_to_v6.py`;
   a full 72B rerun is not required for this calibration-only correction.
2. Run the gold-report audit and save per-target AUC/agreement, confusion
   examples, language strata and confidence histograms.
3. Build three immutable label files:
   - `gold_only`: only the 58 expert studies;
   - `qwen_soft`: current Qwen probabilities with target confidence weights;
   - `teacher_consensus`: only if an independently sourced, license-compatible
     clean teacher is available and passes the gold audit.
4. Train a short fold-0 control on each label policy. Select the policy from
   gold-only validation, not from report-label AUC. If Qwen loses materially on
   gold, reduce its training weights or use it only for targets where it passes.

Do not start a self-training cycle until a clean teacher and an independent
weak-label holdout show improvement. The public evidence is mixed and the
current code has no proof that Qwen pseudo-labels improve the image target.

### Stage 2 — build the independent dense-window cache

Use a fresh output directory and the new preset:

```bash
python src/main.py \
  --data_root /path/to/competition \
  --work_dir /path/to/rsna_run_coat44 \
  --model_dir /path/to/rsna_run_coat44/models \
  --cache_preset coatnet44 \
  --cache_prefix_name coatnet44 \
  --prepare_only \
  --preprocess_workers 8
```

Expected image payload is approximately 171.6 GB for 4,407 studies before small
metadata files, versus approximately 170.3 GB for the v2 518x24 image memmap.
Confirm actual free NVMe space and the preparation manifest before training.
Run the 16-study forge and full QC gates first. The cache must report zero
unreadable/mixed-shape series, complete `done` flags, exact study coverage and
the matching source signature.

### Stage 3 — train the strongest clean arms

Run the arms sequentially on the single DGX Spark unit so the memory ceiling is
meaningful and checkpoint provenance stays simple.

**Arm A: dense-window CoAtNet**

```bash
python src/main.py \
  --data_root /path/to/competition \
  --work_dir /path/to/rsna_run_coat44 \
  --model_dir /path/to/rsna_run_coat44/coatnet \
  --cache_preset coatnet44 \
  --folds 0,1,2,3,4 \
  --model_type coatnet_mil \
  --variant coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k \
  --timm_pooling hierarchical \
  --epochs 15 \
  --batch_size 8 \
  --grad_accum 2 \
  --n_windows_train 16 \
  --num_workers 6 \
  --prefetch_factor 1 \
  --eval_batch_size 12 \
  --inference_batch_size 2 \
  --n_tta 1
```

Start with 16 training windows; after the fold-0 gate compare 16 versus 24 on
the same fold. Validation/inference should encode all 42 valid windows. Do not
increase `n_windows_train` merely because the cache is deeper: the public
ablations show that more windows can be slower or worse when the crop/labels do
not match.

**Arm B: existing DINO control**

Finish the active v2 DINO run if it passes the new gold-only selection check.
If it must be rerun, use the existing v2 cache and a fresh model directory; do
not use the `coatnet44` cache with DINO just to fill a slot in the ensemble.

**Arm C: independent reader (only if time remains)**

Use `timm_mil` with a ConvNeXt variant only if it has both a different view and
an independent label policy. A same-label/same-view ConvNeXt is expected to be
highly correlated with CoAtNet and is lower priority than another strong
CoAtNet-class pipeline. Keep it only if its OOF rank correlation is below 0.95
with Arm A and its fixed-weight rank blend improves the development split.

### Stage 4 — OOF ensemble selection without leakage

For every arm, verify exact fold coverage, finite predictions, target order and
preprocessing metadata. Then create one table containing all fold-held-out
predictions. Split the non-gold studies into a development and a locked
selection slice by the frozen study/site groups:

1. Use the development slice to compare equal family-rank blending,
   probability averaging, and a small grid of target-specific family weights.
2. Keep the simplest blend unless the gain is positive for at least 8 of 12
   targets and survives a paired bootstrap interval.
3. Never use the 58 gold studies to fit weights. Use gold only as a regression
   guard and report it separately.
4. Freeze the chosen weights and apply them to test. Re-rank each target column
   only after the final blend. Save the exact weight JSON next to the
   submission.

Initial blend order:

1. equal rank blend of the five CoAtNet44 folds;
2. equal rank blend of CoAtNet44 and the existing DINO family;
3. optional independent reader at a predeclared 0.10--0.20 global weight;
4. only then evaluate any public D4-style target weights as a diagnostic.

This is intentionally more conservative than the current copied D4 weights.
The public ledger says those weights were leaderboard-probed and that the
gold-58 set cannot resolve small differences.

### Stage 5 — inference and submission gate

Run with `--n_tta 1` first. Test `--n_tta 4` only if a held-out OOF TTA test
shows a stable gain and rank correlation below 0.99. The submission gate is:

- exact test study IDs and order;
- all 12 target columns, finite values and no NaNs;
- all checkpoint/cache preprocessing contracts agree within an arm;
- no fallback/empty-study count beyond the preparation manifest tolerance;
- peak host RAM below the configured 90 GiB ceiling;
- measured cache build, inference wall time and studies/sec;
- source, dependency, label-manifest, fold, weight and checkpoint hashes saved.

## One-week DGX schedule

| Day | Work | Stop/continue decision |
|---|---|---|
| 1 | Finish active run; verify gold-only checkpoint selection and OOF artifacts | Stop if loss/logits/checkpoint reload are invalid |
| 2 | Complete Qwen gold audit and build label controls; forge `coatnet44` | Stop Qwen promotion if target-level audit is poor |
| 3 | CoAtNet44 fold-0 smoke and 2-epoch convergence check; compare 16/24 windows | Keep the faster recipe only if gold is not worse |
| 4--5 | Train CoAtNet44 folds 0--4; record OOF and telemetry | Abort a family with repeated NaNs, cache mismatch or memory breach |
| 6 | Finish DINO control / optional independent reader; fit OOF blend on locked protocol | No public-LB tuning; keep equal blend if weight search is unstable |
| 7 | Offline inference smoke, TTA ablation, final submission candidates and audit | Submit only artifacts passing every gate; keep clean and external tracks separate |

If training time is shorter than expected, prioritize in this order: complete
CoAtNet44 five folds, complete DINO five-fold OOF, equal rank blend, then
optional reader/TTA. Do not sacrifice fold coverage for an unverified sixth
architecture.

## Expected impact and uncertainty

These are planning ranges, not promises:

| Change | Expected role | Confidence |
|---|---|---|
| gold-only checkpoint selection | removes validation-teacher bias; may change epoch/model choice | high for correctness, unknown score delta |
| Qwen audit and target-specific weights | reduces systematic weak-label damage | medium; must be measured |
| dense 384px CoAtNet44 view | stronger through-plane and local-detail signal | medium-high from public evidence, not locally proven |
| equal/OOF-fitted family rank blend | reduces leaderboard-weight overfit | medium |
| extra TTA | possible small variance reduction | low; public evidence often null |
| OAI/external-data public checkpoint track | needed to approach the 0.950+ public frontier | medium for score, dependent on rules/license/provenance |

The realistic clean-track objective for this week is to beat the repository's
measured baseline and approach the reproducible 0.943--0.945 frontier. A 0.970
submission is a stretch target that likely needs a strong external-data or
medical-pretraining arm plus genuine pipeline diversity. It must be verified by
the Kaggle receipt; no local metric can certify it.

## Files changed by this upgrade

- `src/core/config.py`: added the isolated `coatnet44` cache preset and made the
  pairwise-loss default conservative (`0.15`).
- `src/data/preprocess/nlp_extractor.py`: bumped the provenance contract to v6
  and corrected low-versus-medium confidence weights; v5 pseudo-labels must be
  regenerated.
- `src/training/train.py`: checkpoint selection now uses gold-only AUC whenever
  gold classes are available and logs the metric source.
- `src/main.py`: added cache preset/override CLI options and a consistent,
  opt-in TTA count (`--n_tta`, default 1).
- `tests/test_oof_validation.py`: regression test proving pseudo-labels cannot
  override gold checkpoint selection.
- `tests/test_pixels_wide_center.py`: cache-contract test for `coatnet44`.
- `docs/UPGRADE_EXECUTION_2026-10.md`: this audit, research synthesis and
  execution plan.

The new implementation is intentionally backward-compatible with existing v2
checkpoints and does not require rebuilding the active cache until the new arm
is explicitly launched.
