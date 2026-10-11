# Final pipeline readiness plan

This document is the go/no-go checklist before spending the full training budget.

## Blocking gaps addressed

- Gold validation now selects `fold*_best.pt`; later worse epochs no longer
  overwrite the selected model.
- `fold*_last.pt` remains available for recovery/debugging.
- OOF verification must consume the best checkpoint when one exists.
- NLP v5 must be regenerated/resumed under its current provenance contract.
- Dataforge cache construction uses bounded in-flight batches.

## Phase 0 — environment gate

Record:

- Python, PyTorch, CUDA, timm, transformers, vLLM versions.
- GPU count, device names, VRAM, BF16 support.
- Total/available host RAM.
- Cache disk capacity.
- Git commit and dirty-file manifest.

Abort if the active Python environment lacks the training dependencies, or if the
available disk cannot hold the selected cache plus 20% scratch headroom.

## Phase 1 — NLP v5 gate

Run the extractor with the same output path. Resume is allowed only when all of the
following match:

- source `train.csv` hash;
- model ID;
- prompt hash;
- extractor version `clinical-report-labels-v5`;
- report hash per study.

Require complete 12-target records, valid states/confidences, masked not-stated
targets, and a gold-report audit. Save coverage, disagreement, language, and
confidence histograms.

Do not start image training with an incomplete manifest.

## Phase 2 — dataforge gate

Run indexing, slot assignment, and cache construction with bounded workers. Require:

- exact DICOM/CSV study and series coverage;
- zero unreadable series;
- zero mixed-shape series;
- canonical geometry and laterality gates;
- unresolved laterality below 5% (empirically 1.72% on 4,407 train studies);
- no missing selected slots beyond the documented tolerance;
- complete cache `done` flags;
- cache source signature equal to the index source signature.

Run a 16-study debug forge first, then the full forge. Never rebuild a full cache
after training has started without changing the cache prefix.

## Phase 3 — convergence smoke test

Before all folds:

1. Train one fold for two epochs on a small deterministic subset.
2. Confirm finite loss, finite logits, nonzero gradients, and decreasing loss.
3. Confirm at least eight targets have both classes in validation.
4. Confirm one `fold0_best.pt`, one `fold0_last.pt`, and one OOF CSV are emitted.
5. Reload the best checkpoint and reproduce validation predictions.

If any check fails, do not launch the ten-model run.

## Phase 4 — production models

Train ten checkpoints total:

- CoAtNet hierarchical MIL: folds 0–4.
- DINOv2 cross-slot MIL: folds 0–4.

Use fresh output directories per family. Never mix stale artifacts.

Recommended safe defaults:

```text
physical batch      8
gradient accumulation 3
windows per slot    5
prefetch factor     1
workers             8, dynamically reduced by memory guard
loss                auc_asl
rank weight         0.15
clip norm           0.5
warmup              8%
cosine decay        enabled
```

CoAtNet: 15 epochs, head LR 1e-3, backbone LR 3e-5, WD 0.02.

DINOv2: 12 epochs, head LR 2e-3, backbone LR 1e-5, WD 0.05.

## Phase 5 — OOF and model selection

For every fold, verify:

- validation IDs equal the fold manifest;
- no OOF ID occurs twice;
- all predictions are finite and in [0,1];
- target order matches `config.TARGETS`;
- preprocessing metadata matches across ensemble members.

Report gold-only, weak-label-only, and per-target AUC separately. Select the best
epoch from gold validation only. Treat the aggregate as internal evidence because
the 58-study gold set is small and epoch selection is itself validation-dependent.

## Phase 6 — inference efficiency gate

Load checkpoints once. Reuse the test cache. Encode only valid windows. Use
`torch.inference_mode()`, BF16 when supported, bounded model batches, and bounded
TTA. Measure:

- peak host RAM;
- peak VRAM;
- studies/sec;
- total wall-clock time;
- fallback/error count.

Generate one submission only after checking exact UID order, exact target columns,
finite values, and row count.

## Go/no-go criteria

Proceed to the full run only when:

- NLP v5 manifest is complete;
- dataforge acceptance gates pass;
- convergence smoke passes;
- checkpoint reload reproduces predictions;
- memory telemetry remains below the configured ceiling;
- dependency and source hashes are recorded.

No leaderboard claim is valid without all ten model manifests, OOF metrics, and
inference telemetry.
