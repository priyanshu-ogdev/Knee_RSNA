#!/usr/bin/env bash
# Resumable DGX launcher. Never deletes competition data, caches, labels or
# previous model directories. The old launcher forced a 72B re-extraction and
# used a relative model path; both are unacceptable when a run has already
# spent hours on the first epoch.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

DATA_ROOT="${KNEE_DATA:-$ROOT/data/competitions/rsna-knee-abnormality-detection}"
WORK_DIR="${RSNA_OUT_DIR:-$ROOT/pipeline_out}"
MODEL_DIR="${RSNA_MODEL_DIR:-$WORK_DIR/exp_coatnet44}"
PSEUDO="$WORK_DIR/pseudo_labels.csv"

mkdir -p "$WORK_DIR" "$MODEL_DIR"

if [[ ! -f "$DATA_ROOT/train.csv" || ! -d "$DATA_ROOT/train_series" ]]; then
  echo "Dataset is incomplete under $DATA_ROOT" >&2
  exit 2
fi

if [[ ! -f "$PSEUDO.manifest.json" ]]; then
  echo "Missing pseudo-label manifest: $PSEUDO.manifest.json" >&2
  echo "Run the one-time NLP preparation/audit before launching training." >&2
  exit 3
fi

# Existing v5 Qwen outputs can be repaired deterministically from their stored
# confidence weights. This avoids re-running Qwen2.5-72B merely to correct the
# low/medium confidence ordering bug. The branch is idempotent after v6.
NLP_VERSION="$(python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("extractor_version", ""))' "$PSEUDO.manifest.json")"
if [[ "$NLP_VERSION" == "clinical-report-labels-v5" ]]; then
  python scripts/migrate_qwen_v5_to_v6.py \
    --input "$PSEUDO" \
    --output "$PSEUDO.v6.tmp.csv" \
    --backup "$PSEUDO.v5.bak.csv"
  mv "$PSEUDO.v6.tmp.csv" "$PSEUDO"
  mv "$PSEUDO.v6.tmp.csv.manifest.json" "$PSEUDO.manifest.json"
elif [[ "$NLP_VERSION" != "clinical-report-labels-v6" ]]; then
  echo "Unsupported NLP extractor version: $NLP_VERSION" >&2
  exit 4
fi

echo "Launching resumable CoAtNet44 training from: $MODEL_DIR"
python src/main.py \
  --data_root "$DATA_ROOT" \
  --work_dir "$WORK_DIR" \
  --model_dir "$MODEL_DIR" \
  --nlp_engine vllm \
  --nlp_model Qwen/Qwen2.5-72B-Instruct \
  --cache_preset coatnet44 \
  --cache_prefix_name coatnet44 \
  --model_type coatnet_mil \
  --variant coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k \
  --timm_pooling hierarchical \
  --folds 0,1,2,3,4 \
  --epochs "${RSNA_EPOCHS:-12}" \
  --batch_size "${RSNA_BATCH_SIZE:-4}" \
  --grad_accum "${RSNA_GRAD_ACCUM:-2}" \
  --n_windows_train "${RSNA_WINDOWS:-16}" \
  --num_workers "${RSNA_NUM_WORKERS:-4}" \
  --prefetch_factor 1 \
  --preprocess_workers "${RSNA_PREPROCESS_WORKERS:-16}" \
  --eval_batch_size 8 \
  --inference_batch_size 2 \
  --n_tta 1 \
  --max_ram_gb 90
