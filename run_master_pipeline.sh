#!/bin/bash
# ==============================================================================
# RSNA Knee 2026: The Hardened Data Processing & Training Master Pipeline
# ==============================================================================
# This script handles the end-to-end execution of the final fortified system
# on your DGX Cluster. It triggers:
# 1. Gold-Standard NLP Evaluation (Auditing the model against true labels)
# 2. Complete Phase 1 (16k context NLP extraction) and Phase 2 (Image Caching)
# 3. Phase 3 (Multi-GPU Multi-Fold Model Training)
# ==============================================================================

set -e # Exit immediately on any error

# ------------------------------------------------------------------------------
# 0. Safety Cleanup (Guaranteeing 100% Fresh Run)
# ------------------------------------------------------------------------------
echo "[STEP 0] Purging old cache data to ensure a completely fresh DGX run..."
rm -rf temp_gold_eval data/nlp_extractions* data/cache* data/pseudo_labels.csv
echo "Cleanup complete!"
echo "------------------------------------------------------------------------------"

# ------------------------------------------------------------------------------
# 1. Gold Label Evaluation (The Authenticity Check)
# ------------------------------------------------------------------------------
# This proves the CoT logic, json parsing, and translation guards work flawlessly.
echo "[STEP 1] Running Hardened NLP Pipeline on Gold Standard Studies..."
export DEFAULT_MAX_MODEL_LEN=16384
export NLP_CHUNK_SIZE=150

python src/data/preprocess/nlp_extractor.py \
    --evaluate \
    --engine vllm \
    --model Qwen/Qwen2.5-72B-Instruct \
    --force

echo "Gold standard analysis complete. The AUC/Accuracy logs have been saved."
echo "------------------------------------------------------------------------------"

# ------------------------------------------------------------------------------
# 2 & 3. End-To-End Master Pipeline (Preprocessing + Multi-GPU Training)
# ------------------------------------------------------------------------------
# main.py internally routes to nlp_extractor.py (Phase 1), cache.py (Phase 2),
# and train.py (Phase 3). 
echo "[STEP 2 & 3] Commencing End-To-End Extraction, Cache Build, and 5-Fold Training..."

# NOTE: main.py uses --force_nlp and --fresh_preprocessing to trigger the 
# extraction and cache building cleanly from scratch for the massive train set.
python src/main.py \
    --force_nlp \
    --fresh_preprocessing \
    --nlp_engine vllm \
    --nlp_model Qwen/Qwen2.5-72B-Instruct \
    --nlp_batch_size 150 \
    --model_type timm_mil \
    --variant convnext_small.in12k_ft_in1k \
    --timm_pooling hierarchical \
    --epochs 12 \
    --batch_size 128 \
    --grad_accum 1 \
    --num_workers 16 \
    --preprocess_workers 32

echo "=============================================================================="
echo "RUN COMPLETE! The final trained weights and OOF telemetry are in your working directory."
echo "=============================================================================="
