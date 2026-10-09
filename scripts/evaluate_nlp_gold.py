#!/usr/bin/env python
"""RSNA Knee MSK Radiology Gold-Standard Evaluation Runner.

Delegates directly to the unified NLP evaluation engine in src.data.preprocess.nlp_extractor.
"""
import os
import sys
import argparse

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.data.preprocess.nlp_extractor import run_gold_evaluation


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Gold Evaluation Suite for NLP Pseudo-Label Extractor"
    )
    parser.add_argument("--data_root", type=str, default="data", help="Path to raw dataset directory containing train.csv")
    parser.add_argument("--engine", type=str, default="vllm", choices=["vllm"], help="Inference engine (vllm)")
    parser.add_argument("--model", type=str, default=None, help="HuggingFace model ID or local snapshot directory path")
    parser.add_argument("--force", action="store_true", help="Force re-extraction of gold reports via LLM")
    parser.add_argument("--show_errors", action=argparse.BooleanOptionalAction, default=True, help="Print detailed diagnostic audit of any gold label disagreements")
    args = parser.parse_args()

    run_gold_evaluation(
        data_root=args.data_root,
        engine=args.engine,
        model_id=args.model,
        force=args.force,
        show_errors=args.show_errors,
    )


if __name__ == "__main__":
    main()
