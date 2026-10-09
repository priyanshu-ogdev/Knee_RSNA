import os
import time
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from src.core.config import resolve_data_root
from src.data.preprocess.nlp_extractor import auto_complete_extraction, TARGETS, detect_language

def main():
    parser = argparse.ArgumentParser(description="Evaluate NLP pseudo-label extraction against gold-annotated studies")
    parser.add_argument("--data_root", type=str, default="data", help="Path to raw dataset directory")
    parser.add_argument("--engine", type=str, default="vllm", choices=["vllm"], help="Inference engine")
    parser.add_argument("--model", type=str, default=None, help="HuggingFace model ID or local directory path")
    parser.add_argument("--force", action="store_true", default=True, help="Force re-extraction")
    args = parser.parse_args()

    args.data_root = resolve_data_root(args.data_root)
    train_df = pd.read_csv(os.path.join(args.data_root, 'train.csv'))
    
    # Isolate all 58 Gold studies
    gold_df = train_df[train_df['ACL'].notna()].copy()
    print(f"Isolated {len(gold_df)} gold studies for evaluation.")
    
    temp_dir = "temp_gold_eval"
    os.makedirs(temp_dir, exist_ok=True)
    gold_df.to_csv(os.path.join(temp_dir, 'train.csv'), index=False)
    
    out_csv = os.path.join(temp_dir, 'gold_extractions.csv')
    
    print(f"Running NLP Extractor (Engine: {args.engine}, Model: {args.model or 'default'})...")
    start_time = time.time()
    
    out_path, stats = auto_complete_extraction(
        data_root=temp_dir,
        out_csv=out_csv,
        model_id=args.model,
        engine=args.engine,
        force=args.force,
        evaluate=True
    )
    
    elapsed = time.time() - start_time
    extracted_df = pd.read_csv(out_path)
    print(f"Extraction completed in {elapsed:.2f} seconds.")
    print(f"Average time per report: {elapsed / max(1, len(gold_df)):.2f} seconds.")
    
    merged = pd.merge(gold_df, extracted_df, on="StudyInstanceUID", suffixes=("_true", "_pred"))
    merged['token_length'] = merged['Report'].apply(lambda x: len(str(x)) / 4)
    print("\nToken Length Distribution:")
    print(merged['token_length'].describe())
    
    merged['Language'] = merged['Report'].apply(detect_language)
    
    print("\n" + "=" * 60)
    print(f"OVERALL METRICS ({args.engine.upper()} ENGINE)")
    print("=" * 60)
    
    target_aucs = []
    for t in TARGETS:
        mask = merged[f"{t}_weight"].values > 0.0
        y_true = merged[f"{t}_true"].values[mask]
        y_pred = merged[f"{t}_pred"].values[mask]
        if len(set(y_true)) > 1:
            auc = roc_auc_score(y_true, y_pred)
            f1 = f1_score(y_true, (y_pred >= 0.5).astype(int))
            target_aucs.append(auc)
            print(f"{t:18s}: AUC = {auc:.4f}, F1 = {f1:.4f}")
        else:
            print(f"{t:18s}: AUC = N/A (single class in gold set)")
            
    if target_aucs:
        print(f"\n--> MACRO ROC-AUC: {sum(target_aucs)/len(target_aucs):.4f}")
    
    print("\n" + "=" * 60)
    print("METRICS BY LANGUAGE (MACRO AUC)")
    print("=" * 60)
    
    for lang in sorted(merged['Language'].unique()):
        lang_df = merged[merged['Language'] == lang]
        print(f"\nLanguage: {lang} (N={len(lang_df)})")
        lang_aucs = []
        for t in TARGETS:
            mask_l = lang_df[f"{t}_weight"].values > 0.0
            y_true_l = lang_df[f"{t}_true"].values[mask_l]
            y_pred_l = lang_df[f"{t}_pred"].values[mask_l]
            if len(set(y_true_l)) > 1:
                try:
                    auc = roc_auc_score(y_true_l, y_pred_l)
                    lang_aucs.append(auc)
                except ValueError:
                    pass
        if lang_aucs:
            print(f"  Macro AUC: {sum(lang_aucs)/len(lang_aucs):.4f}")
        else:
            print("  Macro AUC: N/A (not enough class variance)")

if __name__ == "__main__":
    main()
