import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
import argparse
from src.core.config import resolve_data_root
import time
from src.data.preprocess.nlp_extractor import auto_complete_extraction, TARGETS

def detect_language(report: str) -> str:
    r_lower = report.lower()
    if 'sağlam' in r_lower or 'yırtık' in r_lower or 'eklem' in r_lower:
        return 'Turkish'
    if 'ruptura' in r_lower or 'meniskus' in r_lower or 'ligament' in r_lower:
        return 'Croatian/Serbian'
    if 'разрыв' in r_lower or 'связка' in r_lower or 'мениск' in r_lower:
        return 'Russian'
    if 'ρήξη' in r_lower or 'άρθρωση' in r_lower:
        return 'Greek'
    if 'rotura' in r_lower or 'derrame' in r_lower:
        return 'Spanish'
    if 'scheur' in r_lower or 'geen' in r_lower:
        return 'Dutch'
    if 'ruptur' in r_lower or 'erguss' in r_lower or 'kein' in r_lower:
        return 'German'
    if 'rupture' in r_lower or 'épanchement' in r_lower or 'sans' in r_lower:
        return 'French'
    return 'English'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--engine", type=str, default="vllm", choices=["vllm"])
    args = parser.parse_args()

    args.data_root = resolve_data_root(args.data_root)
    train_df = pd.read_csv(os.path.join(args.data_root, 'train.csv'))
    
    # Isolate 58 Gold studies
    gold_df = train_df[train_df['ACL'].notna()].copy()
    print(f"Isolated {len(gold_df)} gold studies for evaluation.")
    
    temp_dir = "temp_gold_eval"
    os.makedirs(temp_dir, exist_ok=True)
    gold_df.to_csv(os.path.join(temp_dir, 'train.csv'), index=False)
    
    out_csv = os.path.join(temp_dir, 'gold_extractions.csv')
    
    print(f"Running NLP Extractor (Engine: {args.engine})...")
    start_time = time.time()
    
    # Pass evaluate=True to bypass the fully_labeled_mask skip logic
    out_path, stats = auto_complete_extraction(
        data_root=temp_dir,
        out_csv=out_csv,
        engine=args.engine,
        force=True,
        evaluate=True
    )
    
    elapsed = time.time() - start_time
    extracted_df = pd.read_csv(out_path)
    print(f"Extraction completed in {elapsed:.2f} seconds.")
    print(f"Average time per report: {elapsed / len(gold_df):.2f} seconds.")
    
    merged = pd.merge(gold_df, extracted_df, on="StudyInstanceUID", suffixes=("_true", "_pred"))
    
    merged['token_length'] = merged['Report'].apply(lambda x: len(str(x)) / 4)
    print("\nToken Length Distribution:")
    print(merged['token_length'].describe())
    
    merged['Language'] = merged['Report'].apply(detect_language)
    
    print("\n" + "="*50)
    print(f"OVERALL METRICS ({args.engine.upper()} ENGINE)")
    print("="*50)
    
    target_aucs = []
    for t in TARGETS:
        mask = merged[f"{t}_weight_pred"].values > 0.0
        y_true = merged[f"{t}_true"].values[mask]
        y_pred = merged[f"{t}_pred"].values[mask]
        if len(set(y_true)) > 1:
            auc = roc_auc_score(y_true, y_pred)
            f1 = f1_score(y_true, (y_pred >= 0.5).astype(int))
            target_aucs.append(auc)
            print(f"{t}: AUC = {auc:.4f}, F1 = {f1:.4f}")
        else:
            print(f"{t}: AUC = N/A (only 1 class in gold set)")
            
    if target_aucs:
        print(f"MACRO ROC-AUC: {sum(target_aucs)/len(target_aucs):.4f}")
    
    print("\n" + "="*50)
    print("METRICS BY LANGUAGE (MACRO AUC)")
    print("="*50)
    
    for lang in merged['Language'].unique():
        lang_df = merged[merged['Language'] == lang]
        print(f"\nLanguage: {lang} (N={len(lang_df)})")
        lang_aucs = []
        for t in TARGETS:
            mask_l = lang_df[f"{t}_weight_pred"].values > 0.0
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
