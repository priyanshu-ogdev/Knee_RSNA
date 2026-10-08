import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
import time
from src.data.preprocess.nlp_extractor import extract_by_rules, TARGETS

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
    train_df = pd.read_csv(os.path.join("D:/Knee_RSNA/data", 'train.csv'))
    gold_df = train_df[train_df['ACL'].notna()].copy()
    
    print(f"Isolated {len(gold_df)} gold studies for evaluation.")
    
    start_time = time.time()
    
    extracted_rows = []
    for idx, row in gold_df.iterrows():
        out = extract_by_rules(row['Report'], row['StudyInstanceUID'])
        extracted_rows.append(out)
        
    extracted_df = pd.DataFrame(extracted_rows)
    
    elapsed = time.time() - start_time
    print(f"Extraction completed in {elapsed:.2f} seconds.")
    print(f"Average time per report: {elapsed / len(gold_df):.4f} seconds.")
    
    merged = pd.merge(gold_df, extracted_df, on="StudyInstanceUID", suffixes=("_true", "_pred"))
    
    merged['token_length'] = merged['Report'].apply(lambda x: len(str(x)) / 4)
    print("\nToken Length Distribution:")
    print(merged['token_length'].describe())
    
    merged['Language'] = merged['Report'].apply(detect_language)
    
    print("\n" + "="*50)
    print("OVERALL METRICS (RULES ENGINE)")
    print("="*50)
    
    target_aucs = []
    for t in TARGETS:
        y_true = merged[f"{t}_true"].values
        y_pred = merged[f"{t}_pred"].values
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
            y_true_l = lang_df[f"{t}_true"].values
            y_pred_l = lang_df[f"{t}_pred"].values
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
