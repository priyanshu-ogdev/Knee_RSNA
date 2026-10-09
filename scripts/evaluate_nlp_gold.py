import os
import time
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_fscore_support
from src.core.config import resolve_data_root
from src.data.preprocess.nlp_extractor import auto_complete_extraction, TARGETS, detect_language


def calculate_clinical_metrics(y_true, y_pred, threshold=0.5):
    """Calculates sensitivity, specificity, precision, F1, and disagreement counts."""
    y_true_b = (y_true >= threshold).astype(int)
    y_pred_b = (y_pred >= threshold).astype(int)
    
    tp = int(np.sum((y_true_b == 1) & (y_pred_b == 1)))
    fp = int(np.sum((y_true_b == 0) & (y_pred_b == 1)))
    fn = int(np.sum((y_true_b == 1) & (y_pred_b == 0)))
    tn = int(np.sum((y_true_b == 0) & (y_pred_b == 0)))
    
    sensitivity = tp / max(1, tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / max(1, tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / max(1, tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * (precision * sensitivity) / max(1e-6, precision + sensitivity)
    disagreements = fp + fn
    
    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "sens": sensitivity, "spec": specificity, "prec": precision, "f1": f1,
        "disagreements": disagreements
    }


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Gold Evaluation Suite for NLP Pseudo-Label Extractor"
    )
    parser.add_argument("--data_root", type=str, default="data", help="Path to raw dataset directory containing train.csv")
    parser.add_argument("--engine", type=str, default="vllm", choices=["vllm"], help="Inference engine (vllm)")
    parser.add_argument("--model", type=str, default=None, help="HuggingFace model ID or local snapshot directory path")
    parser.add_argument("--force", action="store_true", help="Force re-extraction of gold reports via LLM")
    parser.add_argument("--show_errors", action="store_true", default=True, help="Print detailed diagnostic audit of any gold label disagreements")
    args = parser.parse_args()

    args.data_root = resolve_data_root(args.data_root)
    train_path = os.path.join(args.data_root, 'train.csv')
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"train.csv not found at {train_path}")

    train_df = pd.read_csv(train_path)
    
    # Isolate all 58 Gold studies with expert ground truth
    gold_df = train_df[train_df['ACL'].notna()].copy()
    print("=" * 80)
    print(f"RSNA KNEE MSK RADIOLOGY EVALUATION - {len(gold_df)} GOLD STUDIES")
    print(f"Data Root: {args.data_root} | Engine: {args.engine.upper()} | Model: {args.model or 'Default'}")
    print("=" * 80)
    
    temp_dir = "temp_gold_eval"
    os.makedirs(temp_dir, exist_ok=True)
    gold_df.to_csv(os.path.join(temp_dir, 'train.csv'), index=False)
    
    out_csv = os.path.join(temp_dir, 'gold_extractions.csv')
    
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
    print(f"\n[TIMING] Extraction finished in {elapsed:.2f}s ({elapsed / max(1, len(gold_df)):.2f}s per report).")
    
    merged = pd.merge(gold_df, extracted_df, on="StudyInstanceUID", suffixes=("_true", "_pred"))
    merged['Language'] = merged['Report'].apply(detect_language)
    merged['token_length'] = merged['Report'].apply(lambda x: len(str(x)) / 4)
    
    print("\n[DATASET PROFILE]")
    print(f"Total Evaluated Studies: {len(merged)}")
    print("Report Languages Represented:")
    for lang, cnt in merged['Language'].value_counts().items():
        print(f"  - {lang:18s}: {cnt:2d} reports ({cnt/len(merged)*100:4.1f}%)")
    
    # =========================================================================
    # DUAL-VIEW METRIC CALCULATION
    # View 1: Stated Clinical Coverage (LLM Comprehension on Addressed Findings)
    # View 2: End-to-End Calibrated AUC (Aligned with build_labels MNAR prior)
    # =========================================================================
    
    print("\n" + "=" * 96)
    print("CLINICAL COMPREHENSION & ROC-AUC MATRIX (ACROSS ALL 12 TARGETS)")
    print("=" * 96)
    print(f"{'Target':<18} | {'Cov%':<6} | {'Stated AUC':<10} | {'Calib AUC':<10} | {'Sens':<6} | {'Spec':<6} | {'Prec':<6} | {'F1':<6} | {'Disagreements'}")
    print("-" * 96)
    
    stated_aucs = []
    calibrated_aucs = []
    
    # Store disagreements for forensic auditing
    disagreement_records = []
    
    for t in TARGETS:
        y_true = merged[f"{t}_true"].values
        y_pred_raw = merged[f"{t}_pred"].values
        y_weight = merged[f"{t}_weight"].values if f"{t}_weight" in merged.columns else np.ones_like(y_true)
        
        # Stated filter: structure was addressed in the report (weight > 0 and pred >= 0)
        stated_mask = (y_weight > 0.0) & (y_pred_raw >= 0.0)
        coverage_pct = (np.sum(stated_mask) / len(y_true)) * 100.0
        
        # 1. Stated AUC
        y_true_stated = y_true[stated_mask]
        y_pred_stated = y_pred_raw[stated_mask]
        stated_auc = float('nan')
        if len(set(y_true_stated)) > 1:
            try:
                stated_auc = roc_auc_score(y_true_stated, y_pred_stated)
                stated_aucs.append(stated_auc)
            except ValueError:
                pass
                
        # 2. Calibrated AUC (aligns with labels.py MNAR calibration for unstated findings)
        gold_valid = y_true[np.isfinite(y_true)]
        gold_prevalence = np.mean(gold_valid) if len(gold_valid) > 0 else 0.05
        # Unstated soft-negative prior
        calibrated_soft_neg = min(0.15, gold_prevalence * 0.8)
        
        y_pred_calibrated = y_pred_raw.copy()
        # Map unstated (-1.0 or masked 0.0 with 0.0 weight) to calibrated soft-negative
        unstated_mask = (y_pred_raw < 0.0) | ((y_pred_raw == 0.0) & (y_weight == 0.0))
        y_pred_calibrated[unstated_mask] = calibrated_soft_neg
        
        calib_auc = float('nan')
        if len(set(y_true)) > 1:
            try:
                calib_auc = roc_auc_score(y_true, y_pred_calibrated)
                calibrated_aucs.append(calib_auc)
            except ValueError:
                pass
                
        # Clinical classification metrics on stated findings
        if len(y_true_stated) > 0:
            m = calculate_clinical_metrics(y_true_stated, y_pred_stated, threshold=0.5)
            s_auc_str = f"{stated_auc:.4f}" if np.isfinite(stated_auc) else "  N/A  "
            c_auc_str = f"{calib_auc:.4f}" if np.isfinite(calib_auc) else "  N/A  "
            print(
                f"{t:<18} | {coverage_pct:5.1f}% | {s_auc_str:<10} | {c_auc_str:<10} | "
                f"{m['sens']:5.3f} | {m['spec']:5.3f} | {m['prec']:5.3f} | {m['f1']:5.3f} | "
                f"{m['disagreements']:2d} (TP:{m['TP']} FP:{m['FP']} FN:{m['FN']} TN:{m['TN']})"
            )
        else:
            s_auc_str = "  N/A  "
            c_auc_str = f"{calib_auc:.4f}" if np.isfinite(calib_auc) else "  N/A  "
            print(f"{t:<18} | {coverage_pct:5.1f}% | {s_auc_str:<10} | {c_auc_str:<10} |   N/A  |   N/A  |   N/A  |   N/A  |  0")
            
        # Record disagreements for forensic audit
        for idx, row in merged.iterrows():
            yt = row[f"{t}_true"]
            yp = row[f"{t}_pred"]
            yw = row[f"{t}_weight"] if f"{t}_weight" in row else 1.0
            if (yp >= 0.0) and (yw > 0.0):
                pred_bin = 1 if yp >= 0.5 else 0
                true_bin = int(yt)
                if pred_bin != true_bin:
                    disagreement_records.append({
                        "UID": row["StudyInstanceUID"],
                        "Target": t,
                        "True": true_bin,
                        "Pred": yp,
                        "Lang": row["Language"],
                        "Report": str(row["Report"])[:120] + "..."
                    })
                    
    print("-" * 96)
    macro_stated = np.nanmean(stated_aucs) if stated_aucs else float('nan')
    macro_calib = np.nanmean(calibrated_aucs) if calibrated_aucs else float('nan')
    print(f"{'MACRO ROC-AUC':<18} |        | {macro_stated:8.4f}   | {macro_calib:8.4f}   |")
    print("=" * 96)
    
    # =========================================================================
    # MULTI-LINGUAL BREAKDOWN
    # =========================================================================
    print("\n" + "=" * 80)
    print("PERFORMANCE BREAKDOWN BY REPORT LANGUAGE")
    print("=" * 80)
    print(f"{'Language':<18} | {'N':<4} | {'Avg Words':<10} | {'Macro Stated AUC':<18} | {'Macro Calib AUC'}")
    print("-" * 80)
    
    for lang in sorted(merged['Language'].unique()):
        lang_df = merged[merged['Language'] == lang]
        l_stated = []
        l_calib = []
        for t in TARGETS:
            y_t = lang_df[f"{t}_true"].values
            y_p = lang_df[f"{t}_pred"].values
            y_w = lang_df[f"{t}_weight"].values if f"{t}_weight" in lang_df.columns else np.ones_like(y_t)
            
            # Stated
            st_m = (y_w > 0.0) & (y_p >= 0.0)
            if len(set(y_t[st_m])) > 1:
                try:
                    l_stated.append(roc_auc_score(y_t[st_m], y_p[st_m]))
                except ValueError:
                    pass
                    
            # Calibrated
            if len(set(y_t)) > 1:
                y_p_c = y_p.copy()
                g_prev = np.mean(y_t[np.isfinite(y_t)]) if len(y_t) > 0 else 0.05
                u_m = (y_p < 0.0) | ((y_p == 0.0) & (y_w == 0.0))
                y_p_c[u_m] = min(0.15, g_prev * 0.8)
                try:
                    l_calib.append(roc_auc_score(y_t, y_p_c))
                except ValueError:
                    pass
                    
        s_res = f"{np.nanmean(l_stated):.4f}" if l_stated else "   N/A   "
        c_res = f"{np.nanmean(l_calib):.4f}" if l_calib else "   N/A   "
        avg_w = np.mean(lang_df['Report'].astype(str).apply(lambda x: len(x.split())))
        print(f"{lang:<18} | {len(lang_df):2d}   | {avg_w:8.1f}   | {s_res:<18} | {c_res}")
        
    print("=" * 80)
    
    # =========================================================================
    # FORENSIC DISAGREEMENT AUDIT
    # =========================================================================
    if args.show_errors and disagreement_records:
        print(f"\n[FORENSIC AUDIT] {len(disagreement_records)} CLINICAL DISAGREEMENTS IDENTIFIED:")
        for i, r in enumerate(disagreement_records[:15]):
            print(f"  [{i+1:2d}] Target: {r['Target']:<16} | True: {r['True']} vs Pred: {r['Pred']:.2f} | Lang: {r['Lang']}")
            print(f"       UID:    {r['UID']}")
            print(f"       Report: {r['Report']}")
        if len(disagreement_records) > 15:
            print(f"  ... and {len(disagreement_records) - 15} more.")
    elif not disagreement_records:
        print("\n[FORENSIC AUDIT] 100% PERFECT CONCORDANCE! Zero disagreements on stated findings.")


if __name__ == "__main__":
    main()
