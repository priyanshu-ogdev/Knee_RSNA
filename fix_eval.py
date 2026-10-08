import os, sys

path = r'src\data\preprocess\nlp_extractor.py'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

start_eval = text.find("    if args.evaluate:")
end_eval = text.find("    else:\n        OUT = os.path.join(DATA_ROOT, \"pseudo_labels.csv\")")

new_eval = '''    if args.evaluate:
        train_df = pd.read_csv(os.path.join(DATA_ROOT, 'train.csv'))
        # Evaluate on the 58 gold studies.
        gold_df = train_df[train_df['ACL'].notna()].copy()
        
        # Keep 10 studies untouched for final checks
        np.random.seed(42)
        untouched_idx = np.random.choice(gold_df.index, size=min(10, len(gold_df)), replace=False)
        eval_df = gold_df.drop(untouched_idx)
        print(f"Isolated {len(eval_df)} gold studies for evaluation. (Held out 10 for final checks).")
        
        temp_dir = "temp_gold_eval"
        os.makedirs(temp_dir, exist_ok=True)
        eval_df.to_csv(os.path.join(temp_dir, 'train.csv'), index=False)
        out_csv = os.path.join(temp_dir, 'gold_extractions.csv')
        
        print(f"Running NLP Extractor (Engine: {args.engine})...")
        import time
        start_time = time.time()
        out_path, stats = auto_complete_extraction(
            data_root=temp_dir, out_csv=out_csv, engine=args.engine,
            model_id=args.model, force=args.force, evaluate=True
        )
        elapsed = time.time() - start_time
        extracted_df = pd.read_csv(out_path)
        print(f"Extraction completed in {elapsed:.2f} seconds.")
        
        merged = pd.merge(eval_df, extracted_df, on="StudyInstanceUID", suffixes=("_true", "_pred"))
        merged['Language'] = merged['Report'].apply(detect_language)
        
        from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
        
        for lang in ['ALL'] + list(merged['Language'].unique()):
            print("\\n" + "="*80)
            if lang == 'ALL':
                lang_df = merged
                print(f"OVERALL METRICS (N={len(lang_df)})")
            else:
                lang_df = merged[merged['Language'] == lang]
                print(f"METRICS FOR LANGUAGE: {lang} (N={len(lang_df)})")
            print("="*80)
            
            macro_aucs = []
            for t in TARGETS:
                y_true = lang_df[f"{t}_true"].values
                y_pred = lang_df[f"{t}_pred"].values
                # -1.0 means not_stated. We drop not_stated for precision/recall of "stated entries".
                # Also drop NaNs in true.
                valid_mask = (~np.isnan(y_true)) & (y_pred >= 0.0)
                y_true_valid = y_true[valid_mask]
                y_pred_valid = y_pred[valid_mask]
                
                coverage = np.sum(y_pred >= 0.0) / len(y_pred) if len(y_pred) > 0 else 0
                
                if len(y_true_valid) > 0:
                    preds_binary = (y_pred_valid >= 0.5).astype(int)
                    true_binary = (y_true_valid >= 0.5).astype(int)
                    p, r, f1, _ = precision_recall_fscore_support(true_binary, preds_binary, average='binary', zero_division=0)
                    
                    try:
                        auc = roc_auc_score(true_binary, y_pred_valid)
                        macro_aucs.append(auc)
                    except ValueError:
                        auc = float('nan')
                        
                    print(f"{t:18s} | Cov: {coverage*100:5.1f}% | AUC: {auc:5.3f} | P: {p:5.3f} | R: {r:5.3f} | Disagreements: {np.sum(preds_binary != true_binary)}")
                else:
                    print(f"{t:18s} | Cov: {coverage*100:5.1f}% | AUC:   N/A | P:   N/A | R:   N/A")
            
            if macro_aucs:
                print(f"\\n--> MACRO AUC for {lang}: {np.nanmean(macro_aucs):.4f}")
                
'''

if start_eval != -1 and end_eval != -1:
    text = text[:start_eval] + new_eval + text[end_eval:]
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    print("Fixed evaluate logic successfully")
else:
    print("Could not find start/end bounds for evaluate block")
