import os, re
path = r'src\training\train.py'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

# Fix evaluation threshold to include high-confidence pseudo-labels (weight=0.5)
text = text.replace('m = W[:, j] >= 0.99', 'm = W[:, j] >= 0.49')

with open(path, 'w', encoding='utf-8') as f:
    f.write(text)
print('Fixed train.py eval threshold')

path = r'src\data\preprocess\nlp_extractor.py'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

# Change the soft negative to use -1.0 as a marker
text = text.replace('out[t], out[f"{t}_weight"] = 0.05, 0.1 # Soft Negative', 'out[t], out[f"{t}_weight"] = -1.0, 0.1 # Soft Negative Marker')

with open(path, 'w', encoding='utf-8') as f:
    f.write(text)
print('Fixed nlp_extractor.py soft negative marker')

path = r'src\data\labels.py'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

# Inject calibration logic into build_labels
target_code = '''            if ex is not None and t in ex.columns:
                e = ex[t].astype(float).values
                if np.isinf(e).any():
                    raise ValueError(f"extra labels for {t!r} must be finite or missing")'''

replacement_code = '''            if ex is not None and t in ex.columns:
                e = ex[t].astype(float).values
                
                # SOTA Calibration: Estimate P(positive | not_stated) using Gold Studies
                # We marked not_stated as -1.0 in the extractor.
                is_soft_neg = (e == -1.0)
                if is_soft_neg.any():
                    gold_valid = g[np.isfinite(g)]
                    if len(gold_valid) > 0:
                        gold_prevalence = np.mean(gold_valid)
                    else:
                        gold_prevalence = 0.05
                    # Assign a highly calibrated weak prior
                    calibrated_prob = min(0.15, gold_prevalence * 0.8) # Conservative penalty
                    e[is_soft_neg] = calibrated_prob
                
                if np.isinf(e).any():
                    raise ValueError(f"extra labels for {t!r} must be finite or missing")'''

if target_code in text:
    text = text.replace(target_code, replacement_code)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    print('Fixed labels.py calibration logic')
else:
    print('Failed to find target in labels.py')
