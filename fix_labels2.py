import os, re
path = r'src\data\labels.py'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

# Locate the start and end of the block
start_idx = text.find("    for t in config.TARGETS:")
end_idx = text.find("    if out_csv:")

new_block = '''    for t in config.TARGETS:
        g = gold[t].values
        y = g.copy()
        w = np.where(np.isfinite(g), 1.0, 0.0)
        if ex is not None and t in ex.columns:
            e = ex[t].astype(float).values
            
            # SOTA Calibration: Estimate P(positive | not_stated) using Gold Studies
            is_soft_neg = (e == -1.0)
            if is_soft_neg.any():
                gold_valid = g[np.isfinite(g)]
                if len(gold_valid) > 0:
                    gold_prevalence = np.mean(gold_valid)
                else:
                    gold_prevalence = 0.05
                calibrated_prob = min(0.15, gold_prevalence * 0.8) # Conservative penalty
                e[is_soft_neg] = calibrated_prob
            
            if np.isinf(e).any():
                raise ValueError(f"extra labels for {t!r} must be finite or missing")
            finite = e[np.isfinite(e)]
            if not np.logical_and(finite >= 0.0, finite <= 1.0).all():
                raise ValueError(f"extra labels for {t!r} must be in [0, 1]")
            ew = ex[f'{t}_weight'].astype(float).fillna(1.0).values if f'{t}_weight' in ex.columns else 1.0
            if np.isscalar(ew):
                pass
            elif not np.isfinite(ew).all() or (ew < 0).any() or (ew > 1).any():
                raise ValueError(f"extra confidence weights for {t!r} must be finite and in [0, 1]")
            use = ~np.isfinite(g) & np.isfinite(e)
            y = np.where(use, e, g)
            w = np.where(use, extra_weight * ew, w)
            out.loc[use & (out['source'] == 'none'), 'source'] = 'extra'
        out[t] = y
        out[f'{t}_weight'] = w
'''

text = text[:start_idx] + new_block + text[end_idx:]

with open(path, 'w', encoding='utf-8') as f:
    f.write(text)
print('Fixed labels.py block')
