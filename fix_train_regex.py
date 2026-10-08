import os, re
path = r'src\training\train.py'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

text = re.sub(r'va_gold = va\.loc\[gold_mask\]\.reset_index\(drop=True\) if len\(va\) else va\s+ds_tr = RSNADataset\(tr, cache_prefix, cfg, True, n_windows_train, seed\)\s+ds_va = RSNADataset\(va_gold, cache_prefix, cfg, False, None, seed, aug=False\) if len\(va_gold\) else None',
    'ds_tr = RSNADataset(tr, cache_prefix, cfg, True, n_windows_train, seed)\n    ds_va = RSNADataset(va, cache_prefix, cfg, False, None, seed, aug=False) if len(va) else None', text)

text = re.sub(r'save = \(ds_va is None\) or \(np\.isfinite\(score\) and score > best\)\s+if save:\s+best = score if ds_va is not None else best\s+epochs_no_improve = 0',
    'save = True\n            if save:\n                best = score\n                epochs_no_improve = 0', text)

with open(path, 'w', encoding='utf-8') as f:
    f.write(text)
print('Regex replaced in train.py')
