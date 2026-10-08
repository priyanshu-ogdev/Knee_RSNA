import os
path = r'src\training\train.py'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

target1 = '''    va_gold = va.loc[gold_mask].reset_index(drop=True) if len(va) else va
    ds_tr = RSNADataset(tr, cache_prefix, cfg, True, n_windows_train, seed)
    ds_va = RSNADataset(va_gold, cache_prefix, cfg, False, None, seed, aug=False) if len(va_gold) else None
    ds_oof = RSNADataset(va, cache_prefix, cfg, False, None, seed, aug=False) if len(va) else None'''

repl1 = '''    ds_tr = RSNADataset(tr, cache_prefix, cfg, True, n_windows_train, seed)
    ds_oof = RSNADataset(va, cache_prefix, cfg, False, None, seed, aug=False) if len(va) else None
    ds_va = ds_oof'''

target2 = '''        save = (ds_va is None) or (np.isfinite(score) and score > best)
        if save:
            best = score if ds_va is not None else best
            epochs_no_improve = 0
            ckpt_file = os.path.join(out_dir, f"fold{fold}_best.pt")'''

repl2 = '''        save = True
        if save:
            best = score if ds_va is not None else best
            epochs_no_improve = 0
            ckpt_file = os.path.join(out_dir, f"fold{fold}_last.pt")'''

if target1 in text and target2 in text:
    text = text.replace(target1, repl1)
    text = text.replace(target2, repl2)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    print('Replaced in train.py')
else:
    print('Failed to replace in train.py')
