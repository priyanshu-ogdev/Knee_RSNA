"""Leakage-safe group folds.

Groups = (report language | scanner model): language is a strong site proxy in this dataset (e.g. every Cyrillic
report is a Philips Ingenia, every German/French/Greek/Dutch report is Siemens) and scanner model separates the
many sites hiding inside the 1,717 English reports. Audit numbers: 72 groups, largest 5.9% of studies, 24 groups
with <10 studies. Rare groups are merged into 'rare|<vendor>'. Known duplicate exams share a group.
Identical report TEXTS are deliberately NOT unioned into groups: templated English reports recur across sites and
would chain unrelated sites into one giant component.
"""
import numpy as np
import pandas as pd

# the only true duplicate exams found by the audit (5 identical series each, Siemens Aera, non-gold)
KNOWN_DUP_PAIRS = [(
    '1.2.826.0.1.3680043.8.498.99725523560254732094432244556623376711',
    '1.2.826.0.1.3680043.8.498.99926624968240681772735102303531613209',
)]


def make_groups(study_meta, min_group=10, dup_pairs=None, n_splits=5):
    sm = study_meta.copy()
    g = sm['lang'].fillna('?').astype(str) + '|' + sm['model'].fillna('?').astype(str)
    cnt = g.map(g.value_counts())
    g = g.where(cnt >= min_group, 'rare|' + sm['vendor'].fillna('?').astype(str))
    cnt = g.map(g.value_counts())                       # a 'rare|<vendor>' bucket can itself be too small to stratify
    g = g.where(cnt >= n_splits, 'rare|ALL')
    sm['group'] = g
    for a, b in (dup_pairs if dup_pairs is not None else KNOWN_DUP_PAIRS):
        if a in set(sm['StudyInstanceUID']) and b in set(sm['StudyInstanceUID']):
            ga = sm.loc[sm['StudyInstanceUID'] == a, 'group'].iloc[0]
            sm.loc[sm['StudyInstanceUID'] == b, 'group'] = ga
    return sm


def group_folds(study_meta, n_splits=5, seed=2026, min_group=10, dup_pairs=None, scheme='site'):
    """-> DataFrame[StudyInstanceUID, group, fold, ...].

    scheme='site'  (default, for model selection): every site/scanner group is spread evenly over all folds
                   (stratified by group) - mirrors the hidden test, which is drawn from the same sites as train.
                   Duplicate exams are kept together.
    scheme='group' (robustness check): whole groups are held out together = unseen-scanner generalisation.
                   Pessimistic by construction: audit shows 205/238 metadata-poor studies land in one fold and
                   most languages are absent from some training folds."""
    sm = make_groups(study_meta, min_group, dup_pairs, n_splits)
    fold = np.full(len(sm), -1)
    if len(sm) < 2 * n_splits:                           # tiny datasets (smoke tests): round-robin, still dup-safe
        order = np.random.default_rng(seed).permutation(len(sm))
        fold[order] = np.arange(len(sm)) % n_splits
        sm['fold'] = fold
        return sm[['StudyInstanceUID', 'group', 'fold'] + [c for c in ('lang', 'model', 'vendor', 'gold') if c in sm.columns]]
    if scheme == 'group':
        from sklearn.model_selection import StratifiedGroupKFold
        y = sm['vendor'].fillna('?').astype('category').cat.codes.values
        for k, (_, va) in enumerate(StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(sm, y, sm['group'])):
            fold[va] = k
    elif scheme == 'site':
        from sklearn.model_selection import StratifiedKFold
        uid = sm['StudyInstanceUID'].values
        partner = {}
        for a, b in (dup_pairs if dup_pairs is not None else KNOWN_DUP_PAIRS):
            partner[b] = a                                           # b follows a
        unit = np.array([partner.get(u, u) for u in uid])
        first = ~pd.Series(unit).duplicated().values                 # one representative row per unit
        rep = sm[first].reset_index(drop=True)
        y = rep['group'].astype('category').cat.codes.values
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        ufold = {}
        for k, (_, va) in enumerate(skf.split(rep, y)):
            for u in rep.loc[va, 'StudyInstanceUID']:
                ufold[u] = k
        fold = np.array([ufold[u] for u in unit])
    else:
        raise ValueError(scheme)
    sm['fold'] = fold
    return sm[['StudyInstanceUID', 'group', 'fold'] + [c for c in ('lang', 'model', 'vendor', 'gold') if c in sm.columns]]


def make_study_meta(index_df, train_csv=None, labels_df=None):
    """Study-level table (vendor, scanner model, report language, gold flag, metadata-poor flag) straight from the
    index + train.csv, so the notebook never needs the audit's study_meta.csv."""
    from .text import guess_lang, vendor_of
    df = index_df[index_df['split'] == 'train'] if 'split' in index_df.columns else index_df
    mode = lambda s: s.dropna().mode().iloc[0] if s.notna().any() else np.nan
    g = df.groupby('StudyInstanceUID')
    sm = pd.DataFrame({
        'vendor': g['Manufacturer'].agg(lambda s: mode(s.map(vendor_of))) if 'Manufacturer' in df else '?',
        'model': g['ManufacturerModelName'].agg(mode) if 'ManufacturerModelName' in df else '?',
        'poor': g['RepetitionTime'].agg(lambda s: s.isna().all()) if 'RepetitionTime' in df else False,
    }).reset_index()
    sm['lang'] = '?'
    if train_csv is not None and 'Report' in train_csv.columns:
        lang = train_csv.set_index('StudyInstanceUID')['Report'].map(guess_lang)
        sm['lang'] = sm['StudyInstanceUID'].map(lang).fillna('?')
    if labels_df is not None:
        import src.core.config as config
        if 'source' in labels_df.columns:                       # build_labels() marks gold / extra / none
            has = labels_df['source'] == 'gold'
        else:
            has = labels_df[[t for t in config.TARGETS if t in labels_df.columns]].notna().any(axis=1)
        gold = labels_df.assign(_g=has).set_index('StudyInstanceUID')['_g']
        sm['gold'] = sm['StudyInstanceUID'].map(gold).fillna(False)
    else:
        sm['gold'] = False
    return sm
