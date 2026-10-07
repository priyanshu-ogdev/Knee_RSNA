"""Leakage-safe group folds.

Groups = (report language | scanner model): language is a strong site proxy in this dataset (e.g. every Cyrillic
report is a Philips Ingenia, every German/French/Greek/Dutch report is Siemens) and scanner model separates the
many sites hiding inside the 1,717 English reports. Audit numbers: 72 groups, largest 5.9% of studies, 24 groups
with <10 studies. Rare groups are merged into 'rare|<vendor>'. Corroborated duplicate exams are represented as
connected components and kept in the same fold.
Identical report TEXTS are deliberately NOT unioned into groups: templated English reports recur across sites and
would chain unrelated sites into one giant component.
"""
import os
import numpy as np
import pandas as pd
from collections import defaultdict

# the only true duplicate exams found by the audit (5 identical series each, Siemens Aera, non-gold)
KNOWN_DUP_PAIRS = [(
    '1.2.826.0.1.3680043.8.498.99725523560254732094432244556623376711',
    '1.2.826.0.1.3680043.8.498.99926624968240681772735102303531613209',
)]


def duplicate_pairs_from_series_meta(path, study_uids=None, min_shared_hashes=2):
    """Return only cross-study pairs corroborated by multiple distinct exact series hashes.

    The compact EDA dHash is collision-prone at single-hash resolution. Single matches
    are reported as candidates but never used to constrain folds.
    """
    if not path or not os.path.exists(path):
        allowed = set(map(str, study_uids)) if study_uids is not None else None
        pairs = [
            pair for pair in KNOWN_DUP_PAIRS
            if allowed is None or (pair[0] in allowed and pair[1] in allowed)
        ]
        return pairs, {
            "source": "known-verified-fallback",
            "candidate_pairs": 0,
            "accepted_pairs": len(pairs),
            "accepted_pair_ids": [list(pair) for pair in pairs],
            "min_shared_hashes": min_shared_hashes,
        }

    meta = pd.read_csv(path)
    required = {"StudyInstanceUID", "dhash"}
    if not required.issubset(meta.columns):
        raise ValueError(f"Duplicate metadata must contain {sorted(required)}: {path}")
    meta = meta[["StudyInstanceUID", "dhash"]].dropna()
    meta["StudyInstanceUID"] = meta["StudyInstanceUID"].astype(str).str.strip()
    meta["dhash"] = meta["dhash"].astype(str).str.strip().str.lower()
    if study_uids is not None:
        allowed = set(map(str, study_uids))
        meta = meta[meta["StudyInstanceUID"].isin(allowed)]
    hashes_by_study = meta.groupby("StudyInstanceUID")["dhash"].agg(lambda values: set(values))
    studies_by_hash = defaultdict(list)
    for study, hashes in hashes_by_study.items():
        for value in hashes:
            if value and value not in {"nan", "none"}:
                studies_by_hash[value].append(study)

    shared = defaultdict(set)
    for value, studies in studies_by_hash.items():
        if len(studies) < 2:
            continue
        studies = sorted(set(studies))
        for i, a in enumerate(studies):
            for b in studies[i + 1:]:
                shared[(a, b)].add(value)
    accepted = sorted(pair for pair, hashes in shared.items() if len(hashes) >= min_shared_hashes)
    return accepted, {
        "source": os.path.abspath(path),
        "candidate_pairs": len(shared),
        "single_hash_pairs": sum(len(hashes) == 1 for hashes in shared.values()),
        "accepted_pairs": len(accepted),
        "accepted_pair_ids": [list(pair) for pair in accepted],
        "min_shared_hashes": min_shared_hashes,
    }


def _duplicate_components(study_uids, dup_pairs):
    parent = {uid: uid for uid in study_uids}

    def find(uid):
        while parent[uid] != uid:
            parent[uid] = parent[parent[uid]]
            uid = parent[uid]
        return uid

    for a, b in dup_pairs:
        if a not in parent or b not in parent:
            continue
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return {uid: find(uid) for uid in study_uids}


def make_groups(study_meta, min_group=10, dup_pairs=None, n_splits=5):
    sm = study_meta.copy()
    g = sm['lang'].fillna('?').astype(str) + '|' + sm['model'].fillna('?').astype(str)
    cnt = g.map(g.value_counts())
    g = g.where(cnt >= min_group, 'rare|' + sm['vendor'].fillna('?').astype(str))
    cnt = g.map(g.value_counts())                       # a 'rare|<vendor>' bucket can itself be too small to stratify
    g = g.where(cnt >= n_splits, 'rare|ALL')
    sm['group'] = g
    pairs = dup_pairs if dup_pairs is not None else KNOWN_DUP_PAIRS
    components = _duplicate_components(sm["StudyInstanceUID"].astype(str).tolist(), pairs)
    sm["duplicate_group"] = sm["StudyInstanceUID"].astype(str).map(components)
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
    components = sm["duplicate_group"].astype(str)
    component_ids = components.drop_duplicates().tolist()
    if len(component_ids) < 2 * n_splits:
        order = np.random.default_rng(seed).permutation(component_ids)
        component_fold = {component: i % n_splits for i, component in enumerate(order)}
        sm["fold"] = components.map(component_fold).astype(int)
        return sm[['StudyInstanceUID', 'group', 'duplicate_group', 'fold'] + [c for c in ('lang', 'model', 'vendor', 'gold') if c in sm.columns]]
    if scheme == 'group':
        site_groups = sm["group"].astype(str).tolist()
        site_parent = {f"site:{group}": f"site:{group}" for group in site_groups}

        def find_site(group):
            while site_parent[group] != group:
                site_parent[group] = site_parent[site_parent[group]]
                group = site_parent[group]
            return group

        uid_to_group = dict(zip(sm["StudyInstanceUID"].astype(str), site_groups))
        for a, b in (dup_pairs if dup_pairs is not None else KNOWN_DUP_PAIRS):
            if a not in uid_to_group or b not in uid_to_group:
                continue
            ga, gb = f"site:{uid_to_group[a]}", f"site:{uid_to_group[b]}"
            ra, rb = find_site(ga), find_site(gb)
            if ra != rb:
                site_parent[max(ra, rb)] = min(ra, rb)
        validation_groups = np.array(
            [find_site(f"site:{group}") for group in site_groups]
        )
        if len(set(validation_groups)) < n_splits:
            raise ValueError(
                f"Cannot create {n_splits} site-held-out folds from "
                f"{len(set(validation_groups))} independent site/duplicate groups"
            )
        from sklearn.model_selection import StratifiedGroupKFold
        y = sm['vendor'].fillna('?').astype('category').cat.codes.values
        for k, (_, va) in enumerate(
            StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(
                sm, y, validation_groups
            )
        ):
            fold[va] = k
    elif scheme == 'site':
        from sklearn.model_selection import StratifiedKFold
        rep = sm.drop_duplicates("duplicate_group").copy().reset_index(drop=True)
        counts = rep['group'].value_counts()
        too_small = counts[counts < n_splits].index
        if len(too_small) > 0 and len(counts) > 0:
            majority_grp = counts.index[0]
            rep['group'] = rep['group'].replace({g: majority_grp for g in too_small})
        y = rep['group'].astype('category').cat.codes.values
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        ufold = {}
        for k, (_, va) in enumerate(skf.split(rep, y)):
            for component in rep.loc[va, 'duplicate_group']:
                ufold[component] = k
        fold = components.map(ufold).to_numpy(dtype=int)
    else:
        raise ValueError(scheme)
    sm['fold'] = fold
    return sm[['StudyInstanceUID', 'group', 'duplicate_group', 'fold'] + [c for c in ('lang', 'model', 'vendor', 'gold') if c in sm.columns]]


def make_study_meta(index_df, train_csv=None, labels_df=None):
    """Study-level table (vendor, scanner model, report language, gold flag, metadata-poor flag) straight from the
    index + train.csv, so the notebook never needs the audit's study_meta.csv."""
    from .text import guess_lang, vendor_of
    if index_df is None or len(index_df) == 0:
        df = pd.DataFrame()
    else:
        df = index_df[index_df['split'] == 'train'] if 'split' in index_df.columns else index_df
    if df.empty or 'StudyInstanceUID' not in df.columns:
        if labels_df is not None and 'StudyInstanceUID' in labels_df.columns:
            studies = pd.Index(labels_df['StudyInstanceUID'].unique(), name='StudyInstanceUID')
        elif train_csv is not None and 'StudyInstanceUID' in train_csv.columns:
            studies = pd.Index(train_csv['StudyInstanceUID'].unique(), name='StudyInstanceUID')
        else:
            studies = pd.Index([], name='StudyInstanceUID')
        sm = pd.DataFrame({'vendor': '?', 'model': '?', 'poor': False}, index=studies).reset_index()
        sm['lang'] = '?'
        if train_csv is not None and 'Report' in train_csv.columns:
            lang = train_csv.set_index('StudyInstanceUID')['Report'].map(guess_lang)
            sm['lang'] = sm['StudyInstanceUID'].map(lang).fillna('?')
        sm['gold'] = False
        if labels_df is not None:
            import src.core.config as config
            if 'source' in labels_df.columns:
                has = labels_df['source'] == 'gold'
            else:
                has = labels_df[[t for t in config.TARGETS if t in labels_df.columns]].notna().any(axis=1)
            gold = labels_df.assign(_g=has).set_index('StudyInstanceUID')['_g']
            sm['gold'] = sm['StudyInstanceUID'].map(gold).fillna(False)
        return sm
    mode = lambda s: s.dropna().mode().iloc[0] if s.notna().any() else np.nan
    g = df.groupby('StudyInstanceUID')
    studies = pd.Index(df['StudyInstanceUID'].unique(), name='StudyInstanceUID')
    vendor = g['Manufacturer'].agg(lambda s: mode(s.map(vendor_of))) if 'Manufacturer' in df else pd.Series('?', index=studies)
    model = g['ManufacturerModelName'].agg(mode) if 'ManufacturerModelName' in df else pd.Series('?', index=studies)
    poor = g['RepetitionTime'].agg(lambda s: s.isna().all()) if 'RepetitionTime' in df else pd.Series(False, index=studies)
    sm = pd.DataFrame({'vendor': vendor, 'model': model, 'poor': poor}, index=studies).reset_index()
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
