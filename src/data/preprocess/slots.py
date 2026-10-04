"""Series -> acquisition-slot assignment.

Annotation constants are the public 0.943 notebook's (regression-tested against it on all 4,407 train studies:
SAG/COR/AX fluid-FS slots filled for 94.3 / 95.5 / 99.4 % of studies).

What the audit added:
  * the CSV `Fat_Suppression` flag agrees with the header-derived fat-sat evidence for 99.14% of series, and
    `Fluid_Sensitive` is IDENTICAL to it in train (it tracks fat-suppression, not PD/T2 contrast), so the CSV fluid
    flag is only used as a last-resort proxy;
  * the metadata-poor cohort (238 studies, no TR/TE/ScanOptions) still has descriptions, so header rules work;
  * hidden-set robustness: every missing input degrades to a weaker signal instead of an exception;
  * 3.0% of 'most slices' picks are 3-D volumes -> `prefer_2d` picks a 2-D series whenever one exists.
"""
import re
import numpy as np
import pandas as pd

import src.core.config as config

FATSAT_OPTS = {'FS', 'FATSAT', 'FAT_SAT', 'FSAT'}
_FATSAT_RX = re.compile('\\bfs\\b|fatsat|fat sat|\\bstir\\b|\\bspair\\b|\\bspir\\b|\\bwe\\b|water excit|\\btirm\\b|'
                        '\\bsting\\b|\\bfatsup\\b|smart fat|\\bwater\\b')
_T1_RX = re.compile('\\bt1\\b|\\bt1w\\b')
_T2_RX = re.compile('\\bt2\\b|\\bt2w\\b')
_PD_RX = re.compile('\\bpd\\b|\\bpdw\\b|proton|\\bdp\\b|dens')
_SEP = re.compile('[_\\-.]')

NEEDED = ['SeriesDescription', 'SequenceName', 'ScanOptions', 'ScanningSequence', 'RepetitionTime', 'EchoTime']


def to_flag(x):
    """Robust 0/1/NaN from CSV values (hidden-set CSV formatting is not guaranteed)."""
    if x is None:
        return np.nan
    if isinstance(x, (bool, np.bool_)):
        return float(bool(x))
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x) if x in (0, 1) else np.nan
    s = str(x).strip().lower()
    if s in ('1', 'true', 'yes', 'y', 't', '1.0'):
        return 1.0
    if s in ('0', 'false', 'no', 'n', 'f', '0.0'):
        return 0.0
    return np.nan


def annotate(df, fs_priority='hdr', csv_fallback=True):
    """Add fatsat / weight / fluid / informative / is3d columns. Returns a copy.

    fatsat: header evidence (description regex or ScanOptions token) as in the public pipeline. When the header
    has no usable text at all (dummy/missing description AND no ScanOptions) the CSV flag is used instead; with
    fs_priority='csv' the CSV flag wins whenever present.
    """
    df = df.copy()
    for c in NEEDED:
        if c not in df.columns:
            df[c] = None
    raw = df['SeriesDescription'].fillna('').astype(str)
    dummy = raw.str.contains('dummyseriesdesc', case=False)
    desc = (raw.where(~dummy, '') + ' ' + df['SequenceName'].fillna('').astype(str)).str.lower()
    desc = desc.str.replace(_SEP, ' ', regex=True)
    opts = df['ScanOptions'].fillna('').astype(str).str.upper().str.split('|')
    opts_fs = opts.apply(lambda ts: any(t.strip() in FATSAT_OPTS for t in ts))
    fs_hdr = desc.str.contains(_FATSAT_RX) | opts_fs
    informative = (desc.str.strip() != '') | (df['ScanOptions'].fillna('').astype(str).str.strip() != '')

    csv_fs = pd.Series(np.nan, index=df.index)
    if 'Fat_Suppression' in df.columns:
        csv_fs = df['Fat_Suppression'].map(to_flag)
    elif 'Fluid_Sensitive' in df.columns:
        # FIX: Test set might drop Fat_Suppression; use Fluid_Sensitive as identical proxy (per EDA)
        csv_fs = df['Fluid_Sensitive'].map(to_flag)
    if fs_priority == 'csv':
        fs = np.where(csv_fs.notna(), csv_fs == 1.0, fs_hdr)
        src = np.where(csv_fs.notna(), 'csv', 'hdr')
    else:
        use_csv = ((~informative) & csv_fs.notna()) if csv_fallback else pd.Series(False, index=df.index)
        fs = np.where(use_csv, csv_fs == 1.0, fs_hdr)
        src = np.where(use_csv, 'csv', 'hdr')
    df['fatsat'] = fs.astype(bool)
    df['fs_source'] = src
    df['fs_conflict'] = csv_fs.notna() & ((csv_fs.fillna(0).astype(float) == 1.0) != df['fatsat'])

    tr = pd.to_numeric(df['RepetitionTime'], errors='coerce')
    te = pd.to_numeric(df['EchoTime'], errors='coerce')
    gre = df['ScanningSequence'].fillna('').astype(str).str.upper().str.contains('GR')
    t1, t2, pdw = desc.str.contains(_T1_RX), desc.str.contains(_T2_RX), desc.str.contains(_PD_RX)
    w = np.where(t1 & ~t2 & ~pdw, 'T1', np.where(t2 & ~pdw, 'T2', np.where(pdw, 'PD', np.where(
        gre, 'GRE', np.where(tr < 800, 'T1', np.where(te > 60, 'T2', np.where(tr >= 800, 'PD', 'UNK')))))))
    df['weight'] = w
    fluid = np.isin(w, ['PD', 'T2'])
    # last resort: no header contrast at all -> the CSV fluid flag (== fat-sat flag in train) as a proxy
    if csv_fallback and 'Fluid_Sensitive' in df.columns:
        cf = df['Fluid_Sensitive'].map(to_flag)
        unk = (w == 'UNK') & cf.notna()
        fluid = np.where(unk, cf == 1.0, fluid)
    df['fluid'] = fluid.astype(bool)

    acq = df['MRAcquisitionType'].fillna('').astype(str).str.upper() if 'MRAcquisitionType' in df.columns else ''
    ns = pd.to_numeric(df['n_slices'] if 'n_slices' in df.columns else df['n_files'], errors='coerce').fillna(0)
    th = pd.to_numeric(df.get('SliceThickness', np.nan), errors='coerce')
    df['is3d'] = (acq == '3D') | ((ns > 120) & (th < 1.5))
    return df


def plane_of(df):
    """CSV plane (verified 100% = DICOM geometry) with geometry as the fallback."""
    p = df['Anatomical_Plane'] if 'Anatomical_Plane' in df.columns else pd.Series(None, index=df.index)
    g = df['plane_geo'] if 'plane_geo' in df.columns else pd.Series(None, index=df.index)
    ok = p.isin(['Sagittal', 'Coronal', 'Axial'])
    return p.where(ok, g)


def assign_slots(g, prefer_2d=True, slots=None):
    """g: annotated rows of ONE study (needs plane, fatsat, fluid, is3d, n_slices, SeriesInstanceUID).

    Returns (chosen, alts, extras):
      chosen: {slot_name: SeriesInstanceUID}
      alts:   {slot_name: [other candidates of the same slot type, best first]}   (view-swap augmentation pool)
      extras: series not used by any slot
    Rule (public): most slices wins. With prefer_2d a 2-D series always beats a 3-D volume.
    """
    slots = slots or config.SLOTS
    chosen, alts, used = {}, {}, set()
    gg = g.assign(_n=pd.to_numeric(g['n_slices'], errors='coerce').fillna(0))
    for name, plane, fluid, fs in slots:
        sel = (gg['plane'] == plane) & (gg['fatsat'] == fs)
        if fluid is not None:
            sel &= (gg['fluid'] == fluid)
        c = gg[sel]
        if len(c) == 0:
            continue
        if prefer_2d:
            c = c.assign(_k=c['is3d'].astype(int))
            c = c.sort_values(['_k', '_n', 'SeriesInstanceUID'], ascending=[True, False, True])
        else:
            c = c.sort_values(['_n', 'SeriesInstanceUID'], ascending=[False, True])
        chosen[name] = c.iloc[0]['SeriesInstanceUID']
        alts[name] = [s for s in c['SeriesInstanceUID'].tolist()[1:]]
        used.add(chosen[name])
    extras = [s for s in gg['SeriesInstanceUID'] if s not in used]
    return chosen, alts, extras


def assign_all(index_df, prefer_2d=True, fs_priority='hdr', csv_fallback=True):
    """Annotate the whole index and assign slots for every study (fully vectorised: ~O(n log n), no per-study loop).

    Returns (annotated_df, slot_table):
      slot_table: DataFrame indexed by StudyInstanceUID, one column per slot holding a SeriesInstanceUID or None;
      slot_table.attrs['alts']: {slot: {study: [alternate SeriesInstanceUIDs]}} for view-swap augmentation.
    Tie-break is deterministic (series UID), unlike a stable sort over directory order."""
    df = index_df.copy()
    df['plane'] = plane_of(df)
    df = annotate(df, fs_priority, csv_fallback)
    ns = pd.to_numeric(df['n_slices'] if 'n_slices' in df.columns else df['n_files'], errors='coerce').fillna(0)
    usable = df['plane'].isin(['Sagittal', 'Coronal', 'Axial']) & (ns > 0)
    work = df.assign(_n=ns, _k=(df['is3d'].astype(int) if prefer_2d else 0))[usable]
    work = work.sort_values(['StudyInstanceUID', '_k', '_n', 'SeriesInstanceUID'], ascending=[True, True, False, True])
    studies = pd.Index(df['StudyInstanceUID'].unique(), name='StudyInstanceUID')
    tab = pd.DataFrame(index=studies, columns=[n for n, *_ in config.SLOTS], dtype=object)
    alts = {}
    for name, plane, fluid, fs in config.SLOTS:
        m = (work['plane'] == plane) & (work['fatsat'] == fs)
        if fluid is not None:
            m &= (work['fluid'] == fluid)
        c = work[m]
        first = ~c.duplicated('StudyInstanceUID')
        tab.loc[c.loc[first, 'StudyInstanceUID'].values, name] = c.loc[first, 'SeriesInstanceUID'].values
        rest = c[~first]
        alts[name] = rest.groupby('StudyInstanceUID')['SeriesInstanceUID'].apply(list).to_dict() if len(rest) else {}
    tab = tab.where(tab.notna(), None)
    tab.attrs['alts'] = alts
    return df, tab
