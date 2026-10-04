"""Slice ordering, orientation, laterality.

Why this exists (verified on the dataset): file names are pydicom-generated random UIDs, so neither
`sorted(os.listdir())` nor any filename order carries anatomical position (observed P(first file == spatial
centre) = 3.67% vs 3.51% expected for random order, z = 1.36, on all 24,371 series). Slice order must come from
ImagePositionPatient projected on the slice normal.
"""
from collections import Counter
import numpy as np

import src.core.config as config

_LET = {(0, 1): 'L', (0, -1): 'R', (1, 1): 'P', (1, -1): 'A', (2, 1): 'S', (2, -1): 'I'}
_AXIS = {'L': 0, 'R': 0, 'P': 1, 'A': 1, 'S': 2, 'I': 2}
PLANES = ('Sagittal', 'Coronal', 'Axial')


# --------------------------------------------------------------------------- orientation
def orient_info(iop):
    """-> (plane from slice normal, |normal| alignment with its axis, 2-letter LPS code).

    Code letter 1 = direction image COLUMNS increase toward; letter 2 = direction image ROWS increase toward.
    Dataset standard: Sagittal 'PI', Coronal 'LI', Axial 'LP'.
    """
    if iop is None or len(iop) != 6:
        return None, float('nan'), None
    r, c = np.asarray(iop[:3], float), np.asarray(iop[3:], float)
    n = np.cross(r, c)
    nn = np.linalg.norm(n)
    if not np.isfinite(nn) or nn < 1e-6:
        return None, float('nan'), None
    n = n / nn
    plane = PLANES[int(np.argmax(np.abs(n)))]
    code = ''
    for v in (r, c):
        ax = int(np.argmax(np.abs(v)))
        code += _LET[(ax, 1 if v[ax] >= 0 else -1)]
    return plane, float(np.abs(n).max()), code


def slice_normal(iop):
    r, c = np.asarray(iop[:3], float), np.asarray(iop[3:], float)
    n = np.cross(r, c)
    return n / (np.linalg.norm(n) + 1e-12)


def center_xyz(ipp, iop, ps, rows, cols):
    """Patient-space centre of a slice (mm)."""
    ipp, iop = np.asarray(ipp, float), np.asarray(iop, float)
    return ipp + iop[:3] * ps[1] * (cols - 1) / 2.0 + iop[3:] * ps[0] * (rows - 1) / 2.0


# --------------------------------------------------------------------------- ordering
def order_records(recs, order_mode='position'):
    """Order one series' slices by anatomical position.

    recs: list of dicts {name, ipp, iop, inst, rows, cols, ps}  (None where a tag is missing)
    order_mode:
      'position' : key = sign(n[axis]) * dot(ipp, n), ascending  -> increasing along +L / +P / +S
                   (direction is independent of the vendor's IOP sign convention)
      'normal'   : key = dot(ipp, r x c) ascending (public-notebook convention, no sign normalisation)
    Falls back to InstanceNumber, then natural name order, and reports which one was used.
    Files whose matrix differs from the majority shape are dropped (and counted).
    Returns dict(ordered, pos, method, n_dropped_shape, plane, code, align, spacing_med, z_extent, shape, ps,
                 center_x, rep_index) or None when there are no readable files.
    """
    recs = [r for r in recs if r is not None]
    if not recs:
        return None
    shapes = Counter((r['rows'], r['cols']) for r in recs)
    shape = shapes.most_common(1)[0][0]
    n_drop = sum(1 for r in recs if (r['rows'], r['cols']) != shape)
    recs = [r for r in recs if (r['rows'], r['cols']) == shape]
    iops = Counter(tuple(np.round(r['iop'], 3)) for r in recs if r['iop'] is not None)
    iop = np.array(iops.most_common(1)[0][0]) if iops else None
    plane, align, code = orient_info(iop) if iop is not None else (None, float('nan'), None)

    have_pos = [r for r in recs if r['ipp'] is not None]
    n = len(recs)
    nsign = 1.0
    # every branch builds tuples (key, tiebreak, name, rec); `pos` is in mm only for method == 'position'
    if iop is not None and len(have_pos) >= max(2, int(0.8 * n)):
        nrm = slice_normal(iop)
        ax = int(np.argmax(np.abs(nrm)))
        nsign = 1.0 if nrm[ax] >= 0 else -1.0
        sgn = nsign if order_mode == 'position' else 1.0
        spare = float(np.median([sgn * float(np.dot(r['ipp'], nrm)) for r in have_pos]))
        keyed = [(sgn * float(np.dot(r['ipp'], nrm)) if r['ipp'] is not None else spare,
                  r['inst'] if r['inst'] is not None else float('inf'), r['name'], r) for r in recs]
        method = 'position'
    elif sum(r['inst'] is not None for r in recs) >= max(2, int(0.8 * n)):
        keyed = [(r['inst'] if r['inst'] is not None else float('inf'), 0.0, r['name'], r) for r in recs]
        method = 'instance'
    else:
        keyed = [(0.0, 0.0, r['name'], r) for r in recs]
        method = 'name'
    keyed.sort(key=lambda t: (t[0], t[1], t[2]))
    ordered = [k[2] for k in keyed]
    pos = np.array([k[0] for k in keyed], dtype=np.float64) if method != 'name' else np.arange(n, dtype=np.float64)
    dp = np.diff(pos) if len(pos) > 1 else np.array([])
    spacing = float(np.median(np.abs(dp))) if len(dp) else float('nan')
    rep = len(ordered) // 2
    rep_rec = keyed[rep][3]
    ps = rep_rec['ps'] if rep_rec['ps'] else (float('nan'), float('nan'))
    cx = float('nan')
    if rep_rec['ipp'] is not None and iop is not None and rep_rec['ps']:
        try:
            cx = float(center_xyz(rep_rec['ipp'], iop, rep_rec['ps'], shape[0], shape[1])[0])
        except Exception:
            cx = float('nan')
    return dict(ordered=ordered, pos=pos, method=method, n_dropped_shape=n_drop, plane=plane, code=code,
                align=align, spacing_med=spacing, z_extent=float(pos[-1] - pos[0]) if len(pos) > 1 else 0.0,
                shape=shape, ps=ps, center_x=cx, rep_index=rep, nsign=nsign)


# --------------------------------------------------------------------------- laterality
def _side_from_tag(v):
    if v is None:
        return None
    s = str(v).strip().upper()
    return s[0] if s and s[0] in ('L', 'R') else None


def resolve_laterality(rows, min_offset=None):
    """Study-level side of the knee.

    rows: iterable of dicts with optional 'Laterality', 'ImageLaterality' (strings) and 'center_x' (mm).
    Priority: DICOM tag (majority vote) -> patient-x of the image centre (median over series, |x| >= min_offset:
    x<0 -> Right, else Left) -> unresolved ('U').
    Returns dict(side in {'L','R','U'}, source, x_med, disagree).
    """
    min_offset = config.LAT_MIN_OFFSET_MM if min_offset is None else min_offset
    tags = []
    xs = []
    for r in rows:
        for k in ('Laterality', 'ImageLaterality'):
            s = _side_from_tag(r.get(k))
            if s:
                tags.append(s)
        x = r.get('center_x')
        if x is not None and np.isfinite(x):
            xs.append(float(x))
    xm = float(np.median(xs)) if xs else float('nan')
    geo = None
    if np.isfinite(xm) and abs(xm) >= min_offset:
        geo = 'R' if xm < 0 else 'L'
    if tags:
        side = Counter(tags).most_common(1)[0][0]
        return dict(side=side, source='tag', x_med=xm, disagree=bool(geo and geo != side))
    if geo:
        return dict(side=geo, source='geometry', x_med=xm, disagree=False)
    return dict(side='U', source='none', x_med=xm, disagree=False)


# --------------------------------------------------------------------------- canonicalisation
def reorient_stack(stack, code, plane):
    """Map a [D,H,W] stack to the dataset-standard orientation code of `plane` (CANON_ORIENT).

    Handles axis swaps (transpose) and sign flips. A no-op for all 24,371 train series (single code per plane)."""
    target = config.CANON_ORIENT.get(plane)
    if not code or len(code) != 2 or not target or code == target:
        return stack, False
    r, c = code
    tr, tc = target
    if _AXIS[r] != _AXIS[tr]:
        stack = stack.transpose(0, 2, 1)
        r, c = c, r
    if r != tr:
        stack = stack[:, :, ::-1]
    if c != tc:
        stack = stack[:, ::-1, :]
    return np.ascontiguousarray(stack), True


def canonical_side(stack, plane, side):
    """Mirror right knees so they look like left knees.

    Coronal/Axial: flip image columns. Sagittal: reverse slice order (slices run medial<->lateral)."""
    if side != 'R':
        return stack
    if plane in ('Coronal', 'Axial'):
        return np.ascontiguousarray(stack[:, :, ::-1])
    return np.ascontiguousarray(stack[::-1])
