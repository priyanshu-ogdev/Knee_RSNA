"""Pixel pipeline: decode -> physical crop -> robust normalise -> resample -> uint8 stack.

Throughput design (each choice measured/justified in tools/bench.py):
  * decode RAW integers, crop FIRST, convert only the cropped window to float32
  * slab averaging of thin 3-D slices happens at crop resolution, once per target slice
  * percentiles on a 2x2-subsampled stack (exact percentiles for the 'public' preset)
  * cv2 resize (SIMD) instead of per-slice torch calls; cv2/BLAS threads pinned to 1 inside workers
Dataset-driven rules (audit): physical mm crop with each series' own row/column spacing (7.3% of matrices are
non-square, 0.39% have non-square pixels, 640x1280 wide frames exist), border-median padding (72% of series have a
non-zero noise floor), negatives clipped (241 Canon series), per-series robust scale (22x spread), heavy tails
(8.2% of series have p99.9 > 2x p99).
"""
import os
import numpy as np
import cv2

from . import dicomio as dio

cv2.setNumThreads(0)                    # parallelism comes from worker processes; avoid oversubscription
PS_FALLBACK = 0.3125                    # dataset median in-plane spacing (mm), only if the header has none


# --------------------------------------------------------------------------- z sampling
def z_groups(pos, cfg, spacing, method='position'):
    """Which native slices feed each of the cfg.stack_depth stored slices.

    Returns (groups: list[list[int]], valid: bool[D]).
      'index' (public): linspace over cfg.band of the ordered slice indices, duplicates padded
      'mm': physical grid cfg.z_step_mm centred on the series; nearest native slice, or the mean of all native
            slices inside +-step/2 when the native spacing is finer than 0.75*step (3-D / thin-slice series)
    """
    n = len(pos)
    D = cfg.stack_depth
    if cfg.z_mode == 'index' or method != 'position':
        # public linspace over the band; also the safe fallback when slice order came from InstanceNumber/name
        # (positions are then index units, not mm, so a millimetre grid would be meaningless)
        band = cfg.band if cfg.z_mode == 'index' else (0.0, 1.0)
        lo, hi = int(band[0] * (n - 1)), int(band[1] * (n - 1))
        idx = np.unique(np.linspace(lo, hi, D).astype(int)) if hi > lo else np.array([n // 2])
        while len(idx) < D:
            idx = np.append(idx, idx[-1])
        return [[int(i)] for i in idx[:D]], np.ones(D, bool)
    pos = np.asarray(pos, np.float64)
    step = cfg.z_step_mm
    mid = 0.5 * (pos[0] + pos[-1])
    targets = mid + (np.arange(D) - (D - 1) / 2.0) * step
    sp = spacing if np.isfinite(spacing) and spacing > 0 else step
    tol = 0.75 * max(sp, step)
    near = np.clip(np.searchsorted(pos, targets), 1, max(n - 1, 1))
    if n == 1:
        near = np.zeros(D, int)
    else:
        left, right = pos[near - 1], pos[near]
        near = np.where(np.abs(targets - left) <= np.abs(right - targets), near - 1, near)
    dist = np.abs(pos[near] - targets)
    valid = dist <= tol
    groups = []
    slab = sp < 0.75 * step
    for t, k in zip(targets, near):
        if slab:
            lo_i = int(np.searchsorted(pos, t - step / 2.0, 'left'))
            hi_i = int(np.searchsorted(pos, t + step / 2.0, 'right'))
            groups.append(list(range(lo_i, hi_i)) if hi_i > lo_i else [int(k)])
        else:
            groups.append([int(k)])
    return groups, valid


# --------------------------------------------------------------------------- crop geometry
def _border_median(a):
    h, w = a.shape
    b = max(2, int(0.02 * min(h, w)))
    ring = np.concatenate([a[:b].ravel(), a[-b:].ravel(), a[b:-b, :b].ravel(), a[b:-b, -b:].ravel()])
    return float(np.median(ring))


def _fg_center(raws):
    """Foreground bbox centre (row, col) in full-frame pixels, from the mean of the decoded slices."""
    ref = np.mean([np.maximum(a[::4, ::4].astype(np.float32), 0) for a in raws], axis=0)
    p99 = np.percentile(ref, 99)
    if p99 <= 0:
        return None
    fg = ref > 0.1 * p99
    if not fg.any():
        return None
    rr, cc = np.where(fg.any(1))[0], np.where(fg.any(0))[0]
    return 4.0 * (rr[0] + rr[-1]) / 2.0, 4.0 * (cc[0] + cc[-1]) / 2.0


def crop_plan(shape, ps_row, ps_col, cfg, raws=None):
    """-> (y0, x0, Hc, Wc) of the physical crop window in full-frame pixels (may exceed the frame -> padded),
    or None when the public rule skips the crop."""
    H, W = shape
    ok = np.isfinite(ps_row) and np.isfinite(ps_col) and ps_row > 0 and ps_col > 0
    if cfg.pad_mode == 'public':                                   # public rule, verbatim
        if not (ok and ps_row > 0):
            return None
        want = int(round(cfg.crop_mm / ps_row))
        if 16 < want < min(H, W):
            cy, cx = H // 2, W // 2
            half = want // 2
            return max(0, cy - half), max(0, cx - half), min(H, cy + half) - max(0, cy - half), \
                min(W, cx + half) - max(0, cx - half)
        return None
    if not ok:
        ps_row = ps_col = PS_FALLBACK
    Hc = max(16, int(round(cfg.crop_mm / ps_row)))
    Wc = max(16, int(round(cfg.crop_mm / ps_col)))
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    aspect = (W * ps_col) / (H * ps_row)
    if cfg.wide_center and raws and (aspect > cfg.wide_ratio or aspect < 1.0 / cfg.wide_ratio):
        c = _fg_center(raws)
        if c is not None:
            if aspect > 1:
                cx = c[1]
            else:
                cy = c[0]
    y0 = int(round(cy - (Hc - 1) / 2.0))
    x0 = int(round(cx - (Wc - 1) / 2.0))
    return y0, x0, Hc, Wc


def _crop_float(raw, slope, icpt, plan, bg_fn):
    """Crop (with border-median padding) and convert ONLY the window to float32."""
    H, W = raw.shape
    if plan is None:
        return raw.astype(np.float32) * slope + icpt
    y0, x0, Hc, Wc = plan
    ys0, xs0, ys1, xs1 = max(y0, 0), max(x0, 0), min(y0 + Hc, H), min(x0 + Wc, W)
    win = raw[ys0:ys1, xs0:xs1].astype(np.float32) * slope + icpt
    if (ys1 - ys0, xs1 - xs0) == (Hc, Wc):
        return win
    out = np.full((Hc, Wc), bg_fn(raw, slope, icpt), np.float32)
    out[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0] = win
    return out


# --------------------------------------------------------------------------- normalise / resize
def normalise(vol, valid, cfg):
    """Stack-wide robust scaling to [0,1] (all slices share one window; invalid slices untouched)."""
    if cfg.clip_negative:
        np.maximum(vol, 0, out=vol)
    v = vol[valid]
    sub = v if cfg.name == 'public' else v[:, ::2, ::2]
    lo, hi = np.percentile(sub, [cfg.norm_lo, cfg.norm_hi])
    vol -= lo
    vol /= max(float(hi - lo), 1e-6)
    np.clip(vol, 0.0, 1.0, out=vol)
    return vol


def resize_stack(vol, size, mode):
    """[D,h,w] float -> [D,size,size] float. 'area' = antialiased when shrinking, bilinear when enlarging."""
    D, h, w = vol.shape
    if mode == 'area' and (h > size or w > size):
        interp = cv2.INTER_AREA
    else:
        interp = cv2.INTER_LINEAR
    out = np.empty((D, size, size), np.float32)
    for i in range(D):
        out[i] = cv2.resize(vol[i], (size, size), interpolation=interp)
    return out


def to_uint8(x):
    x *= 255.0
    np.rint(x, out=x)
    return x.astype(np.uint8)


# --------------------------------------------------------------------------- one series -> stack
def build_stack(rec, cfg):
    """rec: index row (dict-like) with dir, ordered_files, positions, nsign, ps_row, ps_col, spacing_med.

    -> (uint8 [D,S,S], valid bool [D], info) or (None, None, info) when nothing could be decoded.
    Never raises: every slice failure is recorded in info['errors']."""
    info = dict(errors=[], padded=False, ps_missing=False, n_decoded=0)
    files = list(rec['ordered_files'])
    pos = np.asarray(rec['positions'], np.float64)
    if not files:
        info['errors'].append('no_files')
        return None, None, info
    if cfg.order_mode == 'normal' and float(rec.get('nsign', 1.0)) < 0:     # public sign convention
        files, pos = files[::-1], -pos[::-1]
    method = str(rec.get('order_method', 'position'))
    groups, valid = z_groups(pos, cfg, float(rec.get('spacing_med', np.nan)), method)
    info['z_fallback'] = cfg.z_mode == 'mm' and method != 'position'
    need = sorted({i for g in groups for i in g})
    raws = {}
    for i in need:
        try:
            raws[i] = dio.decode_raw(os.path.join(rec['dir'], files[i]))
        except Exception as e:
            info['errors'].append(f'{files[i]}: {type(e).__name__}')
    info['n_decoded'] = len(raws)
    if not raws:
        return None, None, info
    shape = max({r[0].shape for r in raws.values()}, key=lambda s: sum(r[0].shape == s for r in raws.values()))
    raws = {i: r for i, r in raws.items() if r[0].shape == shape}
    ps_row, ps_col = float(rec.get('ps_row', np.nan)), float(rec.get('ps_col', np.nan))
    info['ps_missing'] = not (np.isfinite(ps_row) and np.isfinite(ps_col) and ps_row > 0 and ps_col > 0)
    plan = crop_plan(shape, ps_row, ps_col, cfg, [r[0] for r in raws.values()] if cfg.wide_center else None)
    if plan is not None:
        info['padded'] = plan[0] < 0 or plan[1] < 0 or plan[0] + plan[2] > shape[0] or plan[1] + plan[3] > shape[1]
    def _bg(raw, slope, icpt):
        return _border_median(raw.astype(np.float32) * slope + icpt)

    D = len(groups)
    crops = {}
    for i, (raw, slope, icpt) in raws.items():
        crops[i] = _crop_float(raw, slope, icpt, plan, _bg)
    ref_i = next(iter(crops))
    vol = np.zeros((D,) + crops[ref_i].shape, np.float32)
    ok = np.zeros(D, bool)
    for d, g in enumerate(groups):
        got = [crops[i] for i in g if i in crops]
        if not got:
            continue
        vol[d] = got[0] if len(got) == 1 else np.mean(got, axis=0, dtype=np.float32)
        ok[d] = True
    valid = valid & ok
    if cfg.z_mode == 'index' or method != 'position':   # failed slices borrow the nearest decoded neighbour
        for d in np.where(~ok)[0]:
            j = int(np.argmin(np.where(ok, np.abs(np.arange(D) - d), 10 ** 6)))
            vol[d] = vol[j]
        valid = np.ones(D, bool)
    if not valid.any():
        return None, None, info
    vol = normalise(vol, valid, cfg)
    vol = resize_stack(vol, cfg.img_size, cfg.resize_mode)
    out = to_uint8(vol)
    out[~valid] = 0
    return out, valid, info
