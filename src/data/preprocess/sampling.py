"""Window selection + augmentation (the training hot loop; numpy/OpenCV only).

Throughput rules:
  * choose the windows FIRST, then augment only the slices those windows use (training uses ~8 of 22 windows,
    i.e. ~24 of the 24 stored slices -> ~2x less warping)
  * one affine matrix and one 256-entry LUT per slot, applied with cv2 (SIMD) to all of its slices
  * no flips: knees are canonicalised to 'left' at preprocessing time, so a horizontal flip would destroy the
    medial/lateral image-side consistency that the slot head relies on
"""
import numpy as np
import cv2

cv2.setNumThreads(0)


def n_windows(depth, group, stride):
    return max(1, (depth - group) // stride + 1)


def window_starts(depth, group, stride, n_use=None, train=False, rng=None):
    """Start indices (into the stored stack) of the windows to use.

    eval : evenly spaced subset (or all)           train: one random window per equal-width bin (stratified, so
    the whole depth is covered every step) with a random phase."""
    W = n_windows(depth, group, stride)
    starts = np.arange(W) * stride
    if n_use is None or n_use >= W:
        return starts
    if train:
        rng = rng or np.random.default_rng()
        # UPGRADE 3: Z-axis jitter (through-plane translation augmentation).
        # Shifts the entire window grid by a random offset before binning.
        # Forces the model to learn anatomy rather than memorizing slice positions.
        z_jitter = int(rng.integers(0, max(1, stride // 2 + 1)))
        jittered = np.clip(starts + z_jitter, 0, depth - group)
        edges = np.linspace(0, W, n_use + 1)
        pick = [int(rng.integers(int(np.floor(edges[i])), max(int(np.floor(edges[i])) + 1, int(np.ceil(edges[i + 1]))))) for i in range(n_use)]
        return jittered[np.clip(pick, 0, W - 1)]
    return starts[np.unique(np.rint(np.linspace(0, W - 1, n_use)).astype(int))]


def gather_windows(stack, valid, starts, group):
    """stack [D,H,W] (may be a memmap view) -> windows [n,group,H,W] uint8 and window validity [n].

    A window is valid only if all of its channels are valid slices.

    CONTIGUOUS MEMMAP READ OPTIMIZATION:
    Instead of non-contiguous indexing stack[idx] that triggers scattered disk page faults
    across the 227 GB memmap file, read the bounding slice stack[lo:hi] as a single contiguous
    block into RAM, then extract windows locally at memory bus speeds (>100 GB/s).
    """
    if len(starts) == 0:
        return np.zeros((0, group, stack.shape[1], stack.shape[2]), dtype=stack.dtype), np.zeros(0, dtype=bool)
    idx = np.asarray(starts)[:, None] + np.arange(group)[None, :]
    lo = int(idx.min())
    hi = int(idx.max()) + 1
    block = np.ascontiguousarray(stack[lo:hi])
    win = block[idx - lo]
    wv = np.asarray(valid)[idx].all(axis=1)
    return win, wv


def _lut(rng, a):
    c = 1.0 + rng.uniform(-a, a)
    b = rng.uniform(-a / 2.0, a / 2.0)
    g = 1.0 + rng.uniform(-a, a)
    x = np.arange(256, dtype=np.float32) / 255.0
    y = np.clip((np.power(x, g) - 0.5) * c + 0.5 + b, 0.0, 1.0)
    return np.rint(y * 255.0).astype(np.uint8)


def augment_slot(win, rng, rot_deg=8.0, scale=0.08, shift=0.05, intensity=0.1):
    """win [n,g,H,W] uint8 -> same shape. One random rotation/scale/shift + one intensity LUT for the whole slot."""
    n, g, H, W = win.shape
    ang = rng.uniform(-rot_deg, rot_deg)
    sc = 1.0 + rng.uniform(-scale, scale)
    M = cv2.getRotationMatrix2D(((W - 1) / 2.0, (H - 1) / 2.0), ang, sc)
    M[0, 2] += rng.uniform(-shift, shift) * W
    M[1, 2] += rng.uniform(-shift, shift) * H
    lut = _lut(rng, intensity) if intensity > 0 else None
    out = np.empty_like(win)
    for i in range(n):
        for j in range(g):
            x = cv2.warpAffine(win[i, j], M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            out[i, j] = cv2.LUT(x, lut) if lut is not None else x
    return out
