"""torch Dataset over the memmap cache. All real work happens in preprocess.loader (numpy)."""
import json
import numpy as np
import torch
from torch.utils.data import Dataset

import src.core.config as config
from src.data.preprocess import cache as cache_mod
from src.data.preprocess import loader


class RSNADataset(Dataset):
    """df: StudyInstanceUID + target columns (NaN = unlabeled) [+ '<target>_weight' columns].

    Returns (imgs uint8 [S,W,3,H,W], slot_mask [S], win_mask [S,W], targets [12], weights [12]).
    Throughput notes: the memmap is opened lazily per worker (fork-safe); labels are dense arrays; the augmentation
    RNG is seeded from (seed, epoch, index) so runs are reproducible regardless of worker scheduling."""

    def __init__(self, df, cache_prefix, cfg=None, is_train=False, n_windows_use=None, seed=config.SEED, aug=True):
        self.prefix, self.cfg = cache_prefix, cfg or cache_mod.cfg_of(cache_prefix)   # default: the cache's own config
        self.is_train, self.n_use, self.seed, self.aug, self.epoch = is_train, n_windows_use, seed, 0, 0
        with open(f'{cache_prefix}.meta.json') as fh:
            studies = json.load(fh)['studies']
        available = set(studies)
        aligned = df[df['StudyInstanceUID'].astype(str).isin(available)].reset_index(drop=True)
        self.ids = aligned['StudyInstanceUID'].astype(str).tolist()
        self.rows, self.Y, self.Wt = loader.label_arrays(aligned, {s: i for i, s in enumerate(studies)})
        self._cache = None

    def set_epoch(self, e):
        self.epoch = int(e)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, k):
        if self._cache is None:
            self._cache = cache_mod.StudyCache(self.prefix, 'r')
        rng = np.random.default_rng([self.seed, self.epoch, int(k)])
        imgs, slot, wm = loader.make_sample(self._cache, int(self.rows[k]), self.cfg, self.is_train, self.n_use, rng, self.aug)
        return (torch.from_numpy(imgs), torch.from_numpy(slot).float(), torch.from_numpy(wm).float(),
                torch.from_numpy(self.Y[k]), torch.from_numpy(self.Wt[k]))
