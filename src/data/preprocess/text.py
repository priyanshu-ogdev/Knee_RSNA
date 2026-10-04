"""Tiny text/vendor helpers used to build leakage-safe folds (no external NLP dependency)."""
import re

_STOP = {
    'en': 'the and of with is are there no without in to an at from which seen noted',
    'es': 'el la los las de del con sin se en y una un que es hay no observa',
    'pt': 'o os as de da do com sem em e uma um que não nao há ha',
    'fr': 'le la les des du de et avec sans un une est pas il y a en dans',
    'it': 'il lo gli le di del della con senza non è una un e nel nella si',
    'de': 'der die das und mit ohne kein keine ist nicht eine im den des zeigt',
    'tr': 've ile bir olarak yok izlendi görülmektedir normal de da bulunmaktadır',
    'hr/bs/sr': 'i u je na se bez s nema nije vidljiv vidljiva uredan uredna',
    'nl': 'de het een en met zonder geen is niet van in op te',
}
_STOP = {k: set(v.split()) for k, v in _STOP.items()}


def guess_lang(text):
    """Coarse report-language proxy (script ranges + stop-word scores). Only used as a SITE proxy for fold grouping,
    never as a feature. Known limitation: templated English reports can be tagged 'tr'; paired with scanner model
    in the group key, this does not matter."""
    if not isinstance(text, str) or len(text.strip()) < 5:
        return 'empty'
    t = text.lower()
    letters = [ch for ch in t if ch.isalpha()]
    if not letters:
        return 'empty'

    def frac(a, b):
        return sum(a <= ch <= b for ch in letters) / len(letters)
    if frac('\u0400', '\u04ff') > .3:
        return 'cyrillic'
    if frac('\u0370', '\u03ff') > .3:
        return 'greek'
    if frac('\u0e00', '\u0e7f') > .3:
        return 'thai'
    if frac('\u4e00', '\u9fff') > .2:
        return 'cjk'
    if frac('\u0600', '\u06ff') > .3:
        return 'arabic'
    toks = re.findall(r'[^\W\d_]+', t)
    sc = {k: sum(w in v for w in toks) for k, v in _STOP.items()}
    sc['tr'] += 2 * sum(ch in 'ğışİ' for ch in text) / max(len(toks), 1) * 10
    best = max(sc, key=sc.get)
    return best if sc[best] >= 3 else 'unknown'


def vendor_of(manufacturer):
    """Normalise the many spellings seen in the data (Siemens/SIEMENS/Siemens Healthineers, GE/GEHC, ...)."""
    m = str(manufacturer).lower()
    for k, v in (('siemens', 'SIEMENS'), ('ge medical', 'GE'), ('gehc', 'GE'), ('philips', 'PHILIPS'),
                 ('toshiba', 'CANON/TOSHIBA'), ('canon', 'CANON/TOSHIBA'), ('fuji', 'FUJI/HITACHI'),
                 ('hitachi', 'FUJI/HITACHI')):
        if k in m:
            return v
    return 'OTHER'
