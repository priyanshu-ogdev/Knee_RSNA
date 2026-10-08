import re

path = r'src\data\preprocess\dicomio.py'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

target = '''    if str(getattr(ds, 'PhotometricInterpretation', '')).upper() == 'MONOCHROME1':
        a = -a
    slope = getattr(ds, 'RescaleSlope', 1.0)
    icpt = getattr(ds, 'RescaleIntercept', 0.0)
    return a, float(slope) if slope is not None else 1.0, float(icpt) if icpt is not None else 0.0'''

replacement = '''    slope = getattr(ds, 'RescaleSlope', 1.0)
    icpt = getattr(ds, 'RescaleIntercept', 0.0)
    slope = float(slope) if slope is not None else 1.0
    icpt = float(icpt) if icpt is not None else 0.0
    if str(getattr(ds, 'PhotometricInterpretation', '')).upper() == 'MONOCHROME1':
        slope = -slope
        icpt = -icpt
    return a, slope, icpt'''

text = text.replace(target, replacement)

with open(path, 'w', encoding='utf-8') as f:
    f.write(text)
print("Fixed dicomio.py MONOCHROME1 fallback")
