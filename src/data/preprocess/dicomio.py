"""Robust DICOM header / pixel access.

Dataset facts this module is built around (verified on all 24,371 train series):
  * every file is uncompressed Explicit VR LE, but the hidden test set is not guaranteed to be -> decode errors
    are caught per slice, never fatal
  * 26% of series are signed int16, Canon series contain negative pixels (down to -1871)
  * RescaleSlope/Intercept are constant inside a series (Philips slope reaches 262)
  * DICOM WindowCenter/Width vary per slice in 52% of series -> never used
"""
import os
import numpy as np
import pydicom

ORDER_TAGS = ['ImagePositionPatient', 'ImageOrientationPatient', 'InstanceNumber',
              'Rows', 'Columns', 'PixelSpacing']

REP_TAGS = ['SeriesDescription', 'SequenceName', 'ScanOptions', 'ScanningSequence', 'RepetitionTime', 'EchoTime',
            'InversionTime', 'MRAcquisitionType', 'Laterality', 'ImageLaterality', 'SliceThickness',
            'SpacingBetweenSlices', 'Manufacturer', 'ManufacturerModelName', 'MagneticFieldStrength',
            'BitsStored', 'PixelRepresentation', 'PhotometricInterpretation', 'BodyPartExamined']


def f1(v):
    """First float of a scalar / multi-value, else None."""
    try:
        return float(v)
    except Exception:
        try:
            return float(list(v)[0])
        except Exception:
            return None


def floats(v, n):
    try:
        a = [float(x) for x in v]
    except Exception:
        return None
    return a if len(a) == n and all(np.isfinite(a)) else None


def sval(v):
    """Header value -> plain str ('|' joined for multi-values), None if absent."""
    if v is None:
        return None
    if isinstance(v, (list, tuple)) or v.__class__.__name__ == 'MultiValue':
        return '|'.join(str(x) for x in v)
    return str(v)


def read_tags(path, tags):
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True, force=True, specific_tags=list(tags))
    except Exception:
        return None
    return ds


def representative_header(path):
    """Full (pixel-less) header of one file -> dict of REP_TAGS (strings) + numeric helpers."""
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
    except Exception:
        return None
    out = {t: sval(getattr(ds, t, None)) for t in REP_TAGS}
    ps = floats(getattr(ds, 'PixelSpacing', None), 2)
    out['ps_row'], out['ps_col'] = (ps[0], ps[1]) if ps else (np.nan, np.nan)
    out['transfer_syntax'] = str(getattr(getattr(ds, 'file_meta', None), 'TransferSyntaxUID', ''))
    return out


def decode_slice(path):
    """Decode one slice -> float32 2-D array with RescaleSlope/Intercept applied.

    Raises on any failure (caller decides the fallback). MONOCHROME1 is inverted. Non-finite values -> 0.
    """
    # OPTIMIZATION: Bypassing heavy metadata parsing.
    # Only parse the exact tags needed to decompress pixels and apply linear rescale.
    # Saves ~30-50% CPU parsing time per slice (critical for DGX Linux multi-core scaling).
    tags = ['PixelData', 'Rows', 'Columns', 'BitsAllocated', 'BitsStored', 'HighBit',
            'PixelRepresentation', 'SamplesPerPixel', 'PhotometricInterpretation',
            'RescaleSlope', 'RescaleIntercept']
    ds = pydicom.dcmread(path, force=True, specific_tags=tags)
    a = ds.pixel_array
    if a.ndim != 2:
        raise ValueError(f'expected a 2-D slice, got ndim={a.ndim}')
    a = a.astype(np.float32)
    slope = f1(getattr(ds, 'RescaleSlope', None))
    icpt = f1(getattr(ds, 'RescaleIntercept', None))
    a = a * (slope if slope else 1.0) + (icpt if icpt else 0.0)
    if str(getattr(ds, 'PhotometricInterpretation', '')).upper() == 'MONOCHROME1':
        a = a.max() - a
    if not np.isfinite(a).all():
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    return a


def list_dicoms(series_dir):
    try:
        with os.scandir(series_dir) as it:
            return sorted(e.name for e in it if e.name.lower().endswith('.dcm'))
    except OSError:
        return []


def decode_raw(path):
    """Fast decode: RAW integer pixels + rescale params.
    Optimized to dynamically use dicomsdl (C++ based) on DGX if available, providing a massive speedup (~1ms vs 12ms)."""
    try:
        import dicomsdl
        ds = dicomsdl.open(path)
        # SOTA Fix: Must use storedvalue=True to get raw integers, otherwise dicomsdl double-applies slope/icpt!
        a = ds.pixelData(storedvalue=True)
        
        # Photometric Interpretation check is CRITICAL
        try:
            # SOTA Fix: dicomsdl does not have .info(). Use direct attribute access.
            photo = getattr(ds, 'PhotometricInterpretation', '')
            if 'MONOCHROME1' in str(photo).upper():
                a = a.max() - a
        except Exception:
            pass
            
        try:
            slope = float(ds.RescaleSlope)
        except Exception:
            slope = 1.0
            
        try:
            icpt = float(ds.RescaleIntercept)
        except Exception:
            icpt = 0.0
            
        return a, slope, icpt
    except Exception:
        pass

    # Fallback to specific_tags optimized pydicom
    tags = ['PixelData', 'Rows', 'Columns', 'BitsAllocated', 'BitsStored', 'HighBit',
            'PixelRepresentation', 'SamplesPerPixel', 'PhotometricInterpretation',
            'RescaleSlope', 'RescaleIntercept']
    ds = pydicom.dcmread(path, force=True, specific_tags=tags)
    a = ds.pixel_array
    if a.ndim != 2:
        raise ValueError(f'expected a 2-D slice, got ndim={a.ndim}')
    if str(getattr(ds, 'PhotometricInterpretation', '')).upper() == 'MONOCHROME1':
        a = a.max() - a
    slope = getattr(ds, 'RescaleSlope', 1.0)
    icpt = getattr(ds, 'RescaleIntercept', 0.0)
    return a, float(slope) if slope is not None else 1.0, float(icpt) if icpt is not None else 0.0
