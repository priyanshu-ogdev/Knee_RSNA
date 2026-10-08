import pytest
import numpy as np
import os
import glob
from src.data.preprocess import pixels
from src.core import config


def test_fg_center_empty_or_flat():
    # Empty raws
    assert pixels._fg_center([]) is None
    
    # Flat image (no dynamic range / span <= 0)
    flat = [np.ones((64, 64), dtype=np.uint16) * 100]
    assert pixels._fg_center(flat) is None


def test_fg_center_decentered_across_pedestals():
    """Verify that decentered foreground (col 900 in 640x1280 frame) stays accurately
    centered across background pedestals from 0% up to 20% of dynamic range, rather
    than collapsing to frame center (638-640)."""
    H, W = 640, 1280
    true_col = 900.0
    true_row = 320.0
    
    for ped_pct in [0.0, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]:
        img = np.zeros((H, W), dtype=np.float32)
        # Knee foreground block
        img[170:470, 750:1050] = 800.0
        # Background pedestal
        img += (800.0 * ped_pct)
        # Gaussian noise in background
        noise = np.random.normal(0, 10.0, (H, W)).astype(np.float32)
        img = np.maximum(img + noise, 0)
        
        c = pixels._fg_center([img])
        assert c is not None
        detected_row, detected_col = c
        
        # Must be close to true knee center (900.0), NOT frame center (639.5)
        assert abs(detected_col - true_col) < 25.0, (
            f"Pedestal {ped_pct*100}% collapsed to {detected_col:.1f} (expected ~{true_col})"
        )
        assert abs(detected_row - true_row) < 25.0


def test_fg_center_rejects_peripheral_noise_specks():
    """Verify that isolated bright noise specks on the boundary do not corrupt the bounding box."""
    H, W = 640, 1280
    true_col = 900.0
    
    img = np.zeros((H, W), dtype=np.float32)
    img[170:470, 750:1050] = 800.0
    img += 80.0  # 10% pedestal
    
    # Add bright artifact specks right on the corners/edges
    img[0:4, 0:4] = 800.0
    img[0:4, -4:] = 800.0
    img[-4:, 0:4] = 800.0
    img[-4:, -4:] = 800.0
    
    c = pixels._fg_center([img])
    assert c is not None
    assert abs(c[1] - true_col) < 25.0, f"Edge specks corrupted center to {c[1]:.1f}"


def test_crop_plan_wide_frame_real_dicom_if_available():
    """Verify on real 640x1280 wide DICOM series if available locally."""
    test_dcm_pattern = "data/test_series/*/*/*.dcm"
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    files = glob.glob(os.path.join(project_root, test_dcm_pattern))
    
    wide_files = []
    if files:
        import pydicom
        for f in files:
            d = pydicom.dcmread(f, stop_before_pixels=True)
            if d.Rows == 640 and d.Columns == 1280:
                wide_files.append(f)
                if len(wide_files) >= 5:
                    break
                    
    if wide_files:
        from src.data.preprocess import dicomio as dio
        raws = [dio.decode_raw(f)[0] for f in wide_files]
        c = pixels._fg_center(raws)
        assert c is not None
        # Verify valid coordinates within frame bounds
        assert 100.0 < c[0] < 540.0
        assert 300.0 < c[1] < 980.0
        
        cfg = config.get_cfg('v2')
        plan = pixels.crop_plan((640, 1280), 0.25, 0.25, cfg, raws)
        assert plan is not None
        y0, x0, Hc, Wc = plan
        assert Hc == 520 and Wc == 520
        # Re-centering must shift x0 away from naive border
        center_x = x0 + (Wc - 1) / 2.0
        assert abs(center_x - c[1]) < 2.0
