# RSNA Knee Abnormality Detection — Comprehensive Dataset Audit & Engineering Specification

## Document Overview
This document synthesizes the empirical findings from an exhaustive, definitive census of the RSNA Knee Abnormality Detection challenge dataset (`rsna_knee_eda.py` deep scan). Unlike sampled explorations, this audit inspected:
- **100% of all series headers** (24,371 series across `train_series`)
- **100% of individual slice headers** (819,078 DICOM files)
- **Central slice decoded pixel matrices** for all 24,371 series (evaluating radiometric distributions, perceptual hashes, foreground geometry, and background noise)
- **Cross-study consistency**, NLP report linguistics, and DICOM-to-label ground truth.

This report establishes the foundational engineering constraints, edge cases, failure modes, and preprocessing specifications required to train robust deep neural networks and achieve SOTA AUC under Kaggle's 9-hour runtime and 20 GB disk quotas.

---

## 1. Executive Summary & Core Engineering Directives

| Area | Empirical Finding | Engineering Impact / Mandatory Directive |
| :--- | :--- | :--- |
| **DICOM Integrity** | 0 unreadable files, 0 corrupted headers, 0 mixed sizes/thickness per series. Transfer Syntax is 100% `1.2.840.10008.1.2.1` (Explicit VR Little Endian). | Pure uncompressed DICOM. Fast direct binary parsing or standard `pydicom` works uniformly without decompression codec overhead. |
| **Window Center Fluctuations** | **51.89% of series** (12,646 series) exhibit `WindowCenter` varying by >10% across slices within the same series. | **DO NOT use DICOM header `WindowCenter` or `WindowWidth` tags.** Normalization must be computed directly from slice/volume pixel percentiles. |
| **Radiometric Dynamic Range** | `p99` intensity scales vary by **22.5x across series** (5th to 95th percentile) and **23.2x across scanner models**. Extreme values range from -1871 (Canon/GE signed) to +141,711 (Philips). | **DO NOT use global normalization** (e.g. dividing by 4095 or dataset-wide mean/std). Apply **per-series robust percentile scaling** (`p1` to `p99`), clamp to `[0, 1]`, and clip negative values at 0. |
| **Outlier Bright Pixels** | 8.2% of series exhibit extreme heavy tails where `p99.9 > 2x p99`. | **DO NOT use `image.max()`** as the normalization denominator. Always normalize against `p99` or `p99.5`. |
| **Background Noise Pedestal** | **72.4% of series** have border mean >5% of `p99`. | **DO NOT pad with literal zero.** Zero padding injects high-contrast artificial rectangular boundaries that confound Vision Transformers and CNNs. Pad using edge replication or local median. |
| **Physical Geometry & Aspect** | 7.3% of series have non-square matrices (e.g., 640x1280 with aspect ratio 2.0). Median FOV is 160.0 mm. Pixel spacing is 99.61% isotropic. | Direct anamorphic resizing to square (e.g. $256 \times 256$) distorts anatomical angles and meniscus thicknesses. **Isotropic physical cropping / letterboxing** is mandatory. |
| **In-Plane LPS Orientation** | LPS row/column vectors are 100% uniform: Axial is always `LP`, Coronal is always `LI`, Sagittal is always `PI`. | **Zero flip or orientation mismatch within planes.** No canonical rotation matrix alignment is required across series. |
| **Contrast & Flags** | `Fluid_Sensitive == Fat_Suppression` in **100.00% of series** in `train_series.csv`. 26.1% of physical PD/T2/STIR series have flag=0 because they lack fat suppression. | The competition flag tracks **Fat Suppression**, not pure T2 fluid contrast. Do not treat them as independent features in training, but do not hardcode equality (test set may differ). |
| **Metadata-Poor Cohort** | **238 studies (5.4%, 1,206 series)** have completely missing TR, TE, MagneticFieldStrength, and ScanOptions (Canon Vantage, GE Optima). | **DO NOT route series using DICOM header physics tags (TR/TE).** Route series strictly using the provided CSV flags (`Anatomical_Plane`, `Fluid_Sensitive`) + image heuristics. |
| **Cross-Study Hash Candidates** | **57 exact dHash groups (246 series)** span multiple distinct `StudyInstanceUID`s, but the 64-bit central-slice hash is collision-prone: across the groups, thousands of study pairs share a hash and only one pair shares five distinct series hashes. | Treat exact dHash matches as candidates, not proof of duplicate exams. Current folds keep only pairs corroborated by at least two distinct shared series hashes together; single-hash matches are excluded. |
| **Study Recipes & Coverage** | Dominant recipe is 5 series (`AFx1 Cnx1 CFx1 Snx1 SFx1`, 40%). 3.6% miss Coronal Fluid; 5.8% miss Sagittal Fluid. 1,166 studies have $\ge 2$ fluid series in the same plane. | 6-slot architectures must implement deterministic multi-series arbitration (preferring standard 2D over 3D) and missing-slot masking. |
| **Report Label Noise** | 46 duplicate report templates cover 177 studies. Negative studies mention target terms 76%–91% of the time (e.g. "no effusion"). | Ground-truth labels are NLP-derived and subject to negation extraction noise. Models require label smoothing and robust loss formulations. |
| **Inference Budget** | Median slice decode is 12.0 ms. Test set (~1,000 studies $\times$ 96 slices) decodes in ~19.2 minutes on 4 CPU cores. | No need to pre-cache test DICOMs to disk. Dynamic in-memory decoding and tensor batching easily completes within the 9-hour limit. |

---

## 2. Definitive Integrity Census (A1 Breakdown)

The audit conducted an exhaustive validation of all 819,078 slice files across 24,371 series.

### Integrity Validation Matrix
| Check | Failed Series ($n$) | Failed Series (%) | Evaluation & Diagnostic Severity |
| :--- | :---: | :---: | :--- |
| **Fatal scan errors** | 0 | 0.000% | PASS: Every directory is accessible and well-formed. |
| **Unreadable/invalid slice files** | 0 | 0.000% | PASS: 100% valid DICOM headers and readable pixel payloads. |
| **Mixed image sizes in series** | 0 | 0.000% | PASS: Matrix dimensions (`Rows`, `Columns`) are strictly constant per series. |
| **Mixed orientation in series** | 0 | 0.000% | PASS: `ImageOrientationPatient` is identical across all slices of a series. |
| **Mixed slice thickness** | 0 | 0.000% | PASS: `SliceThickness` is uniform within each series. |
| **Mixed pixel spacing** | 0 | 0.000% | PASS: `PixelSpacing` ($[ps_r, ps_c]$) is strictly constant within each series. |
| **Mixed transfer syntax** | 0 | 0.000% | PASS: Single uniform transfer syntax per series. |
| **Duplicate SOPInstanceUID** | 0 | 0.000% | PASS: No duplicate slice files or repeated instances. |
| **Duplicate slice positions** | 0 | 0.000% | PASS: No co-planar or overlapping slice coordinates. |
| **Gap in slice spacing** | 0 | 0.000% | PASS: Consecutive slice distances along the normal vector are continuous. |
| **Irregular spacing ($CV > 0.05$)** | 0 | 0.000% | PASS: Inter-slice distance along normal has $CV \le 0.05$ across all series. |
| **Missing InstanceNumber** | 0 | 0.000% | PASS: All slices possess integer `InstanceNumber` attributes. |
| **InstanceNumber != Spatial Order** | 0 | 0.000% | PASS: Correlation between `InstanceNumber` and spatial projection is $\ge 0.99$. |
| **No usable geometry** | 0 | 0.000% | PASS: All series contain valid IOP ($6 \times 1$) and IPP ($3 \times 1$) vectors. |
| **Truncated pixel data** | 0 | 0.000% | PASS: File byte size $\ge Rows \times Columns \times \frac{BitsAllocated}{8} \times SamplesPerPixel$. |
| **RescaleSlope varies in series** | 0 | 0.000% | PASS: Linear slope is invariant per series. |
| **RescaleIntercept varies in series**| 0 | 0.000% | PASS: Linear intercept is invariant per series. |
| **Central slice failed to decode** | 0 | 0.000% | PASS: Uncompressed raw pixel buffers unpack with 100% success. |
| **Constant image ($p_{99} == p_1$)** | 0 | 0.000% | PASS: No completely blank or corrupted zero-variance slices. |
| **Saturated pixels $> 1\%$** | 2 | 0.008% | INFO: Only 2 series contain $>1\%$ saturated maximum pixels. |
| **Window Center varies $> 10\%$** | **12,646** | **51.890%** | **CRITICAL WARNING:** Dynamic DICOM windowing tags per slice. |

### Technical Analysis of Integrity
1. **Transfer Syntax Uniformity:**
   Every single series (24,371 / 24,371) uses transfer syntax `1.2.840.10008.1.2.1` (**Explicit VR Little Endian**). There are no JPEG2000, JPEG-Lossless, or RLE compressed files. DICOM reading is purely raw uncompressed byte extraction.
2. **File Overhead Ratio:**
   $$\text{Ratio} = \frac{\text{DICOM File Size (bytes)}}{\text{Rows} \times \text{Columns} \times (\text{BitsAllocated}/8)}$$
   - Mean ratio: **1.0035** ($\pm 0.0026$), Min: 1.0005, Max: 1.0313.
   - Headers represent less than 0.4% of total file size, allowing blazing-fast seek and slice decoding.
3. **The Window Center Hazard:**
   More than half of all series (51.89%) have varying window center/width values across slices. In clinical viewing software, radiologists scroll through slices where local brightness auto-adapts. If a pipeline applies DICOM Window Center/Width tags, adjacent slices within the same volume will experience jarring brightness steps, corrupting 3D convolutional features and axial/sagittal continuity.
   - **Resolution:** Discard `WindowCenter` and `WindowWidth` tags completely.

---

## 3. Physical Geometry, Coordinates & Spatial Coverage (A2 Breakdown)

MRI scanners acquire physical volumes in millimeters. The relationship between physical anatomy and pixel matrices determines how models perceive pathology.

### Field of View (FOV) & Pixel Spacing Distributions
| Metric | Mean | Std | Min | 1% | 50% (Median) | 95% | 99% | Max |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FOV Height ($mm$)** | 163.7 | 15.7 | 70.3 | 130.0 | **160.0** | 189.3 | 205.0 | 320.0 |
| **FOV Width ($mm$)** | 164.7 | 22.6 | 70.3 | 130.0 | **160.0** | 190.0 | 320.0 | 361.0 |
| **In-plane Spacing ($mm$)**| 0.35 | 0.09 | 0.11 | 0.22 | **0.31** | 0.54 | 0.65 | 1.15 |
| **Aspect Ratio ($W/H$)** | 1.007 | 0.107 | 0.500 | 0.844 | **1.000** | 1.000 | **2.000** | 2.000 |
| **Spacing/Thickness Ratio**| 1.16 | 0.15 | 0.40 | 0.50 | **1.12** | 1.37 | 1.68 | 3.03 |

```
FOV Height Distribution:
  [70mm] -------- [130mm = 1%] ================= [160mm = 50%] ============ [205mm = 99%] --- [320mm]
  Core anatomical knee cluster: 140mm – 180mm.
```

### The Non-Square Matrix Problem
- **7.3% of series** have non-square matrices ($Rows \neq Columns$).
- Pixel spacing itself is isotropic ($ps_r = ps_c$) in **99.61% of series**. Thus, non-square matrices represent true non-square physical coverage.
- Top Non-Square Dimensions:
  1. $640 \times 1280$ ($W/H = 2.0$): 258 series (Double-width rectangular acquisition).
  2. $320 \times 300$: 238 series.
  3. $640 \times 540$: 232 series.
  4. $384 \times 348$: 148 series.
  5. $496 \times 490$: 105 series.
- **Architectural Danger:**
  If a $640 \times 1280$ image is naively resized to $256 \times 256$, the image is horizontally compressed by 50%. The cruciate ligaments will appear artificially steepened, menisci will appear compressed, and tears will be obscured.
- **Mandatory Solution:**
  1. Calculate physical bounding box or pad the shorter dimension with edge pixels to make the matrix square before resizing.
  2. Or, apply a fixed physical crop (e.g. $160 \times 160\text{ mm}$) centered on the joint.

### Z-Extent & Slice Counts by Anatomical Plane
| Plane | Series ($n$) | Z-Extent Mean ($mm$) | Z-Extent Median ($mm$) | 2D Slice Count (Median) | 3D Volumetric Count (Median) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Axial** | 5,898 | 119.6 | 121.4 (Range: 36–239) | 30.0 (Range: 12–55) | 144.0 (Range: 50–254) |
| **Coronal** | 8,609 | 100.4 | 101.5 (Range: 22–168) | 30.0 (Range: 11–48) | 80.0 (1 series) |
| **Sagittal** | 9,864 | 99.8 | 99.1 (Range: 21–199) | 29.0 (Range: 11–60) | 136.0 (Range: 60–320) |

- **Slice Spacing vs Thickness:**
  The median ratio is **1.12**. This confirms routine 2D clinical acquisitions with ~10% inter-slice gap (e.g., 3.0 mm thickness with 3.3 mm center-to-center spacing) to prevent RF cross-talk between adjacent slices.
- **3D Volumetric Sequences:**
  418 Axial series (7.1%) and 417 Sagittal series (4.2%) are isotropic 3D volumetric acquisitions (e.g., Siemens SPACE, GE CUBE, Philips 3D VISTA) containing 120 to 320 slices.
  - When feeding a 16-slice or 24-slice depth network, 3D volumes must be uniformly sub-sampled across their physical Z-extent rather than taking raw slice indices.

### Anatomical Directional Alignment (LPS Space)
The DICOM `ImageOrientationPatient` vector $[\mathbf{r}, \mathbf{c}]$ defines the direction cosines of image rows and columns relative to the patient's Left-Posterior-Superior (LPS) frame.
$$\mathbf{n} = \mathbf{r} \times \mathbf{c}$$

| Plane | Orientation Code | Rows Point | Columns Point | Dataset Count | Dataset Purity |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Axial** | `LP` | Left ($+X$) | Posterior ($+Y$) | 5,898 / 5,898 | **100.00%** |
| **Coronal** | `LI` | Left ($+X$) | Inferior ($-Z$) | 8,609 / 8,609 | **100.00%** |
| **Sagittal** | `PI` | Posterior ($+Y$) | Inferior ($-Z$) | 9,864 / 9,864 | **100.00%** |

- **Significance:** There are **zero reversed or flipped series** in the dataset. Every single Sagittal series views the knee along the identical directional axis. Every Coronal series displays Superior at the top and Inferior at the bottom. No arbitrary 90-degree rotations or sagittal flips exist.

---

## 4. Radiometric & Intensity Characteristics (A3 Breakdown)

### Multi-Vendor Scale Disparities
MRI raw intensities are arbitrary radiofrequency (RF) signal magnitudes without absolute physical units (unlike CT Hounsfield units). The audit revealed extreme vendor-dependent scaling:

| Vendor | Pixel Repr. | Series ($n$) | $p_{50}$ (Median) | $p_{99}$ (Median) | $p_{99}$ (5th–95th Range) | Extreme Max ($V_{max}$) | Negative Pixels? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CANON/TOSHIBA** | Signed (1.0) | 1,343 | 817.0 | **5,149.0** | 2,093 – 8,615 | 18,446.0 | **Yes (0.6% neg)** |
| **FUJI/HITACHI** | Unsigned (0.0)| 130 | 214.5 | **749.0** | 387 – 1,310 | 2,054.0 | No (0.0%) |
| **GE** | Unsigned (0.0)| 187 | 51.0 | **327.0** | 120 – 629 | 1,237.0 | No (0.0%) |
| **GE** | Signed (1.0) | 4,914 | 300.5 | **1,290.4** | 387 – 7,140 | 22,229.0 | No (0.0%) |
| **PHILIPS** | Unsigned (0.0)| 7,611 | 219.0 | **1,209.7** | 235 – 2,227 | **141,711.8** | No (0.0%) |
| **SIEMENS** | Unsigned (0.0)| 10,186 | 115.0 | **598.5** | 205 – 1,712 | 4,095.0 | No (0.0%) |

```
p99 Intensity Comparison across Scanners:
  GE (low gain):  [=== 327 ===]
  Siemens 12-bit: [====== 598 ======]
  Philips:        [============ 1,210 ============]
  Canon/Toshiba:  [================================================== 5,149 ==================================================]
  Dynamic Ratio: 22.5x spread!
```

### Statistical Observations on Signal
1. **Dynamic Scale Spread:**
   $$\frac{\text{Series } p_{99} \text{ (95th percentile)}}{\text{Series } p_{99} \text{ (5th percentile)}} = \mathbf{22.5\times}$$
   Dividing all images by a static scalar (e.g. 4095 or 1000) collapses low-gain GE/Siemens images into near-zero black arrays, while blowing out Canon images.
2. **Heavy-Tail Ratio ($p_{99.9} / p_{99}$):**
   - Mean ratio is 1.43; 95th percentile is 2.16; maximum is 4.68.
   - In **8.2% of series**, the brightest 0.1% of pixels are more than twice as bright as the 99th percentile. These represent localized fat saturation failures, RF coil pickup near surface coils, or metallic susceptibility blooming. Normalizing by `max(image)` will compress all relevant knee anatomy into a dark, low-contrast band.
3. **Negative Pixels:**
   Canon/Toshiba and certain GE reconstructions include signed integers with values down to **$-1871.0$** due to reconstruction kernel ringing and DC baseline offsets.
   - Direct conversion to `uint8` without clipping produces catastrophic byte underflow wrap-around (e.g. -1 becomes 255, creating blinding bright specks).
4. **Non-Zero Background Pedestal:**
   - Background border level relative to $p_{99}$: mean **0.0991** (~10% of tissue signal).
   - In **72.4% of series**, the border mean exceeds 5% of $p_{99}$.
   - MRI background consists of Rician / Rayleigh noise from multi-channel receiver coils. It is non-zero. If preprocessed images are padded with 0, the neural network learns spurious edge detectors at the border of the real background noise and the artificial zero padding.

### Mandatory Intensity Preprocessing Formula
For any series volume or slice $V$:
$$p_1 = \text{Percentile}(V, 1.0), \quad p_{99} = \text{Percentile}(V, 99.0)$$
$$V_{\text{norm}} = \text{Clip}\left(\frac{V - p_1}{p_{99} - p_1 + 1\times 10^{-6}}, \, 0.0, \, 1.0\right)$$
If padding is required to reach square dimensions:
$$\text{Pad value} = \text{Median of outer border pixels (or reflect edge)}$$

---

## 5. Duplicate Exams & Cross-Study Contamination (A4 Breakdown)

Perceptual differential hashing (`dhash`) of central slices is useful for candidate discovery, but is not by itself a reliable duplicate-exam detector. The 64-bit hash compresses image morphology aggressively and creates many cross-study collisions:

### Duplication Statistics
- **57 distinct exact-hash groups cover 246 series** with hashes appearing in more than one study.
- Pairwise expansion of those groups yields **3,401 candidate study pairs** sharing at least one hash; some single hashes occur in as many as 78 studies. A one-hash match must not automatically join folds.
- **Only one cross-study pair shares five distinct series hashes** in the EDA metadata. This is the high-confidence pair already recorded in `KNOWN_DUP_PAIRS`; the other pairs have only one shared hash and remain unconfirmed.
- Repeated/related series within a study are common and are not cross-study evidence.
- **Within-Study Duplicates:** 118 hash groups repeated within the same study (derived series, repeat scans, localizers).

### Cross-Validation Handling
The preprocessing pipeline builds connected components only from corroborated pairs (at least two distinct shared series hashes), then assigns each component wholly to one fold. Site/scanner stratification remains the primary competition-like validation design; an indiscriminate graph over all exact dHash matches would merge unrelated studies and is not a valid leakage-control strategy. `folds_manifest.json` records the hash-candidate count, acceptance threshold, accepted edges, and index fingerprint.

This hash evidence is not a definitive patient identity or proof of duplicated examinations. A future stronger audit should compare multiple decoded series or DICOM identity metadata before expanding the accepted-pair set.

---

## 6. Contrast Class & Acquisition Metadata Reliability (A5 & A6 Breakdown)

### The Fluid-Sensitive vs Fat-Suppressed Equivalence
The competition overview states: *"Fluid-sensitive and fat-suppression are not necessarily equivalent."*
However, empirical analysis of `train_series.csv` reveals:
$$\mathbf{Fluid\_Sensitive \equiv Fat\_Suppression \text{ in 100.00\% of series (24,371 / 24,371)}}!$$
- $14,010$ series have `Fluid_Sensitive = 1` and `Fat_Suppression = 1`.
- $10,361$ series have `Fluid_Sensitive = 0` and `Fat_Suppression = 0`.
- Zero series have discordant values in the training set.

### Physical Contrast vs Competition Flag
By analyzing physical acquisition parameters ($TR, TE, TI$):
- **IR/STIR** ($TI > 0$): 502 series $\rightarrow$ 100% flagged `Fluid_Sensitive = 1`.
- **Proton Density (PD)** ($TR \ge 1000, TE \le 45$): 2,250 flagged 0, 9,963 flagged 1.
- **T2-Weighted** ($TR \ge 1000, TE > 45$): 2,251 flagged 0, 2,254 flagged 1.
- **T1-Weighted** ($TR < 1000, TE < 40$): 5,612 flagged 0, 333 flagged 1.

**Key Insight:** **26.1% of all PD/T2/STIR-like series are flagged `Fluid_Sensitive = 0`.**
Why? Because they are non-fat-suppressed PD or non-fat-suppressed T2 acquisitions!
Therefore, the label `Fluid_Sensitive` in this competition is a misnomer: **it empirically tracks Fat Suppression, not whether the sequence has long-TE fluid contrast.**
- In train, treating `Fluid_Sensitive` and `Fat_Suppression` as separate features provides zero additional information.
- However, do not hardcode a single feature pipeline: in the hidden test set, non-fat-sat STIR or unusual sequences could decouple them.

### SeriesDescription Quality
- **17.7% of series** contain missing descriptions or placeholder text (`"DummySeriesDesc!"`).
- Pipelines that use regex on `SeriesDescription` to route series to planes or contrast slots will completely fail on nearly one-fifth of the dataset!

### The Metadata-Poor Cohort (A6 Breakdown)
- **1,206 series across 238 studies (5.4% of all studies)** have stripped DICOM headers:
  - Missing $TR$ (Repetition Time)
  - Missing $TE$ (Echo Time)
  - Missing $TI$ (Inversion Time)
  - Missing `MagneticFieldStrength`
  - Missing `ScanOptions`
- In all 238 studies, **every single series in the study is metadata-poor.**
- Offending Scanner Models:
  1. *Canon/Toshiba Vantage*: 656 series
  2. *GE Optima MR450w*: 384 series
  3. *GE SIGNA EXCITE*: 115 series
  4. *GE Signa HDxt*: 40 series
  5. *GE DISCOVERY MR750w*: 6 series
  6. *GE SIGNA HDx*: 5 series
- All 238 studies are English language reports. 6 of them contain verified Gold labels.
- Characteristics: Median slice thickness is 4.0 mm (vs 3.0 mm for rest); BitsStored is 16 (vs 12).
- **Rule:** Series selection and routing must never rely on DICOM tags $TR, TE, TI$. It must rely strictly on `train_series.csv` columns: `Anatomical_Plane` and `Fluid_Sensitive`.

---

## 7. Study Recipe Topologies & 6-Slot Routing Engine (A7 Breakdown)

### Top Study Recipes
Clinical knee MRI examinations acquire a standard set of views. Across the 4,400+ studies, the distribution of acquired series per study follows distinct recipes:

```
Distribution of Study Protocols:
  [==================== AF1 C1 CF1 S1 SF1 (40%) ====================]
  [========== AF1 CF1 S1 SF1 (11%) ==========]
  [====== AF1 C2 CF1 S1 SF1 (7%) ======]
  [==== AF1 CF1 S1 SF2 (5%) ====]
  [=== A1 AF1 CF1 S2 (5%) ===]
  [============== All Other Recipes (33%) ==============]
```

| Recipe Code | Breakdown ($A$=Axial, $C$=Coronal, $S$=Sagittal; $F$=Fluid/FS, $n$=Non-FS) | Studies ($n$) | Share (%) |
| :--- | :--- | :---: | :---: |
| **`AFx1 Cnx1 CFx1 Snx1 SFx1`** | 1 Axial FS, 1 Coronal Non-FS, 1 Coronal FS, 1 Sagittal Non-FS, 1 Sagittal FS | 1,747 | 39.7% |
| **`AFx1 CFx1 Snx1 SFx1`** | 1 Axial FS, 1 Coronal FS, 1 Sagittal Non-FS, 1 Sagittal FS | 475 | 10.8% |
| **`AFx1 Cnx2 CFx1 Snx1 SFx1`** | 1 Axial FS, 2 Coronal Non-FS, 1 Coronal FS, 1 Sagittal Non-FS, 1 Sagittal FS | 290 | 6.6% |
| **`AFx1 CFx1 Snx1 SFx2`** | 1 Axial FS, 1 Coronal FS, 1 Sagittal Non-FS, 2 Sagittal FS | 228 | 5.2% |
| **`Anx1 AFx1 CFx1 Snx2`** | 1 Axial Non-FS, 1 Axial FS, 1 Coronal FS, 2 Sagittal Non-FS | 203 | 4.6% |
| **`Anx2 AFx1 Cnx1 CFx1 Snx1 SFx1`** | 2 Axial Non-FS, 1 Axial FS, 1 Coronal Non-FS, 1 Coronal FS, 1 Sag Non, 1 Sag FS| 166 | 3.8% |

### Slot Availability & Missingness Rates
For a multi-view model designed around the standard **6 Slots**:
1. `Axial_Fluid`
2. `Axial_NonFluid`
3. `Coronal_Fluid`
4. `Coronal_NonFluid`
5. `Sagittal_Fluid`
6. `Sagittal_NonFluid`

| Plane | Studies with 0 Fluid Series | Studies with 1 Fluid Series | Studies with $\ge 2$ Fluid Series |
| :--- | :---: | :---: | :---: |
| **Axial** | **0.0%** | 93.2% | 6.8% |
| **Coronal** | **3.6%** | 88.3% | 8.1% |
| **Sagittal** | **5.8%** | 82.6% | 11.6% |

- **Missing Slot Fallback:**
  - 5.8% of patients have no Sagittal Fluid series.
  - 3.6% of patients have no Coronal Fluid series.
  - Axial Non-Fluid is missing in ~48% of studies (as seen in recipe `AFx1 CFx1 Snx1 SFx1`).
  - The model architecture must use **Slot Missingness Embeddings** or **Zero-Tensor Masking** with self-attention so the network gracefully ignores empty views.

### Multi-Series Competition Resolution
In 1,166 (study, plane) instances, a study contains 2 or more series competing for the same slot (e.g. two Sagittal Fluid series).
- Why? Usually one standard 2D FSE series and one 3D high-resolution or thin-slice volume.
- Across these competing pairs:
  - $TE$ differs by $>10\text{ ms}$ in 44% of cases.
  - Slice thickness differs by $>0.5\text{ mm}$ in 23% of cases.
  - Slice count differs by $>5$ in 34% of cases.
- **Deterministic Selection Priority for Slot Assignment:**
  1. Filter to candidates matching the exact plane and target `Fluid_Sensitive` flag.
  2. Reject 3D sequences ($n\_files > 80$) if a standard 2D sequence ($20 \le n\_files \le 50$) exists.
  3. Prefer series with highest in-plane resolution (smallest `ps_row`).
  4. If tied, pick the series with lowest $TE$ difference from clinical standard (30–50 ms).

---

## 8. Multi-Lingual Radiology Reports & Label Noise (A8 & A9 Breakdown)

### Multi-Site & Multi-Lingual Demographics
The dataset was aggregated across multiple international medical centers, yielding 10 distinct language cohorts and 21 unique site proxies (`language | vendor`):

| Language Cohort | Studies ($n$) | Primary Scanner Manufacturers | Distinct Scanner Models |
| :--- | :---: | :--- | :---: |
| **English (`en`)** | 1,717 | GE (469), Philips (589), Siemens (417), Canon (218), Fuji (24) | 36 models |
| **Spanish (`es`)** | 657 | Philips (310), Siemens (210), GE (137) | 9 models |
| **Turkish (`tr`)** | 568 | GE (299), Siemens (224), Philips (37), Canon (8) | 10 models |
| **Croatian/Bosnian/Serbian (`hr/bs/sr`)**| 406 | Siemens (286), Philips (120) | 4 models |
| **Greek (`greek`)** | 321 | Siemens (321) | 2 models |
| **German (`de`)** | 261 | Siemens (261) | 1 model |
| **Cyrillic (`ru/bg`)**| 220 | Philips (220) | 1 model |
| **Dutch (`nl`)** | 151 | Siemens (151) | 2 models |
| **French (`fr`)** | 78 | Siemens (78) | 1 model |
| **Unknown / Blank** | 28 | Philips (25), Siemens (3) | 6 models |

- **58 Gold-Labeled Studies:**
  The competition dataset includes a gold validation subset of 58 studies labeled by expert consensus. Their linguistic distribution spans English (28), Spanish (10), Turkish (6), Croatian (4), Cyrillic (3), Greek (3), Dutch (2), and German (2). This proves that test evaluation encompasses multi-site data.

### Duplicate Report Templates
46 distinct report texts appear multiple times, covering 177 studies:
- Turkish: 37 studies share the identical phrase: *"Diz eklemi içi sıvı miktarı normal. Çapraz ve yan bağlar normal..."* (All structures normal).
- Spanish: 14 studies share *"Técnica: RMN de la rodilla... Pinzamiento de la almohadilla grasa de Hoffa"*.
- Spanish: 12 studies share *"Técnica: RMN de la rodilla... Sin anomalías"*.

### Lexical Probe vs Gold Truth: The Negation Dilemma
Lexical regex probes were tested against the 58 Gold-annotated studies to measure mention frequency:

| Concept | Gold Positive ($n$) | Concept Mentioned when Positive ($P(M \mid +)$) | Concept Mentioned when Negative ($P(M \mid -)$) |
| :--- | :---: | :---: | :---: |
| **Meniscus** | 37 | **1.00** (100%) | **0.90** (90%) |
| **ACL** | 24 | **0.96** (96%) | **0.76** (76%) |
| **Effusion** | 35 | **0.86** (86%) | **0.91** (91%) |
| **Contusion / Edema** | 19 | **0.79** (79%) | **0.69** (69%) |
| **Collateral Ligament (MCL)** | 9 | **0.78** (78%) | **0.61** (61%) |
| **Baker's Cyst** | 12 | **0.83** (83%) | **0.39** (39%) |
| **Fracture** | 18 | **0.44** (44%) | **0.25** (25%) |

```
Mention Rate in Normal / Negative Cases:
  Effusion: [============================================= 91% =============================================]
  Meniscus: [========================================== 90% ==========================================]
  ACL:      [==================================== 76% ====================================]
```

### Critical Takeaway on Labels
Because radiologists systematically dictate standard negatives (e.g. *"Effusion: No joint effusion seen"*, *"ACL: Intact and normal in course"*), NLP-derived labels in `train.csv` are inherently noisy:
1. Inverting a complex negation across 10 languages leads to false positive labels in `train.csv`.
2. Weakly supervised NLP extraction introduces label ambiguity.
3. **Modeling Imperative:** Use **BCEWithLogitsLoss with Label Smoothing** ($\alpha = 0.05$) or **Focal Loss** to prevent the vision network from overfitting to corrupted NLP labels. Evaluate model checkpoints against the 58 Gold studies.

---

## 9. Computational & Storage Budget Specification (A11 Breakdown)

### In-Plane Resolution vs Spatial Fidelity
- Median physical FOV: **160 mm**.
- Native median pixel spacing: **0.31 mm/px**.
- At different target square resolutions:
  - **192 px**: $160 / 192 = \mathbf{0.83\text{ mm/px}}$ (Loses $2.7\times$ resolution vs native).
  - **256 px**: $160 / 256 = \mathbf{0.63\text{ mm/px}}$ (Loses $2.0\times$ resolution vs native; strong balance).
  - **320 px**: $160 / 320 = \mathbf{0.50\text{ mm/px}}$ (Preserves fine meniscal tears and ligament fibers).

### Offline Disk Storage Feasibility (Kaggle Limit: 20 GB)
Total `uint8` storage requirements for the entire training set (4,400+ studies):

| Selection Mode | Slices per Series ($D$) | Resolution 192 | Resolution 256 | Resolution 320 |
| :--- | :---: | :---: | :---: | :---: |
| **1 series/plane (3 series/study)** | 16 | 7.8 GB | **13.9 GB** | 21.7 GB (Over) |
| | 24 | 11.7 GB | 20.8 GB (Over) | 32.5 GB (Over) |
| | 32 | 15.6 GB | 27.7 GB (Over) | 43.3 GB (Over) |
| **All Fluid-Sensitive Series** | 16 | 8.3 GB | **14.7 GB** | 23.0 GB (Over) |
| | 24 | 12.4 GB | 22.0 GB (Over) | 34.4 GB (Over) |
| **All Series (Full 24k series)** | 16 | 14.4 GB | 25.6 GB (Over) | 39.9 GB (Over) |

- **Caching Guideline:**
  If caching preprocessed arrays to Kaggle disk `/kaggle/working`:
  - A 3-series setup at **$16 \times 256 \times 256$** fits safely (**13.9 GB**).
  - Storing all 6 slots at 256x256 exceeds 20 GB and triggers `DiskQuotaExceeded`.

### Online Inference Feasibility (Kaggle 9-Hour Limit)
- Measured DICOM decode speed: **12.0 ms per slice** on standard Kaggle CPU cores.
- For a test set of **1,000 hidden studies**:
  - Selecting 6 slots $\times$ 16 slices = **96 slices per study**.
  - Total slices to decode: $1,000 \times 96 = 96,000$ slices.
  - Total decode time on 4 CPU workers:
    $$\text{Time} = \frac{96,000 \times 0.012\text{ s}}{4} = \mathbf{288\text{ seconds (4.8 minutes)}}!$$
  - GPU forward pass on 1,000 studies (batch size 8, DINOv2 / ConvNeXt): $\approx \mathbf{12\text{ minutes}}$.
- **Conclusion:** There is zero need to cache test images to disk. Dynamic multi-threaded DICOM decoding directly during inference consumes less than **20 minutes total**, well within the 540-minute (9-hour) timeout.

---

## 10. Complete Preprocessing & Model Architecture Directives

Based on the definitive audit, all pipeline components must adhere to the following specifications:

```
[Raw DICOM Files]
       │
       ▼
1. SPATIAL ORDERING
   Projection = IPP · (r × c)
   Sort slices by Projection ascending
       │
       ▼
2. 6-SLOT ROUTING
   Route by Anatomical_Plane + Fluid_Sensitive
   Arbitrate competing series (prefer 2D, high-res)
   Mask missing slots
       │
       ▼
3. UNIFORM DEPTH SAMPLING
   Sample N slices (e.g. 16) evenly across volume
       │
       ▼
4. RADIOMETRIC NORMALIZATION
   Compute p1 and p99 on volume
   V_norm = clip((V - p1) / (p99 - p1), 0, 1)
   Pad non-square dimensions with edge reflection
       │
       ▼
5. RESIZE & BATCHING
   Bicubic interpolate to 256x256
   Stack into [B, 6, 16, 256, 256]
       │
       ▼
[Multi-View Vision Backbone]
```

### Preprocessing Code Specifications

#### A. Spatial Slice Ordering
```python
import numpy as np

def get_slice_order(slice_datasets):
    """
    Orders slices along the physical normal vector.
    slice_datasets: list of pydicom dataset objects
    """
    iop = [float(x) for x in slice_datasets[0].ImageOrientationPatient]
    normal = np.cross(iop[:3], iop[3:])
    normal = normal / (np.linalg.norm(normal) + 1e-9)
    
    positions = []
    for ds in slice_datasets:
        ipp = np.array([float(x) for x in ds.ImagePositionPatient])
        proj = np.dot(ipp, normal)
        positions.append(proj)
        
    order = np.argsort(positions)
    return [slice_datasets[i] for i in order]
```

#### B. Robust Radiometric Normalization & Padding
```python
def normalize_and_pad(volume, target_size=(256, 256)):
    """
    volume: 3D float32 array [D, H, W]
    """
    # 1. Percentile scaling (avoiding max outliers and signed negatives)
    p1, p99 = np.percentile(volume, (1.0, 99.0))
    denom = max(p99 - p1, 1e-6)
    v_norm = np.clip((volume - p1) / denom, 0.0, 1.0)
    
    # 2. Aspect-preserving pad to square
    D, H, W = v_norm.shape
    if H != W:
        max_dim = max(H, W)
        pad_h = (max_dim - H) // 2
        pad_w = (max_dim - W) // 2
        # Edge padding to avoid artificial 0-contrast boundaries
        v_norm = np.pad(
            v_norm, 
            ((0, 0), (pad_h, max_dim - H - pad_h), (pad_w, max_dim - W - pad_w)),
            mode='edge'
        )
    return v_norm
```

#### C. Validation Leakage Prevention
```python
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix

def build_leakage_free_groups(study_df, duplicate_pairs):
    """
    study_df: DataFrame with StudyInstanceUID
    duplicate_pairs: list of tuples (study_a, study_b) sharing central dhash
    """
    study_map = {uid: idx for idx, uid in enumerate(study_df['StudyInstanceUID'].unique())}
    n = len(study_map)
    row, col = [], []
    for s1, s2 in duplicate_pairs:
        if s1 in study_map and s2 in study_map:
            row.extend([study_map[s1], study_map[s2]])
            col.extend([study_map[s2], study_map[s1]])
            
    adj = csr_matrix((np.ones(len(row)), (row, col)), shape=(n, n))
    n_components, labels = connected_components(adj, directed=False)
    
    group_mapping = {uid: labels[idx] for uid, idx in study_map.items()}
    return study_df['StudyInstanceUID'].map(group_mapping)
```
