# RSNA Knee MRI Dataset: Comprehensive Data Analysis Report

This report summarizes the foundational Exploratory Data Analysis (EDA) conducted on the RSNA Knee Abnormality Detection dataset. The findings here have directly influenced the robust data preprocessing pipeline implemented in `src/preprocess`.

## 1. Data Integrity and Geometry
- **Windowing Variations:** We observed that the window center varies by more than 10% inside a series for **51.89%** of the series (12,646 out of 24,371 series). This indicates that relying on raw DICOM windowing metadata can be highly unstable across slices. 
- **Field of View (FOV):** The median FOV across the dataset is **160.0 mm**, with the 1st percentile at 130.0 mm and the 99th percentile at 205.0 mm. 
- **Resolution & Aspect Ratios:** Around **7.26%** of the images are non-square, and the physical aspect ratio varies from 0.84 to 2.0 (1st to 99th percentiles). Physical millimeter cropping (as implemented in our `v2` pipeline config) is essential to standardize the anatomical scale before resizing.

## 2. Pixel Intensity and Normalisation
- **Dynamic Range:** The per-series intensity scale differs dramatically, by roughly **22.5x** across the dataset (measured between the 5th and 95th percentiles of p99 values). Global normalization is impossible; the pipeline must normalize per-series using robust percentiles (e.g., p1 to p99).
- **Background & Ringing:** **72.4%** of the series have a non-zero background level. Moreover, negative pixel values exist due to signed data or filter ringing. The pipeline handles this by padding with the border-median value rather than a literal zero, and by clipping at zero after percentile scaling.

## 3. Metadata Reliability and Flags
- **Conflated Flags:** In the training dataset, the `Fluid_Sensitive` and `Fat_Suppression` metadata flags are **100% identical**. They effectively represent a single feature. We must avoid hard-wiring a dependency on both being independent, as they may diverge in the hidden test set.
- **Routing Inconsistencies:** 26.1% of PD/T2/STIR-like series are flagged as non-fluid. The fluid-sensitive flag seems to track fat-suppression rather than contrast weighting.
- **Metadata-Poor Cohort:** A specific cohort of **238 studies (5.4%)** comes from scanners that strip critical metadata (TR, TE, magnetic field strength, and ScanOptions). For robust inference, routing logic must utilize the provided CSV flags and raw image content rather than relying solely on DICOM headers.

## 4. Cross-Study Duplicates
- **Duplicate Hashes:** Hashing central slices revealed **57 identical-image groups** that span across different studies (affecting 246 series). These represent likely duplicate exams and must be strictly grouped within the same fold during cross-validation (CV) to prevent data leakage.

## 5. NLP Report Probing
- A lexical analysis of the original radiologist reports across multiple languages (en, es, tr, de, etc.) shows that conditions like "Meniscus" and "Effusion" are mentioned in over 90% of reports, *even when the label is negative*. This indicates that negation (e.g., "no effusion") is highly prevalent, and simple keyword matching is insufficient for extracting accurate pseudo-labels without deep NLP context.

---
*Note: All logic to counteract these data irregularities (e.g., robust intensity clipping, `border_median` padding, physical `crop_mm`, and position-based slice ordering) has been integrated into `src/config.py` (Preset `v2`) and the `src/preprocess` module.*
