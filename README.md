# RSNA Knee MRI Abnormality Detection

This repository contains a state-of-the-art (SOTA) baseline solution for the **RSNA Knee Abnormality Detection** Kaggle competition. The architecture and preprocessing pipelines have been heavily optimized based on the top-scoring public solutions (Public LB ~0.941), refactored into a clean, modular Python codebase.

## Competition Overview
The goal of this competition is to automate the detection and classification of various abnormalities in Knee MRI scans. 

**Target Abnormalities (12 Classes):**
- **Ligaments/Menisci:** Anterior Cruciate Ligament (ACL) Tear, Medial Collateral Ligament (MCL) Tear, Medial Meniscus Tear, Lateral Meniscus Tear
- **Osteoarthritis (OA):** Medial Compartment OA, Lateral Compartment OA, Patellofemoral (PF) Joint OA
- **Other Findings:** Joint Effusion, Synovitis, Baker's Cyst, Bone Contusion, Fracture

## Evaluation Metric
Submissions are evaluated using the **Macro Area Under the Receiver Operating Characteristic Curve (Macro AUC)** across all 12 target classes. 

## Solution Architecture

The current baseline implements a highly customized vision transformer architecture tailored for 3D MRI volumes:

1. **Backbone**: dinov2-small initialized with pretrained self-supervised weights. Only the last 6 transformer blocks and the final LayerNorm are unfrozen for fine-tuning.
2. **Physical Millimeter Cropping**: To account for extreme variations in MRI scanner resolutions and fields-of-view, DICOM pixel spacing metadata is used to rigidly crop volumes to a **130.0 mm** physical footprint prior to resizing to 336x336.
3. **Slot Attention Head (SlotHead)**: Instead of flattening 3D scans, the model extracts features from 6 standardized acquisition planes (Sagittal T1, Coronal Fluid, Axial Fluid, etc.) and routes them using a specialized einsum-based attention pooling layer.
4. **Optimization**: Dual learning rates using AdamW (8e-6 for the backbone, 1e-3 for the head) managed by a OneCycleLR scheduler over 10 epochs. 

## Repository Structure

`	ext
├── src/
│   ├── config.py       # Global constants, hyperparams, and slot definitions
│   ├── dataset.py      # DICOM loading, physical mm cropping, and laterality flipping
│   ├── model.py        # DINOv2 instantiation and SlotHead architecture
│   └── train.py        # Custom weighted BCE training loop and AMP scaling
├── docs/
│   ├── data_analysis_report.md  # Comprehensive EDA findings (Intensity, Geometry, Flags)
│   ├── architecture.md          # Detailed pipeline & model architecture design
│   └── deep_report.md           # Original EDA report from Kaggle
├── requirements.txt    # Project dependencies
└── README.md
`

## Documentation & Data Analysis
For a detailed review of the imaging constraints, data integrity issues, and dataset preprocessing rules, see:
- [Data Analysis Report](docs/data_analysis_report.md)
- [Architecture & Pipeline Design](docs/architecture.md)

## Getting Started

1. **Install dependencies:**
   `ash
   pip install -r requirements.txt
   `
2. **Configure Data Paths:** 
   Update dataset directory paths in src/config.py as needed for your local environment.
3. **Train the Model:**
   `ash
   python src/train.py
   `
