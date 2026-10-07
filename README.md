# RSNA 2026: End-to-End Installation & Execution Guide

This guide covers the deployment of the RSNA Knee Abnormality Detection pipeline on a DGX/Linux server for mass parallelized extraction and training, followed by Kaggle deployment.

## 1. System Requirements
- **OS:** Linux (Ubuntu 20.04/22.04 recommended for DGX)
- **Memory:** 128GB unified memory is suitable for the DGX Spark workflow. NLP model memory depends on the selected engine/model and its quantization; the pipeline does not assume multi-GPU availability.
- **Storage:** Provision space for the competition DICOMs plus the selected cache. The default cache is approximately **227 GB decimal** for 4,407 studies at 518×518×32×6, before metadata and scratch space. Keep at least 1 TB free for the full dataset, cache, checkpoints, and logs.
- **Hardware:** DGX Spark / GB10 is supported through the regular PyTorch CUDA path. DICOM/cache work is CPU- and NVMe-intensive; it does not use unified memory as a substitute for disk capacity.

---

## 2. Environment Setup

### Step 2.1: Clone the Repository
Clone your completely updated and hardened repository:
```bash
git clone https://github.com/priyanshu-ogdev/Knee_RSNA.git
cd Knee_RSNA
```

### Step 2.2: Install Python Environment
It is highly recommended to use Miniconda or a virtual environment (Python 3.10+):
```bash
conda create -n rsna python=3.10 -y
conda activate rsna
```

### Step 2.3: Install Dependencies
Install all required dependencies. Our patches have ensured that `vllm`, `dicomsdl`, `timm`, `pydicom`, and `tenacity` are fully integrated.
```bash
pip install -r requirements.txt
```

---

## 3. Authentication (.env Configuration)
Both the Kaggle dataset download and the HuggingFace `Nemotron-70B` model require authentication.

Create a `.env` file in the root of the repository (`Knee_RSNA/.env`):
```bash
touch .env
```
Open it and add your keys:
```env
# HuggingFace Token (Required to download the Llama-3.1-Nemotron-70B model)
HF_TOKEN=your_huggingface_token_here

# Kaggle Credentials (Required to download the 570GB dataset)
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
```
*(Make sure to never commit this `.env` file to GitHub!)*

---

## 4. Execution Pipeline

### Phase 1: Prepare and validate the complete dataset
Use the competition data directory containing `train.csv`, `train_series.csv`, and the complete `train_series/` DICOM tree. Run preparation separately before launching training:

```bash
python src/main.py --data_root /path/to/competition --work_dir /path/to/rsna_run --fresh_preprocessing --prepare_only
```

`--fresh_preprocessing` deliberately rebuilds the DICOM index and cache at the selected dimensions; it also forces fresh NLP extraction. This overwrites generated artifacts in `work_dir`, so use it only when a from-scratch run is intended. The default NLP engine is `auto`: it selects vLLM only when available with CUDA and the configured model/quantization is within the supported memory policy; otherwise it selects the explicit clinical-rules extractor. In particular, auto mode will not start the default unquantized 70B model. To request that model, configure supported FP8 quantization first. Set `--nlp_engine vllm` or `--nlp_engine rules` to choose directly; an explicitly unsafe 70B configuration fails before model loading. Runtime failures do not switch engines mid-run. vLLM's memory utilization is capped dynamically using current unified-memory headroom, the configured 88 GiB operating target, and the 20 GiB recovery reserve. Each pseudo-label file has a provenance manifest; the label merge preserves gold values per target and uses report-derived values only where gold is missing.

Preparation checks:
1. NLP outputs are bound to the exact `train.csv`, extractor/prompt version, model, report hashes, and label-engine choice.
2. DICOM series are indexed and compared against both competition CSVs; study/series coverage, geometry, laterality, and selected-slot decode failures are checked.
3. Five folds are regenerated from current data. The compact EDA dHash is used only for cross-study pairs sharing at least two distinct series hashes; single-hash collisions are not treated as confirmed duplicates.
4. The cache shape and source signature are recorded. It does not silently reduce resolution when disk is short; configure `CACHE_IMG_SIZE` / `CACHE_STACK_DEPTH` explicitly if needed.
5. Machine-readable reports are written to `dataset_preparation_manifest.json`, `folds_manifest.json`, `idx/report.json`, and `qc/cache_sanity.json`. Training is not allowed unless the preparation manifest reaches `status: complete` and the full-cache QC gate passes.

If DICOM download is incomplete, the run emits `status: awaiting_dicom` and does not proceed to training. Once preparation passes, launch the ordinary pipeline to resume the validated cache and train:

```bash
python src/main.py --data_root /path/to/competition --work_dir /path/to/rsna_run --model_dir /path/to/rsna_run/dino
```

Without `--prepare_only`, the same pipeline performs the preparation gates first and then proceeds directly to the configured training run. Keep `work_dir` shared for the validated data/cache and use a fresh `model_dir` for each model run; requested fold directories containing old checkpoints/OOF files are rejected rather than mixed into a new run.

The DGX Spark defaults cap preprocessing at 8 workers, use 8 persistent training workers with one prefetched batch each, validate in batches of 8, and infer in batches of 4. The estimated queued training-image payload is logged at startup. System memory is monitored with an 88 GiB operating target, a 100 GiB hard ceiling, and a minimum 20 GiB available-memory guard. These are safety/throughput bounds, not a promise that the machine will sit at a particular memory level: the pipeline does not allocate dummy memory to reach 80–90 GiB. Tune `--num_workers`, `--prefetch_factor`, `--batch_size`, and `--eval_batch_size` only after observing DGX throughput and available memory.

Validation for checkpoint selection runs only on Gold-labeled studies. The subsequent OOF export still predicts every study in each requested held-out fold, and pipeline completion is blocked unless every requested fold has a checkpoint and complete OOF coverage.

---

## 5. Kaggle Offline Submission (Inference)
Kaggle requires an offline (internet-disabled) environment for submission. 

### Step 5.1: Create Kaggle Datasets
1. Zip your entire `Knee_RSNA` folder (which now contains `pipeline_out/` and your trained `foldX_ema.pt` weights). Upload this as a Private Kaggle Dataset named `rsna-knee-models`.
2. Download `.whl` files for `dicomsdl` and `timm` locally, and upload them as a second Private Kaggle Dataset named `rsna-wheels`.

### Step 5.2: Run the Notebook
1. Open a new Kaggle Notebook attached to the competition.
2. Attach both of your private datasets (`rsna-knee-models` and `rsna-wheels`).
3. Import the `notebooks/kaggle_submission.ipynb` code from your repo.
4. Ensure the paths in the notebook point to your attached datasets.
5. Click **Submit**. The notebook is already hardcoded to dynamically deploy Batch=8 Dual-T4 DataParallel inference using your EMA checkpoints.
