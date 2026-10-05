# RSNA 2026: End-to-End Installation & Execution Guide

This guide covers the deployment of the RSNA Knee Abnormality Detection pipeline on a DGX/Linux server for mass parallelized extraction and training, followed by Kaggle deployment.

## 1. System Requirements
- **OS:** Linux (Ubuntu 20.04/22.04 recommended for DGX)
- **RAM:** 64GB+ (128GB+ Unified Memory or GPU memory for `Nemotron-70B` NLP Extraction)
- **Storage:** ~700GB NVMe SSD (570GB for Kaggle Dataset + 70GB for Memmap Cache + 50GB for weights/logs)
- **GPUs:** Multi-GPU supported (e.g., 4x T4, A100, H100). The NLP Extractor and Training engines will automatically utilize all available GPUs via Tensor Parallelism and Distributed Data Parallelism where applicable.

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

### Phase 1: NLP Pseudo-Label Extraction
Before running the main pipeline, you **must** extract the pseudo-labels from the radiology reports. The script will automatically use `kagglehub` to download the competition dataset (~570GB) and then boot up the `vLLM` engine to process the 4,349 reports.

```bash
python src/data/preprocess/nlp_extractor.py
```
**What happens here:**
1. The 570GB dataset is downloaded to `Knee_RSNA/data/competitions/rsna-knee-abnormality-detection`.
2. `vLLM` loads the 70B model across your GPUs.
3. The hardened Chain-of-Thought (CoT) engine extracts the states.
4. It outputs `pseudo_labels.csv` directly into the Kaggle data directory.

---

### Phases 2 & 3: Master Training Pipeline
Once `pseudo_labels.csv` exists, you can launch the master orchestrator.

```bash
python src/main.py
```
**What happens here:**
1. **Idempotency Check:** It verifies the NLP labels exist.
2. **Cache Build (Phase 2):** It parses the DICOMs, applies the mathematically correct single-rescale pixel pipeline, and builds a massive 60GB memory-mapped `cache_v2/` binary array. 
   - *Note: You will see a live ETA in your terminal (e.g., `45.3 studies/s | ETA: 00:01:35`).*
3. **Training (Phase 3):** It initiates the 5-Fold DINOv2 3D Transformer training loop.
4. **Logging:** Everything you see in the terminal is permanently saved to `pipeline_run_YYYYMMDD_HHMMSS.log`.
5. **Output:** Superior Exponential Moving Average (EMA) checkpoints are saved to `pipeline_out/models_foldX/foldX_ema.pt`.

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
