# RSNA Knee Classification: Installation & Execution

This guide covers the deployment of the RSNA Knee Abnormality Detection pipeline on a DGX/Linux server for mass parallelized extraction and training, followed by Kaggle deployment.

## 1. System Requirements
- **OS:** Linux (Ubuntu 20.04/22.04 recommended for DGX) or Windows for local development.
- **Memory:** 128GB unified memory is suitable for the DGX Spark workflow. NLP model memory depends on the selected engine/model and its quantization; the pipeline does not assume multi-GPU availability.
- **Storage:** Provision space for the competition DICOMs plus the selected cache. The default cache is approximately **227 GB decimal** for 4,407 studies at 518×518×32×6, before metadata and scratch space. Keep at least 1 TB free for the full dataset, cache, checkpoints, and logs.
- **Hardware:** DGX Spark / GB10 is supported through the regular PyTorch CUDA path. DICOM/cache work is CPU- and NVMe-intensive; it does not use unified memory as a substitute for disk capacity.

---

## 2. Environment Setup

### Step 2.1: Clone the Repository
Clone the repository:
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
Install the base Python dependencies:
```bash
pip install -r requirements.txt
```

---

## 3. Authentication (.env Configuration)
Credentials are needed only for the corresponding optional downloads: Kaggle
credentials for `kagglehub` competition downloads, and a Hugging Face token if
you select a gated model. The default NLP engine can use the clinical-rules
extractor and does not require vLLM or a Hugging Face model.

Create a `.env` file in the root of the repository (`Knee_RSNA/.env`):
```bash
touch .env
```
Open it and add your keys:
```env
# Optional: Hugging Face access for a selected gated NLP model
HF_TOKEN=your_huggingface_token_here

# Optional: KaggleHub competition download credentials
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
1. NLP outputs are bound to the exact `train.csv`, extractor/prompt version (`clinical-report-labels-v5`), model, report hashes, and label-engine choice. Missing findings remain strictly masked (`NaN`).
2. DICOM series are indexed and compared against both competition CSVs; study/series coverage, geometry, laterality, and selected-slot decode failures are checked.
3. Five folds are regenerated from current data. The compact EDA dHash is used only for cross-study pairs sharing at least two distinct series hashes; single-hash collisions are not treated as confirmed duplicates.
4. The cache shape and source signature are recorded. It does not silently reduce resolution when disk is short; configure `CACHE_IMG_SIZE` / `CACHE_STACK_DEPTH` explicitly if needed.
5. In-flight cache build tasks are bounded (`2 × workers`, minimum 8) to eliminate excessive task future queues and host RAM bloat.
6. Machine-readable reports are written to `dataset_preparation_manifest.json`, `folds_manifest.json`, `idx/report.json`, and `qc/cache_sanity.json`. Training is not allowed unless the preparation manifest reaches `status: complete` and the full-cache QC gate passes.

If DICOM download is incomplete, the run emits `status: awaiting_dicom` and does not proceed to training. Once preparation passes, launch the ordinary pipeline to resume the validated cache and train:

### Phase 2: Model Training
The production candidate ensemble consists of 10 checkpoints across two diverse families:
- **Family 1 (CoAtNet Hierarchical MIL):** 5 folds using `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k` with learnable gated anatomical priors (`prior_gate` initialized at -2.0, $\sigma(-2) \approx 0.12$).
- **Family 2 (DINOv2 Cross-Slot MIL):** 5 folds using `dinov2-base` with cross-slot transformer interaction.

```bash
# Family 1: CoAtNet Hierarchical MIL
python src/main.py \
  --data_root /path/to/competition \
  --work_dir /path/to/rsna_run \
  --model_dir /path/to/rsna_run/coatnet \
  --model_type coatnet_mil \
  --variant coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k \
  --timm_pooling hierarchical \
  --epochs 15 \
  --batch_size 8 \
  --grad_accum 3 \
  --n_windows_train 5

# Family 2: DINOv2 Cross-Slot MIL
python src/main.py \
  --data_root /path/to/competition \
  --work_dir /path/to/rsna_run \
  --model_dir /path/to/rsna_run/dino \
  --model_type dinov2 \
  --variant dinov2-base \
  --epochs 12 \
  --batch_size 8 \
  --grad_accum 3 \
  --n_windows_train 5
```

Key training features:
- **Macro-AUC aligned loss (`auc_asl`):** Combines 85% Asymmetric Loss (ASL) with 15% Pairwise Ranking AUC (`AUC_RANK_WEIGHT = 0.15`), directly penalizing score inversion between positive and negative studies while respecting weak-label masks.
- **Memory safety & effective batching:** Physical batch size 8 with gradient accumulation 3 (`GRAD_ACCUM = 3`) provides an effective batch of ~24 studies without large VRAM spikes.
- **Dynamic host-memory guard:** DataLoader workers are dynamically budgeted against available host RAM and queued image payloads, dropping to 0 workers under host pressure to prevent prefetch OOMs.
- **Strict gold-checkpoint selection:** Best checkpoints (`fold*_best.pt`) are saved strictly when gold validation macro-AUC improves. `fold*_last.pt` is maintained separately for diagnostics.
- Keep `work_dir` shared for the validated data/cache and use a fresh `model_dir` for each model run; requested fold directories containing old checkpoints/OOF files are rejected rather than mixed into a new run.

---

## 5. Kaggle Offline Submission (Inference)
Kaggle requires an offline (internet-disabled) environment for submission. 

### Step 5.1: Create a checkpoint dataset
1. Upload a zip containing the trained task-specific checkpoint files
   (`fold*_best.pt`, preserving their model-family/fold directories) as a
   private Kaggle dataset. Do not zip the DICOM cache or the entire work
   directory; the cache is large and is rebuilt for test inference.
2. Attach compatible offline wheels only for dependencies missing from the
   Kaggle runtime. Check the runtime first; do not assume `vllm` or
   `dicomsdl` is installed by `requirements.txt`.

### Step 5.2: Run the Notebook
1. Open a new Kaggle Notebook attached to the competition.
2. Attach the checkpoint dataset and any required offline dependency wheels.
3. Import the `notebooks/kaggle_submission.ipynb` code from your repo.
4. Ensure the paths in the notebook point to your attached datasets.
5. Click **Submit**. The notebook loads DINOv2 and CoAtNet checkpoints, prefers `fold*_best.pt`, and falls back to EMA/SWA only when a best checkpoint is absent.
6. **Efficiency track & memory tuning:**
   - **Host RAM safety (30 GiB):** Uses `workers=1` and `chunk=8` to avoid worker memory duplication during test DICOM caching.
   - **GPU utilization (2× 16 GiB T4):** Uses `batch=2` with `torch.nn.DataParallel` (active when `batch >= n_gpus`) to distribute computation across both GPUs.
   - **Native mixed precision:** Uses BF16 where supported, falling back to FP16 on T4.
   - **Deterministic path:** TTA is disabled (`n_tta=1`), and torch compilation is disabled by default (`RSNA_COMPILE=0`) to avoid compilation warm-up delays.
   - **Offline code snapshot:** The notebook embeds an auto-synchronized base64 snapshot of all 34 current `src/` modules, ensuring full offline functionality without external code zips.
