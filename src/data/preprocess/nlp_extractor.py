import os
import json
import re
import pandas as pd
import torch
import time

# SOTA Fix: Load .env so the HuggingFace token (HF_TOKEN) is available for downloading the gated Nemotron model.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM, SamplingParams = None, None

# The exact targets expected by the training pipeline
TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", 
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"
]

def build_prompt(report: str) -> str:
    return f"""You are an expert MSK radiologist. Extract the presence of 12 knee abnormalities from the following MRI radiology report. 
The report may be in any language (English, Spanish, Dutch, German, etc.). Translate mentally if needed.

TARGETS:
- ACL (Anterior Cruciate Ligament tear/injury)
- MCL (Medial Collateral Ligament tear/injury)
- Medial Meniscus (tear/injury)
- Lateral Meniscus (tear/injury)
- Medial OA (Medial compartment Osteoarthritis / cartilage loss)
- Lateral OA (Lateral compartment Osteoarthritis / cartilage loss)
- PF OA (Patellofemoral Osteoarthritis / cartilage loss)
- Effusion (Joint fluid)
- Synovitis (Synovial thickening/inflammation)
- Baker's (Baker's cyst / Popliteal cyst)
- Contusion (Bone bruise/contusion)
- Fracture (Bone fracture)

RULES:
1. Output MUST be valid JSON matching the exact output schema.
2. "reasoning": Think step-by-step. Explain your finding based on the quote.
3. "exact_quote": You MUST literally copy/paste the exact sentence from the report that proves the condition. If the condition is not mentioned at all, write "None".
4. "state": EXACTLY ONE of ["present", "absent", "not_stated"].
5. "present": Ligaments/Menisci = explicitly torn/injured. OA/Effusion/Synovitis/Bakers/Contusion/Fracture = explicitly present/seen.
6. "absent" = explicitly normal/intact.
7. "not_stated" = omitted, hedged (e.g. "suspected"), or "None" quote.

REPORT:
{report}

OUTPUT SCHEMA:
{{
  "ACL": {{"reasoning": "...", "exact_quote": "...", "state": "..."}},
  "MCL": {{"reasoning": "...", "exact_quote": "...", "state": "..."}},
  "Medial Meniscus": {{"reasoning": "...", "exact_quote": "...", "state": "..."}},
  "Lateral Meniscus": {{"reasoning": "...", "exact_quote": "...", "state": "..."}},
  "Medial OA": {{"reasoning": "...", "exact_quote": "...", "state": "..."}},
  "Lateral OA": {{"reasoning": "...", "exact_quote": "...", "state": "..."}},
  "PF OA": {{"reasoning": "...", "exact_quote": "...", "state": "..."}},
  "Effusion": {{"reasoning": "...", "exact_quote": "...", "state": "..."}},
  "Synovitis": {{"reasoning": "...", "exact_quote": "...", "state": "..."}},
  "Baker's": {{"reasoning": "...", "exact_quote": "...", "state": "..."}},
  "Contusion": {{"reasoning": "...", "exact_quote": "...", "state": "..."}},
  "Fracture": {{"reasoning": "...", "exact_quote": "...", "state": "..."}}
}}
"""

def parse_json_response(raw_text: str, uid: str) -> dict:
    """Safely extracts JSON from the LLM output and formats it for labels.py"""
    if not raw_text or not isinstance(raw_text, str):
        print(f"[ERROR] Empty raw_text for {uid}")
        return None
        
    clean_json = None
    # SOTA Fix: Case-insensitive and optional 'json' tag for markdown blocks
    md_match = re.search(r'```(?:json)?\s*(.*?)\s*```', raw_text, re.DOTALL | re.IGNORECASE)
    if md_match:
        clean_json = md_match.group(1).strip()
    else:
        # Fallback to brace matching
        start = raw_text.find('{')
        end = raw_text.rfind('}')
        if start != -1 and end != -1 and end > start:
            clean_json = raw_text[start:end+1].strip()
            
    if not clean_json:
        print(f"[ERROR] No JSON block found in output for {uid}")
        return None
        
    # SOTA Fix: Strip trailing commas before closing braces/brackets (fatal for json.loads)
    clean_json_sanitized = re.sub(r',\s*([}\]])', r'\1', clean_json)
        
    try:
        try:
            data = json.loads(clean_json_sanitized)
        except Exception:
            # Fallback using ast.literal_eval with JSON literals translated to Python literals
            import ast
            ast_str = clean_json_sanitized.replace('true', 'True').replace('false', 'False').replace('null', 'None')
            data = ast.literal_eval(ast_str)
            
        out = {"StudyInstanceUID": str(uid).strip()}
        
        # SOTA Fix: Aggressive alphanumeric key normalization to completely eliminate 
        # missing keys due to LLM hallucinating curly quotes (Baker’s vs Baker's) or extra spaces.
        def normalize_key(k):
            return re.sub(r'[^a-zA-Z0-9]', '', str(k)).lower()
            
        target_map = {normalize_key(t): t for t in TARGETS}
        
        normalized_data = {}
        for k, v in data.items():
            norm_k = normalize_key(k)
            if norm_k in target_map:
                normalized_data[target_map[norm_k]] = v
            
        for t in TARGETS:
            val = normalized_data.get(t, {})
            # SOTA Fix: Robust semantic parsing of 'state'
            # Handles booleans, integers, and qualified strings like 'present (mild)', 'absent (normal)'
            raw_state = val.get("state") if isinstance(val, dict) else None
            
            is_present = False
            is_absent = False
            
            if isinstance(raw_state, bool):
                is_present = raw_state
                is_absent = not raw_state
            elif isinstance(raw_state, (int, float)):
                is_present = (raw_state == 1)
                is_absent = (raw_state == 0)
            elif isinstance(raw_state, str):
                s_low = raw_state.lower().strip()
                # Check for explicit absent/normal/intact/no
                if any(x in s_low for x in ["absent", "normal", "intact", "unremarkable", "no tear", "no fracture", "not present", "not seen"]):
                    is_absent = True
                elif any(x in s_low for x in ["present", "torn", "tear", "fracture", "positive"]):
                    is_present = True
            
            if is_present:
                out[t], out[f"{t}_weight"] = 1.0, 0.5
            elif is_absent:
                out[t], out[f"{t}_weight"] = 0.0, 0.5
            else:
                # not_stated / hedged / missing
                if t in ["ACL", "Medial Meniscus", "Lateral Meniscus", "Effusion", "MCL"]:
                    out[t], out[f"{t}_weight"] = 0.0, 0.0 # Strict Mask
                else:
                    out[t], out[f"{t}_weight"] = 0.0, 0.1 # Soft Negative
                    
        return out
    except Exception as e:
        print(f"[ERROR] Failed to parse JSON for {uid}: {e}")
        return None

def run_offline_extraction(data_root: str, out_csv: str, model_id: str = "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"):
    print("=" * 80)
    print("PHASE 1: OFFLINE MASS-BATCH NLP EXTRACTION (vLLM)")
    print("=" * 80)
    
    # Allow model override via env var
    model_id = os.environ.get("LLM_MODEL_ID", model_id)
    
    train_df = pd.read_csv(os.path.join(data_root, 'train.csv'))
    train_df['StudyInstanceUID'] = train_df['StudyInstanceUID'].astype(str).str.strip()
    gold_mask = train_df[TARGETS].notna().any(axis=1)
    
    # Filter reports that exist and are not empty
    to_extract = train_df[~gold_mask & train_df['Report'].notna() & (train_df['Report'].astype(str).str.strip() != '')].copy()
    
    # SOTA Fix: Seamlessly resume from previous crashes by filtering out already processed UIDs
    existing_results = []
    if os.path.exists(out_csv):
        try:
            existing_df = pd.read_csv(out_csv)
            existing_df['StudyInstanceUID'] = existing_df['StudyInstanceUID'].astype(str).str.strip()
            existing_results = existing_df.to_dict('records')
            processed_uids = set(existing_df['StudyInstanceUID'])
            to_extract = to_extract[~to_extract['StudyInstanceUID'].isin(processed_uids)]
            print(f"[INFO] Resuming... Found {len(processed_uids)} already extracted reports.")
        except Exception as e:
            print(f"[WARNING] Could not read existing {out_csv} ({e}). Starting fresh.")
            existing_results = []
    
    if len(to_extract) == 0:
        print("[INFO] No reports left to extract. Exiting.")
        return
        
    if LLM is None:
        raise ImportError("vLLM is not installed. Please install vLLM: pip install vllm")
    print(f"[INFO] Initializing vLLM Engine for {model_id}...")
    
    # SOTA Fix: vLLM does NOT support Tensor Parallelism with bitsandbytes quantization!
    # Running tensor_parallel_size > 1 with bitsandbytes raises ValueError immediately.
    # bitsandbytes INT8 requires ~70GB VRAM, fitting comfortably into a single 80GB/96GB/144GB GPU.
    # If the user sets VLLM_QUANTIZATION="none" or uses unquantized/FP8, TP can scale to all GPUs.
    use_quant = os.environ.get("VLLM_QUANTIZATION", "bitsandbytes")
    if use_quant.lower() in ["bitsandbytes", "bnb"]:
        tp_size = 1
        llm_kwargs = {
            "quantization": "bitsandbytes",
            "load_format": "bitsandbytes",
        }
    else:
        tp_size = max(1, torch.cuda.device_count())
        llm_kwargs = {}
    
    gpu_util = float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.85"))
    
    llm = LLM(
        model=model_id,
        enforce_eager=False,
        max_model_len=4096,
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=gpu_util,
        **llm_kwargs
    )
    
    # SOTA Fix: 2048 tokens is generous for chain of thought reasoning + 12 target JSON
    sampling_params = SamplingParams(temperature=0.0, max_tokens=2048)
    
    print(f"[INFO] Processing {len(to_extract)} reports in chunks...")
    start_time = time.time()
    
    results = existing_results
    CHUNK_SIZE = 500
    
    for i in range(0, len(to_extract), CHUNK_SIZE):
        chunk_df = to_extract.iloc[i:i+CHUNK_SIZE]
        messages_chunk = [[{"role": "user", "content": build_prompt(str(row['Report'])[:12000])}] for _, row in chunk_df.iterrows()]
        uids_chunk = chunk_df['StudyInstanceUID'].tolist()
        
        print(f"\n[INFO] Processing chunk {i//CHUNK_SIZE + 1} / {((len(to_extract)-1)//CHUNK_SIZE) + 1} ({len(chunk_df)} reports)...")
        outputs = llm.chat(messages_chunk, sampling_params, use_tqdm=True)
        
        for output, uid in zip(outputs, uids_chunk):
            text = output.outputs[0].text if (output.outputs and len(output.outputs) > 0) else ""
            res = parse_json_response(text, uid)
            if res:
                results.append(res)
                
        # SOTA Fix: Atomic checkpoint write to prevent corrupted CSV files if killed mid-write
        df_out = pd.DataFrame(results).drop_duplicates(subset=['StudyInstanceUID'], keep='last')
        out_dir = os.path.dirname(out_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        tmp_csv = f"{out_csv}.tmp"
        df_out.to_csv(tmp_csv, index=False)
        os.replace(tmp_csv, out_csv)
        print(f"[INFO] Checkpoint saved atomically. Total extracted: {len(df_out)}")
    
    print(f"[SUCCESS] Processed {len(results)} reports in {time.time() - start_time:.2f} seconds.")
    print(f"[SUCCESS] Saved to {out_csv}. Ready for Phase 2 training pipeline.")

if __name__ == "__main__":
    import kagglehub
    # Dynamically resolve project root relative to this script
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    
    # SOTA Fix: Force Kagglehub cache to root data folder to align with main.py
    os.environ['KAGGLEHUB_CACHE'] = os.path.abspath(os.path.join(PROJECT_ROOT, 'data'))
    
    # SOTA Fix: Redirect HuggingFace cache to root data folder to avoid home directory quota limit (Errno 122)
    if 'HF_HOME' not in os.environ:
        os.environ['HF_HOME'] = os.path.abspath(os.path.join(PROJECT_ROOT, 'data', 'hf_cache'))
    
    print("Checking/Downloading RSNA dataset via Kagglehub...")
    DATA_ROOT = kagglehub.competition_download('rsna-knee-abnormality-detection')
    print(f"[SUCCESS] Dataset located at: {DATA_ROOT}")
    
    OUT = os.path.join(DATA_ROOT, "pseudo_labels.csv")
    run_offline_extraction(DATA_ROOT, OUT)
