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
    raise ImportError("Please install vLLM to run local extraction: pip install vllm")

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
    # SOTA Fix: Prioritize strict markdown extraction to avoid regex greediness on trailing conversational {}
    clean_json = None
    md_match = re.search(r'```json\s*(.*?)\s*```', raw_text, re.DOTALL)
    if md_match:
        clean_json = md_match.group(1)
    else:
        # Fallback to brace matching, but strictly find the first { and the last } 
        # (Still vulnerable if trailing conversation has }, but much safer than failing immediately)
        start = raw_text.find('{')
        end = raw_text.rfind('}')
        if start != -1 and end != -1 and end > start:
            clean_json = raw_text[start:end+1]
            
    if not clean_json:
        print(f"[ERROR] No JSON block found in output for {uid}")
        return None
        
    try:
        data = json.loads(clean_json)
        out = {"StudyInstanceUID": uid}
        
        # Normalize keys for Baker's cyst (curly quotes)
        normalized_data = {}
        for k, v in data.items():
            norm_k = k.replace("’", "'")
            normalized_data[norm_k] = v
            
        for t in TARGETS:
            val = normalized_data.get(t, {})
            # Ensure state is a string to prevent AttributeError on .lower() if state is null/None
            raw_state = val.get("state")
            if isinstance(raw_state, str):
                # Clean punctuation and whitespace
                state = re.sub(r'[^a-z_]', '', raw_state.lower().strip())
                if state == "present":
                    out[t], out[f"{t}_weight"] = 1.0, 0.5
                elif state == "absent":
                    out[t], out[f"{t}_weight"] = 0.0, 0.5
                else:
                    if t in ["ACL", "Medial Meniscus", "Lateral Meniscus", "Effusion", "MCL"]:
                        out[t], out[f"{t}_weight"] = 0.0, 0.0 # Strict Mask
                    else:
                        out[t], out[f"{t}_weight"] = 0.0, 0.1 # Soft Negative
            else:
                out[t], out[f"{t}_weight"] = 0.0, 0.0 # Fallback Mask
        return out
    except Exception as e:
        print(f"[ERROR] Failed to parse JSON for {uid}")
        return None

def run_offline_extraction(data_root: str, out_csv: str, model_id: str = "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"):
    print("=" * 80)
    print("PHASE 1: OFFLINE MASS-BATCH NLP EXTRACTION (NEMOTRON-70B INT8)")
    print("=" * 80)
    
    train_df = pd.read_csv(os.path.join(data_root, 'train.csv'))
    gold_mask = train_df[TARGETS].notna().any(axis=1)
    to_extract = train_df[~gold_mask & train_df['Report'].notna()].copy()
    
    if len(to_extract) == 0:
        print("[INFO] No reports left to extract. Exiting.")
        return
        
    print(f"[INFO] Initializing vLLM Engine for {model_id}...")
    # Speedup Upgrade: vLLM naturally implements FlashAttention-2, PagedAttention, and Continuous Batching.
    # We load in 8-bit (bitsandbytes) to fit the 70B model into ~70GB of the DGX's 128GB Unified Memory.
    # SOTA Fix: Dynamically scale Tensor Parallelism to prevent OOM on partitioned DGX nodes (e.g. 4x 32GB GPUs)
    tensor_parallel = torch.cuda.device_count()
    
    llm = LLM(
        model=model_id,
        quantization="bitsandbytes", 
        load_format="bitsandbytes",
        enforce_eager=False,
        max_model_len=4096,
        tensor_parallel_size=tensor_parallel,
        gpu_memory_utilization=0.9 # Dedicate 90% of available VRAM to KV cache for massive batching
    )
    
    # Deterministic generation
    sampling_params = SamplingParams(temperature=0.0, max_tokens=1024)
    
    print(f"[INFO] Processing {len(to_extract)} reports in massive parallel batches...")
    start_time = time.time()
    
    # Format prompts as messages
    messages_list = [[{"role": "user", "content": build_prompt(row['Report'])}] for _, row in to_extract.iterrows()]
    uids = to_extract['StudyInstanceUID'].tolist()
    
    # SOTA Fix: Directly use llm.chat to prevent the double-BOS token generation bug caused by manual apply_chat_template
    # This also massively speeds up initialization by skipping the local HuggingFace tokenizer initialization.
    outputs = llm.chat(messages_list, sampling_params, use_tqdm=True)
    
    results = []
    for output, uid in zip(outputs, uids):
        res = parse_json_response(output.outputs[0].text, uid)
        if res:
            results.append(res)
            
    df_out = pd.DataFrame(results)
    
    # SOTA Fix: Ensure output directory exists before saving to prevent FileNotFoundError crash after 1-hour run
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df_out.to_csv(out_csv, index=False)
    
    print(f"[SUCCESS] Processed {len(results)} reports in {time.time() - start_time:.2f} seconds.")
    print(f"[SUCCESS] Saved to {out_csv}. You can now upload this to Kaggle.")

if __name__ == "__main__":
    import sys
    import kagglehub
    # Dynamically resolve project root relative to this script
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    
    # SOTA Fix: Force Kagglehub cache to root data folder to align with main.py
    os.environ['KAGGLEHUB_CACHE'] = os.path.abspath(os.path.join(PROJECT_ROOT, 'data'))
    
    print("Checking/Downloading RSNA dataset via Kagglehub...")
    DATA_ROOT = kagglehub.competition_download('rsna-knee-abnormality-detection')
    print(f"[SUCCESS] Dataset located at: {DATA_ROOT}")
    
    OUT = os.path.join(DATA_ROOT, "pseudo_labels.csv")
    run_offline_extraction(DATA_ROOT, OUT)
