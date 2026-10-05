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
    return f"""You are an expert subspecialty musculoskeletal (MSK) radiologist. 
Accurately extract the presence of the following 12 knee conditions from this MRI radiology report.

TARGETS & CLINICAL DEFINITIONS:
1. ACL: Anterior Cruciate Ligament tear (complete, partial, high-grade, low-grade, or chronic tear). Intact ACL graft/reconstruction = absent. (CAUTION: Do NOT confuse with PCL / Posterior Cruciate).
2. MCL: Medial Collateral Ligament tear or sprain (grade 1, 2, or 3). (CAUTION: Do NOT confuse with LCL / Lateral Collateral. MPFL / Medial Patellofemoral Ligament or medial retinaculum tears are NOT MCL tears).
3. Medial Meniscus: Medial meniscus tear (MM, anterior/posterior horn, body, root, horizontal, radial, flap, bucket-handle, complex, or degenerative tear). (NOTE: Grade 1 or Grade 2 intrameniscal signal / myxoid degeneration WITHOUT articular surface extension/tear = absent. Only true articular surface extension tear (Grade 3) = present).
4. Lateral Meniscus: Lateral meniscus tear (LM, anterior/posterior horn, body, root, horizontal, radial, flap, bucket-handle, or complex tear). (NOTE: Grade 1 or Grade 2 intrameniscal signal / myxoid degeneration WITHOUT articular surface extension/tear = absent. Only true articular surface extension tear (Grade 3) = present).
5. Medial OA: Medial compartment osteoarthritis, cartilage loss, chondral thinning/defect, subchondral sclerosis/osteophytes of medial femoral condyle (MFC) or medial tibial plateau (MTP).
6. Lateral OA: Lateral compartment osteoarthritis, cartilage loss, chondral thinning/defect, subchondral sclerosis/osteophytes of lateral femoral condyle (LFC) or lateral tibial plateau (LTP).
7. PF OA: Patellofemoral osteoarthritis, chondromalacia patellae (CMP), patellar or trochlear cartilage defect / fissure / loss / thinning, patellofemoral joint space narrowing.
8. Effusion: Joint effusion, suprapatellar effusion, intra-articular fluid distension (mild, moderate, or large). (NOTE: Physiological / minimal trace fluid within normal limits = absent. Only pathological joint effusion or distension = present).
9. Synovitis: Synovial thickening, synovitis, synovial proliferation, hypervascular pannus, synovial enhancement.
10. Baker's: Baker's cyst, popliteal cyst, gastrocnemius-semimembranosus bursal distension.
11. Contusion: Bone bruise, bone contusion, trabecular microfracture, bone marrow edema / signal abnormality following trauma. (NOTE: Subchondral sclerosis or subchondral cysts from osteoarthritis without acute marrow edema = absent).
12. Fracture: Cortical bone fracture, subchondral fracture, avulsion fracture, tibial plateau / femoral / patellar / fibular fracture. (NOTE: Old healed fracture without acute fracture = absent/not_stated).

CLINICAL RULES (ZERO TOLERANCE FOR HALLUCINATION):
1. Output MUST be valid JSON matching the exact output schema.
2. "reasoning": Think step-by-step. Analyze findings, compartments, and quotes carefully before determining state.
3. "exact_quote": Copy/paste the EXACT verbatim sentence from the report in its ORIGINAL language (do NOT translate the quote). If the condition is not mentioned at all, write "None".
4. "state": EXACTLY ONE of ["present", "absent", "not_stated"].
5. "present": Finding is explicitly present, torn, injured, seen, or described as abnormal.
6. "absent": Finding is explicitly normal, intact, unremarkable, or without abnormality. Intact surgical graft = "absent".
7. "not_stated": Omitted, hedged (e.g. "cannot exclude", "cannot rule out", "suspected", "questionable", "possible", "borderline", "differential"), or "None" quote.
8. MULTI-LINGUAL: The report may be in any language (English, German, Spanish, Dutch, French, Greek, etc.). Translate mentally to extract findings accurately:
   - German: Kreuzband=ACL, Innenmeniskus=Medial Meniscus, Knorpeldefekt/Gonarthrose=OA, Erguss=Effusion, Knochenmarködem=Contusion, keine Ruptur/intakt=absent.
   - Spanish: LCA=ACL, derrame=effusion, edema óseo=contusion, sin rotura/conservado=absent.
   - Dutch: VKB/voorste kruisband=ACL, hydrops=effusion, beenmergoedeem=contusion, geen scheur/intact=absent.
   - French: LCA=ACL, épanchement=effusion, sans fissure/intact=absent.
   - Greek: ρήξη=tear, αρθρική συλλογή=effusion, οστεομυελικό οίδημα=contusion, ακέραιο/χωρίς ρήξη=absent.
9. CROSS-TALK PREVENTION: PCL (Posterior Cruciate) and LCL (Lateral Collateral) are NOT targets! Never assign PCL findings to ACL, nor LCL/MPFL findings to MCL.
10. CONSISTENCY: If exact_quote is "None", state MUST be "not_stated". Never mark "present" with "None" quote.

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

def clean_txt(s: str) -> str:
    # SOTA Fix: Unicode-aware stripping keeps Greek, Spanish, German, French, Dutch letters
    return re.sub(r'[^\w]', '', str(s).lower(), flags=re.UNICODE)

def parse_json_response(raw_text: str, uid: str, original_report: str = "") -> dict:
    """Safely extracts JSON from the LLM output with a Quadruple-Layer Clinical Hallucination Shield."""
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
        
    try:
        # SOTA Fix: Remove trailing commas before closing braces/brackets which frequently crash json.loads
        clean_json_sanitized = re.sub(r',\s*([}\]])', r'\1', clean_json)
        try:
            data = json.loads(clean_json_sanitized)
        except json.JSONDecodeError:
            # Fallback using ast.literal_eval for non-strict python-like dicts
            import ast
            ast_str = clean_json_sanitized.replace('true', 'True').replace('false', 'False').replace('null', 'None')
            data = ast.literal_eval(ast_str)
            
        out = {"StudyInstanceUID": str(uid).strip()}
        
        # SOTA Fix: Aggressive alphanumeric key normalization to completely eliminate 
        # missing keys due to LLM hallucinating curly quotes (Baker's vs Baker's) or extra spaces.
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
            raw_state = val.get("state") if isinstance(val, dict) else None
            exact_quote = str(val.get("exact_quote", "")).strip() if isinstance(val, dict) else ""
            
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
                # Priority 1: Check for explicit not_stated / none / missing to avoid substring false positives
                if any(ns in s_low for ns in ["not_stated", "not stated", "none", "unknown", "unclear", "missing", "n/a"]):
                    is_present = False
                    is_absent = False
                # Priority 2: Exact matching for single status tokens
                elif s_low in [
                    "absent", "normal", "intact", "unremarkable", "negative", "negativo", 
                    "negatief", "no", "ausente", "afwezig", "unauffällig", "unauffaellig", 
                    "regelrecht", "intakt", "conservado", "conservada", "íntegro", "integro", "χωρίς"
                ]:
                    is_absent = True
                elif s_low in [
                    "present", "torn", "tear", "fracture", "positive", "positivo", 
                    "positief", "presente", "vorhanden", "anwesend", "ρήξη"
                ]:
                    is_present = True
                # Priority 3: Multi-word phrase matching
                else:
                    absent_phrases = [
                        "no tear", "no fracture", "not present", "not seen", "without tear", 
                        "within normal limits", "sin rotura", "geen scheur", "keine ruptur"
                    ]
                    present_phrases = [
                        "present", "torn", "tear", "fracture", "positive", "mild", "moderate", "severe"
                    ]
                    if any(x in s_low for x in absent_phrases):
                        is_absent = True
                    elif any(x in s_low for x in present_phrases):
                        is_present = True
            
            # =========================================================================
            # QUADRUPLE-LAYER CLINICAL HALLUCINATION SHIELD
            # =========================================================================
            q_low = exact_quote.lower().strip()
            if is_present:
                # ---------------------------------------------------------------------
                # Shield 1: Discard ungrounded "present" with empty or "None" quote
                # ---------------------------------------------------------------------
                if q_low in ["none", "null", "n/a", "", "not mentioned", "not stated", "none."]:
                    is_present = False
                    
                # ---------------------------------------------------------------------
                # Shield 2: Anatomical Cross-Talk Prevention
                # ---------------------------------------------------------------------
                # ACL vs PCL
                elif t == "ACL":
                    pcl_terms = ["posterior cruciate", "pcl", "achterste kruisband", "akb", "hinteres kreuzband", "hkb", "cruzado posterior", "lcp", "croise posterieur", "croisé postérieur"]
                    acl_terms = ["anterior", "acl", "voorste", "vkb", "vorderes", "lca", "croise anterieur", "croisé antérieur"]
                    if any(p in q_low for p in pcl_terms) and not any(a in q_low for a in acl_terms):
                        is_present = False
                
                # MCL vs LCL & MPFL
                elif t == "MCL":
                    lcl_terms = ["lateral collateral", "lcl", "fcl", "fibular collateral", "laterale band", "aussenband", "außenband", "colateral lateral", "colateral externo", "lce", "collatéral latéral"]
                    mcl_terms = ["medial", "mcl", "binnenband", "innenband", "colateral medial", "colateral interno", "lcm", "collatéral médial"]
                    if any(l in q_low for l in lcl_terms) and not any(m in q_low for m in mcl_terms):
                        is_present = False
                        
                    mpfl_terms = ["mpfl", "patellofemoral ligament", "patelofemoral", "medial retinaculum", "retináculo medial", "retinaculo medial", "mediale retinaculum"]
                    mcl_specific = ["collateral", "colateral", "mcl", "tibiaal", "tibial", "binnenband", "innenband"]
                    if any(mp in q_low for mp in mpfl_terms) and not any(ms in q_low for ms in mcl_specific):
                        is_present = False

                # Medial Meniscus vs Lateral Meniscus
                elif t == "Medial Meniscus":
                    lm_terms = ["lateral meniscus", "lateralen meniskus", "buitenmeniscus", "menisco lateral", "menisco externo", "menisque lateral"]
                    mm_terms = ["medial", "innenmeniskus", "binnenmeniscus", "menisco medial", "menisco interno", "menisque medial"]
                    if any(lm in q_low for lm in lm_terms) and not any(mm in q_low for mm in mm_terms):
                        is_present = False

                elif t == "Lateral Meniscus":
                    mm_terms = ["medial meniscus", "medialen meniskus", "binnenmeniscus", "innenmeniskus", "menisco medial", "menisco interno", "menisque medial"]
                    lm_terms = ["lateral", "aussenmeniskus", "außenmeniskus", "buitenmeniscus", "externo", "externe"]
                    if any(mm in q_low for mm in mm_terms) and not any(lm in q_low for lm in lm_terms):
                        is_present = False

                # Grade 1 / Grade 2 Intrameniscal Degeneration Shield (Non-tear signal)
                if is_present and t in ["Medial Meniscus", "Lateral Meniscus"]:
                    grade_terms = ["grade 1", "grade i", "grade 2", "grade ii", "grad 1", "grad i", "grad 2", "grad ii", "intrameniscal", "intrameniscale", "myxoid", "myxoide"]
                    no_tear_terms = ["without surface", "no surface extension", "no tear", "ohne riss", "zonder scheur", "sin rotura", "no articular", "sin extension", "sin extensión", "zonder doorbraak", "intact surface"]
                    if any(g in q_low for g in grade_terms) and any(nt in q_low for nt in no_tear_terms):
                        is_present = False
                        is_absent = True

                # Physiological Fluid Shield for Effusion
                if is_present and t == "Effusion":
                    phys_terms = ["physiological", "minimal trace", "trace fluid", "no effusion", "kein erguss", "geen hydrops", "sin derrame", "fisiológico", "fisiologico", "fysiologische"]
                    heavy_terms = ["moderate", "large", "marked", "copious", "severo", "groot", "ausgeprägt", "substantieel"]
                    if any(p in q_low for p in phys_terms) and not any(h in q_low for h in heavy_terms):
                        is_present = False
                        is_absent = True

                # ---------------------------------------------------------------------
                # Shield 3: Multi-Lingual Hedging & Uncertainty Demotion
                # ---------------------------------------------------------------------
                if is_present:
                    hedging_phrases = [
                        # English
                        "cannot exclude", "cannot rule out", "suspected", "questionable", "possible", 
                        "borderline", "differential diagnosis", "may represent", "inconclusive", "not definitely",
                        # Spanish
                        "no descartable", "sospecha", "cuestionable", "no se puede descartar", "dudoso", "dudosa",
                        # Dutch
                        "niet uit te sluiten", "niet geheel uit te sluiten", "verdacht", "mogelijk", "twijfelachtig", "geen zekere",
                        # German
                        "nicht auszuschliessen", "nicht auszuschließen", "fraglich", "differenzialdiagnose", "v.a.", "verdacht auf", "unklar",
                        # French
                        "ne peut etre exclu", "ne peut être exclu", "douteux", "douteuse", "suspect",
                        # Greek
                        "δεν μπορεί να αποκλειστεί", "πιθανή", "αμφίβολ", "ύποπτ"
                    ]
                    if any(h in q_low for h in hedging_phrases):
                        is_present = False

                # ---------------------------------------------------------------------
                # Shield 4: Invert Contradictory Quotes Describing Normal / Intact Structure
                # ---------------------------------------------------------------------
                if is_present:
                    explicit_normal_phrases = [
                        # English
                        "intact", "normal", "unremarkable", "preserved", "in continuity", "without tear",
                        "no tear", "no fracture", "no acute tear", "no evidence of tear", "not torn", "no effusion",
                        # Spanish
                        "sin rotura", "sin desgarro", "sin signos de rotura", "sin lesiones", "sin alteraciones", 
                        "conservado", "conservada", "íntegro", "integro", "sin derrame", "sin fractura", "sin edema", 
                        "dentro de límites normales", "dentro de limites normales",
                        # Dutch
                        "geen scheur", "ongestoord", "geen afwijkingen", "geen meniscusletsel", "geen hydrops", 
                        "geen kraakbeendefect", "slank en doorlopend", "zonder scheur", "zonder ruptuur", 
                        "zonder afwijkingen", "geen fractuur",
                        # German
                        "keine ruptur", "kein riss", "intakt", "regelrecht", "unauffällig", "unauffaellig", 
                        "ohne befund", "ohne riss", "ohne fraktur", "kein erguss", "kein knorpelschaden", "keine meniskusläsion",
                        # French
                        "sans rupture", "sans fissure", "sans anomalie", "sans lesion", "sans lésion", 
                        "sans épanchement", "sans epanchement", "intégrité", "integrite",
                        # Greek
                        "χωρίς ρήξη", "χωρίς κάταγμα", "ακέραι", "φυσιολογικ", "χωρίς παθολογ", "χωρίς συλλογή"
                    ]
                    injury_words = [
                        "tear", "torn", "ruptur", "rotur", "scheur", "riss", "sprain", "fractur", 
                        "fracture", "fraktur", "edema", "oedeem", "ödem", "defect", "loss", "thinning", 
                        "effusion", "erguss", "derrame", "hydrops", "cyst", "kyste", "zyste"
                    ]
                    has_normal = any(n in q_low for n in explicit_normal_phrases)
                    has_injury = any(inj in q_low for inj in injury_words)
                    
                    if has_normal and not has_injury:
                        # Clean normal quote: invert to absent
                        is_present = False
                        is_absent = True
                    elif has_normal and has_injury:
                        # Contains both (e.g. "no tear of the medial meniscus").
                        # Check if injury is explicitly negated:
                        explicit_neg = any(neg in q_low for neg in [
                            "no tear", "without tear", "no acute tear", "sin rotura", "sin signos de rotura",
                            "geen scheur", "zonder scheur", "keine ruptur", "ohne riss", "sans fissure",
                            "sans rupture", "no fracture", "sin fractura", "geen fractuur", "ohne fraktur",
                            "no effusion", "sin derrame", "geen hydrops", "kein erguss", "χωρίς ρήξη"
                        ])
                        pos_injuries = [
                            "acute tear", "complete tear", "partial tear", "radial tear", "horizontal tear", 
                            "bucket-handle", "rotura completa", "rotura parcial", "scheur van", "complexe scheur", 
                            "knochenmarködem", "bone bruise", "joint effusion"
                        ]
                        has_pos_injury = any(p in q_low for p in pos_injuries)
                        if explicit_neg and not has_pos_injury:
                            is_present = False
                            is_absent = True

                # ---------------------------------------------------------------------
                # Shield 5: Grounding Verification Against Original Report
                # ---------------------------------------------------------------------
                if is_present and original_report:
                    clean_q = clean_txt(exact_quote)
                    clean_rep = clean_txt(original_report)
                    if len(clean_q) > 10 and clean_q not in clean_rep:
                        # Check word overlap if direct character substring fails
                        q_words = set(re.findall(r'\b\w{4,}\b', q_low, flags=re.UNICODE))
                        rep_words = set(re.findall(r'\b\w{4,}\b', original_report.lower(), flags=re.UNICODE))
                        overlap = len(q_words & rep_words) / max(1, len(q_words))
                        
                        # Only reject on word overlap if report is English (avoids penalizing mental translations of foreign reports)
                        common_en = {"the", "and", "with", "knee", "tear", "intact", "effusion", "ligament", "meniscus", "fluid"}
                        is_english_report = len(common_en & rep_words) >= 2
                        if is_english_report and overlap < 0.3:
                            # Fabricated quote hallucination
                            is_present = False
                        elif not is_english_report and len(q_words) >= 3 and overlap < 0.15:
                            # Fabricated non-English quote
                            is_present = False
            
            # Map verified findings to labels & confidence weights
            if is_present:
                out[t], out[f"{t}_weight"] = 1.0, 0.5
            elif is_absent:
                out[t], out[f"{t}_weight"] = 0.0, 0.5
            else:
                # not_stated / hedged / missing
                if t in ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Effusion"]:
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
    
    # Case-insensitive report column resolution
    report_col = 'Report' if 'Report' in train_df.columns else ('report' if 'report' in train_df.columns else None)
    if report_col is None:
        raise KeyError("Could not find 'Report' or 'report' column in train.csv")
    
    # Filter reports that exist and are not empty
    to_extract = train_df[~gold_mask & train_df[report_col].notna() & (train_df[report_col].astype(str).str.strip() != '')].copy()
    
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
    
    # DGX Spark GB10 Hardware Telemetry
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[HARDWARE] Detected System: {gpu_name} ({vram_gb:.1f} GB Coherent Unified Memory)")
    
    print(f"[INFO] Initializing vLLM Engine for {model_id} on DGX Spark GB10...")
    
    # SOTA Fix for DGX Spark GB10 (Single Unit Grace Blackwell, 128GB Unified Memory):
    # 1. Single unit = tensor_parallel_size=1
    # 2. bitsandbytes INT8 requires ~70GB, leaving plenty of room on 128GB
    # 3. Memory utilization defaults to 0.80 (102.4 GB) to ensure 25.6 GB CPU RAM is reserved
    #    for the ARM Grace CPU, Linux OS kernel, and Kagglehub I/O buffers without triggering OOM killer.
    use_quant = os.environ.get("VLLM_QUANTIZATION", "bitsandbytes")
    if use_quant.lower() in ["bitsandbytes", "bnb"]:
        tp_size = 1
        llm_kwargs = {
            "quantization": "bitsandbytes",
            "load_format": "bitsandbytes",
        }
    elif use_quant.lower() in ["none", "null", "false", "fp16", "bf16"]:
        tp_size = 1
        llm_kwargs = {}
    elif use_quant.lower() in ["fp8", "fp8_e4m3", "fp8_e5m2"]:
        tp_size = 1
        llm_kwargs = {
            "quantization": "fp8",
        }
    else:
        tp_size = 1
        llm_kwargs = {
            "quantization": use_quant,
        }
    
    gpu_util = float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.80"))
    enforce_eager_flag = os.environ.get("VLLM_ENFORCE_EAGER", "0") in ["1", "true", "True"]
    
    llm = LLM(
        model=model_id,
        enforce_eager=enforce_eager_flag,
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
        raw_reports = [str(r)[:12000] for r in chunk_df[report_col]]
        messages_chunk = [[{"role": "user", "content": build_prompt(r)}] for r in raw_reports]
        uids_chunk = chunk_df['StudyInstanceUID'].tolist()
        
        print(f"\n[INFO] Processing chunk {i//CHUNK_SIZE + 1} / {((len(to_extract)-1)//CHUNK_SIZE) + 1} ({len(chunk_df)} reports)...")
        outputs = llm.chat(messages_chunk, sampling_params, use_tqdm=True)
        
        for output, uid, report_str in zip(outputs, uids_chunk, raw_reports):
            text = output.outputs[0].text if (output.outputs and len(output.outputs) > 0) else ""
            res = parse_json_response(text, uid, original_report=report_str)
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
    
    knee_env = os.environ.get('KNEE_DATA')
    local_data = os.path.abspath(os.path.join(PROJECT_ROOT, 'data'))
    if knee_env and os.path.exists(os.path.join(knee_env, 'train.csv')):
        DATA_ROOT = os.path.abspath(knee_env)
        print(f"[SUCCESS] Dataset located via KNEE_DATA at: {DATA_ROOT}")
    elif os.path.exists(os.path.join(local_data, 'train.csv')):
        DATA_ROOT = local_data
        print(f"[SUCCESS] Dataset already present locally at: {DATA_ROOT}")
    else:
        print("Checking/Downloading RSNA dataset via Kagglehub...")
        DATA_ROOT = kagglehub.competition_download('rsna-knee-abnormality-detection')
        print(f"[SUCCESS] Dataset located at: {DATA_ROOT}")
    
    OUT = os.path.join(DATA_ROOT, "pseudo_labels.csv")
    run_offline_extraction(DATA_ROOT, OUT)
