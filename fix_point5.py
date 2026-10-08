import os, re
path = r'src\data\preprocess\nlp_extractor.py'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

# Fix the fallback issue and log the mode
text = re.sub(r'(\s+)try:\s+from vllm\.sampling_params import GuidedDecodingParams\s+guided = GuidedDecodingParams\(json=schema_str\)\s+sampling_params = SamplingParams\(temperature=0\.0, max_tokens=4096, guided_decoding=guided\)\s+except Exception:\s+try:\s+sampling_params = SamplingParams\(temperature=0\.0, max_tokens=4096, guided_json=schema_str\)\s+except Exception:\s+sampling_params = SamplingParams\(temperature=0\.0, max_tokens=4096\) # Fallback without JSON constraint if version incompatible',
r'''\1try:
\1    from vllm.sampling_params import GuidedDecodingParams
\1    guided = GuidedDecodingParams(json=schema_str)
\1    sampling_params = SamplingParams(temperature=0.0, max_tokens=4096, guided_decoding=guided)
\1    decoding_mode = "GuidedDecodingParams"
\1except Exception:
\1    try:
\1        sampling_params = SamplingParams(temperature=0.0, max_tokens=4096, guided_json=schema_str)
\1        decoding_mode = "guided_json"
\1    except Exception:
\1        sampling_params = SamplingParams(temperature=0.0, max_tokens=4096)
\1        decoding_mode = "unconstrained"
\1        print("[WARNING] vLLM JSON guided decoding failed to initialize. Falling back to unconstrained decoding.")''', text)

# Fix the retry block temperature
text = re.sub(r'(\s+)try:\s+from vllm\.sampling_params import GuidedDecodingParams\s+guided_retry_p = GuidedDecodingParams\(json=schema_str\)\s+retry_params = SamplingParams\(temperature=0\.0, max_tokens=4096, guided_decoding=guided_retry_p\)\s+except Exception:\s+try:\s+retry_params = SamplingParams\(temperature=0\.0, max_tokens=4096, guided_json=schema_str\)\s+except Exception:\s+retry_params = SamplingParams\(temperature=0\.0, max_tokens=4096\)',
r'''\1try:
\1    from vllm.sampling_params import GuidedDecodingParams
\1    guided_retry_p = GuidedDecodingParams(json=schema_str)
\1    retry_params = SamplingParams(temperature=0.4, seed=42+attempt, max_tokens=4096, guided_decoding=guided_retry_p)
\1except Exception:
\1    try:
\1        retry_params = SamplingParams(temperature=0.4, seed=42+attempt, max_tokens=4096, guided_json=schema_str)
\1    except Exception:
\1        retry_params = SamplingParams(temperature=0.4, seed=42+attempt, max_tokens=4096)''', text)

with open(path, 'w', encoding='utf-8') as f:
    f.write(text)
print('Applied Point 5 fixes to nlp_extractor.py')
