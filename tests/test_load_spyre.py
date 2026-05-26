# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Spyre loading test: every auto-class entry loads cleanly onto Spyre.

No forward pass — just verifies that ``AutoSpyreModelForCausalLM`` and
``AutoSpyreModel`` resolve, prepare, and move the model onto Spyre without
error. Causal-LM entries also check that a ``generate`` method is attached.

Usage (on Spyre pod):
    python3 tests/test_load_spyre.py                          # all models
    python3 tests/test_load_spyre.py qwen3 minilm             # subset
"""

import sys
import time
import traceback

# Registries are duplicated here on purpose: this script runs on the Spyre pod
# as a standalone ``python3`` invocation, not under pytest, so ``tests/conftest.py``
# (which holds the shared ``CAUSAL_LM_MODELS`` / ``EMBEDDING_MODELS``) is never
# imported. Keep these in sync with conftest when models are added or paths
# change. Same convention as ``test_e2e_smoke_spyre.py`` and the other
# ``*_spyre.py`` scripts.
CAUSAL_LM_MODELS = {
    "qwen3": "Qwen/Qwen3-0.6B",
    "qwen2": "Qwen/Qwen2.5-1.5B",
    "llama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "granite": "ibm-granite/granite-3.3-2b-instruct",
    "granite4": "ibm-granite/granite-4.0-1b-base",
    "smollm3": "HuggingFaceTB/SmolLM3-3B-Base",
    "phi4": "microsoft/Phi-4-mini-instruct",
    "olmo": "allenai/OLMo-1B-hf",
    "olmo2": "allenai/OLMo-2-0425-1B",
    "mistral": "mistralai/Mistral-7B-v0.3",
    "granite-vision": "ibm-granite/granite-vision-4.1-4b",
}

EMBEDDING_MODELS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "qwen3_embed": "Qwen/Qwen3-Embedding-0.6B",
}


def load_causal_lm(key):
    import torch

    from hf_adapters import AutoSpyreModelForCausalLM

    path = CAUSAL_LM_MODELS[key]
    dtype = torch.float32 if key == "granite4" else torch.float16

    t0 = time.time()
    model = AutoSpyreModelForCausalLM.from_pretrained(path, dtype=dtype)
    load_s = time.time() - t0

    assert model is not None, f"{key}: from_pretrained returned None"
    assert callable(
        getattr(model, "generate", None)
    ), f"{key}: AutoSpyreModelForCausalLM did not attach generate()"
    return load_s


def load_embedding(key):
    import torch

    from hf_adapters import AutoSpyreModel

    path = EMBEDDING_MODELS[key]
    t0 = time.time()
    model = AutoSpyreModel.from_pretrained(path, dtype=torch.float16)
    load_s = time.time() - t0

    assert model is not None, f"{key}: from_pretrained returned None"
    return load_s


def run(key):
    if key in CAUSAL_LM_MODELS:
        kind = "causal-LM"
        path = CAUSAL_LM_MODELS[key]
        loader = load_causal_lm
    elif key in EMBEDDING_MODELS:
        kind = "embedding"
        path = EMBEDDING_MODELS[key]
        loader = load_embedding
    else:
        raise KeyError(key)

    print(f"\n{'=' * 70}")
    print(f"  [{kind}] {key}: loading {path}")
    print(f"{'=' * 70}")

    load_s = loader(key)
    print(f"  Load time: {load_s:.1f}s  -> PASS")
    return {"key": key, "kind": kind, "status": "PASS", "load_s": load_s}


if __name__ == "__main__":
    all_keys = list(CAUSAL_LM_MODELS.keys()) + list(EMBEDDING_MODELS.keys())
    which = sys.argv[1:] if len(sys.argv) > 1 else all_keys

    results = []
    for key in which:
        if key not in CAUSAL_LM_MODELS and key not in EMBEDDING_MODELS:
            print(f"Unknown: {key}. Options: {all_keys}")
            continue
        try:
            results.append(run(key))
        except Exception:
            print(f"\n!!! {key} FAILED:")
            traceback.print_exc()
            results.append({"key": key, "kind": "?", "status": "ERROR", "load_s": 0.0})

    print("\n## Spyre Load Test Results\n")
    print("| Key | Kind | Status | Load (s) |")
    print("|-----|------|--------|----------|")
    for r in results:
        print(f"| {r['key']} | {r['kind']} | {r['status']} | {r['load_s']:.1f} |")

    failed = [r for r in results if r["status"] != "PASS"]
    sys.exit(1 if failed else 0)
