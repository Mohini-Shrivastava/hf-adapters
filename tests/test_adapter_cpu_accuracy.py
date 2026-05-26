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
CPU accuracy test: compare adapter forward passes against stock HF on CPU.

For each registered causal-LM, two parametrized test cases run:

  test_manual_path[<key>]
    Token-by-token greedy comparison: prefill + 4 decode steps. Loads HF
    once for the reference, then a fresh copy + ``prepare_for_spyre`` for
    the adapter. Asserts the same top-1 token at every step.

  test_auto_loader[<key>]
    End-to-end ``AutoSpyreModelForCausalLM.from_pretrained`` +
    ``model.generate(...)``, compared to ``hf_model.generate(do_sample=False)``.
    Asserts the decoded text matches.

DEVICE='cpu' patching of ``hf_common`` happens once in ``tests/conftest.py``;
this file is plain pytest.
"""

import gc

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT = "The capital of France is"
NUM_DECODE = 4

MODELS = {
    "qwen3": {
        "name": "Qwen3 0.6B",
        "path": "Qwen/Qwen3-0.6B",
        "adapter": "hf_qwen3.py",
    },
    "granite": {
        "name": "Granite 3.3 8B",
        "path": "ibm-granite/granite-3.3-8b-instruct",
        "adapter": "hf_granite.py",
    },
    "granite2b": {
        "name": "Granite 3.3 2B",
        "path": "ibm-granite/granite-3.3-2b-instruct",
        "adapter": "hf_granite.py",
    },
    "granite4": {
        "name": "Granite 4.0 1B",
        "path": "ibm-granite/granite-4.0-1b-base",
        "adapter": "hf_granitemoehybrid.py",
        "dtype": "float32",  # fp16 overflows on CPU due to multipliers
    },
    "smollm3": {
        "name": "SmolLM3 3B",
        "path": "HuggingFaceTB/SmolLM3-3B-Base",
        "adapter": "hf_smollm3.py",
    },
    "llama": {
        "name": "TinyLlama 1.1B",
        "path": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "adapter": "hf_llama.py",
    },
    "phi4": {
        "name": "Phi-4 mini",
        "path": "microsoft/Phi-4-mini-instruct",
        "adapter": "hf_phi3.py",
    },
    "qwen2": {
        "name": "Qwen2.5 1.5B",
        "path": "Qwen/Qwen2.5-1.5B",
        "adapter": "hf_qwen2.py",
    },
    "mistral": {
        "name": "Mistral 7B v0.3",
        "path": "mistralai/Mistral-7B-v0.3",
        "adapter": "hf_mistral.py",
    },
    "olmo": {
        "name": "OLMo 1B",
        "path": "allenai/OLMo-1B-hf",
        "adapter": "hf_olmo.py",
    },
    "olmo2": {
        "name": "OLMo2 1B",
        "path": "allenai/OLMo-2-0425-1B",
        "adapter": "hf_olmo2.py",
    },
    "falcon3": {
        "name": "Falcon 3 1B",
        "path": "tiiuae/Falcon3-1B-Base",
        "adapter": "hf_llama.py",
    },
    "deepseek-coder": {
        "name": "DeepSeek-Coder 1.3B",
        "path": "deepseek-ai/deepseek-coder-1.3b-base",
        "adapter": "hf_llama.py",
    },
    # Ministral 3B is gated — requires HF auth. Tested on Spyre pod only.
    # "ministral": {
    #     "name": "Ministral 3B",
    #     "path": "mistralai/Ministral-3B-Instruct",
    #     "adapter": "hf_mistral.py",
    # },
    "yi": {
        "name": "Yi 1.5 6B",
        "path": "01-ai/Yi-1.5-6B",
        "adapter": "hf_llama.py",
    },
    "granite-vision": {
        "name": "Granite Vision 4.1 4B",
        "path": "ibm-granite/granite-vision-4.1-4b",
        "adapter": "hf_granite_vision.py",
        "load_fn": True,
    },
}


def _torch_dtype(info):
    return torch.float32 if info.get("dtype") == "float32" else torch.float16


def _load_hf_causal_lm(info, torch_dtype, adapter_mod=None):
    """Load the HF reference model, honoring the per-entry ``load_fn`` flag."""
    if info.get("load_fn"):
        # Adapter module exposes load_hf_model for non-standard loading paths
        # (e.g. granite-vision).
        if adapter_mod is None:
            raise RuntimeError("load_fn=True requires adapter_mod")
        return adapter_mod.load_hf_model(info["path"], torch_dtype)
    return AutoModelForCausalLM.from_pretrained(
        info["path"], torch_dtype=torch_dtype, device_map="cpu"
    )


def hf_greedy_steps(model, input_ids, num_decode=NUM_DECODE):
    """Run stock HF model for prefill + N greedy decode steps with DynamicCache.

    Returns a list of dicts with ``logits``, ``token``, ``step``.
    Step 0 is prefill; steps 1..N are decode.
    """
    from transformers import DynamicCache

    results = []
    past = DynamicCache()
    ids = input_ids.clone()
    seq_len = ids.shape[1]

    for step in range(num_decode + 1):
        if step == 0:
            position_ids = torch.arange(seq_len).unsqueeze(0)
        else:
            position_ids = torch.tensor([[seq_len + step - 1]])

        with torch.no_grad():
            out = model(
                input_ids=ids,
                position_ids=position_ids,
                past_key_values=past,
                use_cache=True,
            )

        last_logits = out.logits[0, -1, :].float()
        token = last_logits.argmax().item()
        results.append({"logits": last_logits, "token": token, "step": step})

        past = out.past_key_values
        ids = torch.tensor([[token]])

    return results


def adapter_greedy_steps(run_forward_fn, model, input_ids, num_decode=NUM_DECODE):
    """Run adapter forward for prefill + N greedy decode steps on CPU."""
    results = []
    batch_size = input_ids.shape[0]
    seq_len = input_ids.shape[1]

    num_layers = model.config.num_hidden_layers
    num_kv_heads = model.config.num_key_value_heads
    head_dim = (
        getattr(model, "_spyre_head_dim", None)
        or getattr(model.config, "head_dim", None)
        or model.config.hidden_size // model.config.num_attention_heads
    )
    v_head_dim = getattr(model, "_spyre_v_head_dim", head_dim)
    vocab_size = model.config.vocab_size

    param_dtype = next(model.parameters()).dtype
    max_cache_len = seq_len + num_decode

    key_caches = [
        torch.zeros(
            batch_size, num_kv_heads, max_cache_len, head_dim, dtype=param_dtype
        )
        for _ in range(num_layers)
    ]
    value_caches = [
        torch.zeros(
            batch_size, num_kv_heads, max_cache_len, v_head_dim, dtype=param_dtype
        )
        for _ in range(num_layers)
    ]

    # --- Prefill ---
    position_ids = torch.arange(seq_len).unsqueeze(0)
    causal_mask = torch.zeros((1, 1, seq_len, max_cache_len), dtype=param_dtype)
    for i in range(seq_len):
        causal_mask[:, :, i, i + 1 :] = -torch.inf

    with torch.no_grad():
        logits = run_forward_fn(
            model,
            input_ids,
            position_ids,
            causal_mask,
            key_caches,
            value_caches,
            is_filling=False,
            token_index=0,
            cache_position=0,
        )

    last_logits = logits[0, -1, :].float()[:vocab_size]
    token = last_logits.argmax().item()
    results.append({"logits": last_logits, "token": token, "step": 0})

    cache_len = seq_len

    # --- Decode steps ---
    for step in range(1, num_decode + 1):
        next_ids = torch.tensor([[token]])
        next_pos = torch.tensor([[seq_len + step - 1]])
        decode_mask = torch.zeros((1, 1, 1, max_cache_len), dtype=param_dtype)
        decode_mask[:, :, :, cache_len + 1 :] = -torch.inf

        with torch.no_grad():
            logits = run_forward_fn(
                model,
                next_ids,
                next_pos,
                decode_mask,
                key_caches,
                value_caches,
                is_filling=False,
                token_index=0,
                cache_position=cache_len,
            )

        last_logits = logits[0, -1, :].float()[:vocab_size]
        token = last_logits.argmax().item()
        results.append({"logits": last_logits, "token": token, "step": step})
        cache_len += 1

    return results


@pytest.mark.parametrize("model_key", list(MODELS.keys()), ids=list(MODELS.keys()))
def test_manual_path(model_key, load_adapter, unwrap_compiled_blocks):
    info = MODELS[model_key]
    adapter_mod = load_adapter(info["adapter"])
    torch_dtype = _torch_dtype(info)

    tokenizer = AutoTokenizer.from_pretrained(info["path"])
    input_ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"]

    # Phase 1: HF reference
    model = _load_hf_causal_lm(info, torch_dtype, adapter_mod=adapter_mod)
    model.eval()
    model.requires_grad_(False)
    hf_results = hf_greedy_steps(model, input_ids, num_decode=NUM_DECODE)
    del model
    gc.collect()

    # Phase 2: adapter (fresh load — prepare_for_spyre is destructive)
    model = _load_hf_causal_lm(info, torch_dtype, adapter_mod=adapter_mod)
    model.eval()
    model.requires_grad_(False)
    adapter_mod.prepare_for_spyre(model)
    unwrap_compiled_blocks(model)
    adapter_results = adapter_greedy_steps(
        adapter_mod._run_forward, model, input_ids, num_decode=NUM_DECODE
    )
    del model
    gc.collect()

    for hf_r, ad_r in zip(hf_results, adapter_results):
        step_label = "prefill" if hf_r["step"] == 0 else f"decode-{hf_r['step']}"
        assert hf_r["token"] == ad_r["token"], (
            f"{step_label}: HF token {hf_r['token']!r} "
            f"({tokenizer.decode([hf_r['token']])!r}) != "
            f"adapter token {ad_r['token']!r} "
            f"({tokenizer.decode([ad_r['token']])!r})"
        )


@pytest.mark.parametrize("model_key", list(MODELS.keys()), ids=list(MODELS.keys()))
def test_auto_loader(model_key, auto_spyre_model, unwrap_compiled_blocks, load_adapter):
    info = MODELS[model_key]
    torch_dtype = _torch_dtype(info)
    tokenizer = AutoTokenizer.from_pretrained(info["path"])

    # Phase 1: auto-loader generate
    model = auto_spyre_model.AutoSpyreModelForCausalLM.from_pretrained(
        info["path"], dtype=torch_dtype
    )
    unwrap_compiled_blocks(model)
    auto_outputs = model.generate(tokenizer, [PROMPT], max_new_tokens=NUM_DECODE)
    del model
    gc.collect()

    # Phase 2: HF reference (fresh)
    adapter_mod = load_adapter(info["adapter"]) if info.get("load_fn") else None
    hf_model = _load_hf_causal_lm(info, torch_dtype, adapter_mod=adapter_mod)
    hf_model.eval()
    hf_model.requires_grad_(False)
    encoded = tokenizer(PROMPT, return_tensors="pt")
    with torch.no_grad():
        hf_out = hf_model.generate(
            **encoded, max_new_tokens=NUM_DECODE, do_sample=False
        )
    hf_text = tokenizer.decode(
        hf_out[0][encoded["input_ids"].shape[1] :], skip_special_tokens=True
    )
    del hf_model
    gc.collect()

    assert (
        auto_outputs[0].strip() == hf_text.strip()
    ), f"auto-loader output {auto_outputs[0]!r} != HF reference {hf_text!r}"
