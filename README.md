# HF Adapters for Spyre

Minimal runtime patches that make stock [HuggingFace Transformers](https://github.com/huggingface/transformers) models run on [Spyre](https://research.ibm.com/blog/ibm-spyre) accelerators.

No forks, no custom model classes — each adapter monkey-patches the
standard HF model at load time, replacing only the operations Spyre
cannot execute natively (RoPE, RMSNorm, KV cache management, generation
loop). Everything else — weights, tokenizer, config — comes straight
from `transformers`.

## Supported Models

| Model | Adapter | Status |
|-------|---------|--------|
| Granite 3.3 (8B) | `hf_granite.py` | Compiles and runs on Spyre |
| Granite 3.3 (2B) | `hf_granite.py` | Compiles and runs on Spyre (head-dim padded) |
| Qwen3 (0.6B) | `hf_qwen3.py` | Compiles and runs on Spyre |
| Granite 4.0 (1B dense) | `hf_granitemoehybrid.py` | Compiles and runs on Spyre |
| SmolLM3 (3B) | `hf_smollm3.py` | Compiles and runs on Spyre |
| Phi-4 mini | `hf_phi3.py` | Blocked on Spyre (sub-stick `head_dim`) |

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full compatibility
matrix, architecture details, and known issues.

## Quick Start

```python
from hf_adapters.hf_granite import load_model, generate
from transformers import AutoTokenizer

model = load_model("ibm-granite/granite-3.3-8b-instruct")
tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-3.3-8b-instruct")

outputs = generate(model, tokenizer, ["What is 2+2?"], max_new_tokens=128)
print(outputs[0])
```

Replace `hf_granite` with `hf_qwen3`, `hf_granitemoehybrid`, `hf_smollm3`,
or `hf_phi3` for other model families.

## Repo Structure

```
README.md
ARCHITECTURE.md                        Detailed status, architecture docs

hf_adapters/
├── hf_common.py              Shared utilities: RoPE precomputation,
│                              RMSNorm patching, LM head padding,
│                              head-dim padding, mask builders,
│                              KV cache helpers, generate loop
├── hf_granite.py              Granite 3.3 adapter
├── hf_qwen3.py                Qwen3 adapter
├── hf_granitemoehybrid.py     Granite 4.0 dense adapter
├── hf_smollm3.py              SmolLM3 adapter
├── hf_phi3.py                 Phi-4 mini adapter
└── __init__.py

tests/
├── test_adapter_cpu_accuracy.py       CPU: adapter vs stock HF
├── test_block_cpu_vs_spyre.py         Per-layer CPU vs Spyre comparison
├── test_e2e_smoke_spyre.py            E2E: load + generate on Spyre
└── test_e2e_token_compare_spyre.py    E2E: HF CPU vs adapter Spyre tokens
```

## Requirements

- Python 3.10+
- PyTorch 2.x
- `transformers`
- `sentencepiece` (for some tokenizers)
- `torch_spyre` (for Spyre tests only — not needed for CPU tests)

## Running Tests

There are two classes of tests: **CPU-only** tests that verify the
adapter produces identical results to stock HuggingFace on CPU, and
**Spyre** tests that run on Spyre hardware.

### CPU Tests (no Spyre required)

These compare the adapter's patched forward pass against stock HF
`transformers` on CPU. Both use the same weights — the test runs stock
HF first, then applies the adapter patches and reruns. Greedy tokens
must match at every step.

```bash
# Run all supported models
python tests/test_adapter_cpu_accuracy.py

# Run a specific model
python tests/test_adapter_cpu_accuracy.py qwen3
python tests/test_adapter_cpu_accuracy.py granite
python tests/test_adapter_cpu_accuracy.py granite4
python tests/test_adapter_cpu_accuracy.py smollm3
```

This downloads model weights from HuggingFace Hub on first run. The test
runs prefill + 4 greedy decode steps and compares logits and top-1 token
selections between stock HF and the adapter. All top-1 tokens should
match (PASS).

### Spyre Tests (requires Spyre hardware)

These must be run on a machine with Spyre accelerators and `torch_spyre`
installed.

**Per-layer block comparison** — creates tiny random-weight models (no
download), runs each decoder block on CPU (uncompiled) and Spyre
(compiled), compares hidden state outputs numerically:

```bash
# Run all models
python tests/test_block_cpu_vs_spyre.py all

# Run a specific model
python tests/test_block_cpu_vs_spyre.py qwen3
python tests/test_block_cpu_vs_spyre.py granite
python tests/test_block_cpu_vs_spyre.py granite4
python tests/test_block_cpu_vs_spyre.py smollm3
```

**E2E smoke test** — loads a real model onto Spyre, generates tokens,
verifies non-trivial output:

```bash
python tests/test_e2e_smoke_spyre.py qwen3
python tests/test_e2e_smoke_spyre.py granite
python tests/test_e2e_smoke_spyre.py granite4
python tests/test_e2e_smoke_spyre.py smollm3
```

**E2E token comparison** — runs stock HF on CPU and the adapter on Spyre
side-by-side, comparing greedy tokens at each step:

```bash
python tests/test_e2e_token_compare_spyre.py qwen3
python tests/test_e2e_token_compare_spyre.py granite
```

Note: Spyre has known numerical accuracy limitations being addressed in
hardware. Token mismatches between CPU and Spyre are expected until those
fixes land.

## License

Apache 2.0
