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
Shared scaffolding for CPU accuracy tests.

Module-level code below runs at conftest import — i.e. before any test module
in tests/ is loaded — so adapter modules bind to a CPU-patched ``hf_common``.
The trick: load ``hf_common.py`` via importlib, set ``DEVICE = "cpu"``, install
it in ``sys.modules`` under the canonical name, then synthesize an
``hf_adapters`` package pointing at the source directory. Subsequent
``import hf_adapters.X`` calls find our patched version first.

The defensive ``assert`` at the top of this file fails loudly if anything
imported ``hf_adapters`` before pytest reached us — which would lock in the
un-patched DEVICE and silently break CPU tests.
"""

import gc
import importlib.util
import os
import sys
import types

import pytest
from _helpers import (  # noqa: F401  (re-exported for tests via `from conftest import ...`)
    cosine_per_row,
    encode_padded,
    load_hf_causal_lm,
    min_cosine,
    torch_dtype_for,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADAPTERS_DIR = os.path.join(REPO_ROOT, "hf_adapters")

assert "hf_adapters.hf_common" not in sys.modules, (
    "hf_adapters.hf_common was imported before tests/conftest.py ran; "
    "the DEVICE='cpu' patch will not apply. Check for plugins or other "
    "conftests that import hf_adapters at collection time."
)

_common_path = os.path.join(ADAPTERS_DIR, "hf_common.py")
_common_spec = importlib.util.spec_from_file_location(
    "hf_adapters.hf_common", _common_path
)
_common_mod = importlib.util.module_from_spec(_common_spec)
sys.modules["hf_adapters.hf_common"] = _common_mod
_common_spec.loader.exec_module(_common_mod)
_common_mod.DEVICE = "cpu"

_pkg = types.ModuleType("hf_adapters")
_pkg.__path__ = [ADAPTERS_DIR]
sys.modules["hf_adapters"] = _pkg


CAUSAL_LM_MODELS = {
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


EMBEDDING_MODELS = {
    "qwen3_embed": {
        "name": "Qwen3-Embedding 0.6B",
        "path": "Qwen/Qwen3-Embedding-0.6B",
        "adapter": "hf_qwen3.py",
    },
    "qwen2_embed": {
        "name": "GTE-Qwen2-1.5B",
        "path": "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        "adapter": "hf_qwen2.py",
    },
    "e5_mistral": {
        "name": "E5-Mistral-7B",
        "path": "intfloat/e5-mistral-7b-instruct",
        "adapter": "hf_mistral.py",
    },
    "bge_base": {
        "name": "BGE-base-en-v1.5",
        "path": "BAAI/bge-base-en-v1.5",
        "adapter": "hf_bert.py",
    },
    "minilm": {
        "name": "all-MiniLM-L6-v2",
        "path": "sentence-transformers/all-MiniLM-L6-v2",
        "adapter": "hf_bert.py",
    },
}


def _load_adapter(filename):
    """Load an adapter .py file under hf_adapters/ as a real submodule."""
    mod_name = f"hf_adapters.{filename.replace('.py', '')}"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    filepath = os.path.join(ADAPTERS_DIR, filename)
    spec = importlib.util.spec_from_file_location(mod_name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    setattr(_pkg, filename.replace(".py", ""), mod)
    return mod


# Pre-load every adapter referenced by CONFIG_TO_ADAPTER_MODULE_MAPPING, then
# auto_spyre_model itself. Doing this here means tests can grab AutoSpyre*
# off the module without paying the cost on first use.
_auto_path = os.path.join(ADAPTERS_DIR, "auto_spyre_model.py")
_auto_spec = importlib.util.spec_from_file_location(
    "hf_adapters.auto_spyre_model", _auto_path
)
_auto_mod = importlib.util.module_from_spec(_auto_spec)
sys.modules["hf_adapters.auto_spyre_model"] = _auto_mod
_auto_spec.loader.exec_module(_auto_mod)
setattr(_pkg, "auto_spyre_model", _auto_mod)


def _unwrap_compiled_blocks(model):
    """Replace torch.compile-wrapped blocks with their CPU-runnable originals."""
    if not hasattr(model, "_spyre_compiled_blocks"):
        return
    unwrapped = []
    for cb in model._spyre_compiled_blocks:
        orig = getattr(cb, "_orig_mod", getattr(cb, "_torchdynamo_orig_callable", None))
        unwrapped.append(orig if orig is not None else cb)
    model._spyre_compiled_blocks = unwrapped


@pytest.fixture(scope="session")
def hf_common_mod():
    return sys.modules["hf_adapters.hf_common"]


@pytest.fixture(scope="session")
def auto_spyre_model():
    return sys.modules["hf_adapters.auto_spyre_model"]


@pytest.fixture
def load_adapter():
    return _load_adapter


@pytest.fixture
def unwrap_compiled_blocks():
    return _unwrap_compiled_blocks


@pytest.fixture(autouse=True)
def _gc_after_test():
    yield
    gc.collect()
