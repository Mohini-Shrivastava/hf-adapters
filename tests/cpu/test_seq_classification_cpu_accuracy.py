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

"""CPU accuracy test for standard sequence-classification forward on Spyre.

Two test cases run for each registered sequence-classification model:

  test_manual_path[<key>]
    Loads the model via stock ``AutoModelForSequenceClassification`` on CPU,
    runs a reference forward to get ``[B, num_labels]`` logits, then loads a
    fresh copy, applies ``prepare_for_spyre`` + ``_unwrap_compiled_blocks``,
    and calls ``prefill_sequence_classification`` directly.
    Asserts:
      - Output shape matches ``[B, num_labels]``
      - Per-label cosine similarity across the batch is >= threshold
      - Predicted class ids match exactly

  test_auto_loader[<key>]
    Same comparison, but the adapter side goes through
    ``AutoSpyreModelForSequenceClassification.from_pretrained`` and standard HF
    ``model(**encoded, return_dict=True)``.

DEVICE is patched to ``"cpu"`` by ``tests/conftest.py``; torch.compile is
unwrapped by ``_unwrap_compiled_blocks`` so blocks run eagerly.
"""

import gc
import sys

import pytest
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from hf_adapters.auto_spyre_model import (
    SEQUENCE_CLASSIFICATION_CONFIG_TO_ADAPTER_MODULE_MAPPING,
    AutoSpyreModelForSequenceClassification,
)
from tests.conftest import load_ref_model, resolve_adapter_module_for_test
from tests.cpu.conftest import _unwrap_compiled_blocks
from tests.model_registry import SEQ_CLASSIFICATION_PATHS

pytestmark = pytest.mark.model_harness("seq_classification")

TEXTS: list[str] = [
    "Hello, my dog is cute.",
    "This movie was absolutely terrible.",
    "The weather is nice today.",
]

# fp16 encoder vs fp32 reference: per-label cosine should be very tight.
COSINE_THRESHOLD: float = 0.999


@pytest.mark.parametrize(
    "model_path", SEQ_CLASSIFICATION_PATHS, ids=SEQ_CLASSIFICATION_PATHS
)
def test_manual_path(model_path: str) -> None:
    """Adapter logits via prepare_for_spyre + prefill_sequence_classification match HF reference."""
    hf_common_mod = sys.modules["hf_adapters.hf_common"]
    adapter_module = resolve_adapter_module_for_test(
        model_path,
        mapping=SEQUENCE_CLASSIFICATION_CONFIG_TO_ADAPTER_MODULE_MAPPING,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    encoded = tokenizer(
        TEXTS,
        return_tensors="pt",
        padding=True,
        truncation=True,
        padding_side="right",
        return_attention_mask=True,
    )

    # --- HF reference ---
    ref_model = load_ref_model(
        model_path=model_path,
        adapter_mod=adapter_module,
        auto_model_cls=AutoModelForSequenceClassification,
    )
    with torch.no_grad():
        ref_logits = ref_model(
            **encoded, return_dict=True
        ).logits.float()  # [B, num_labels]
    del ref_model
    gc.collect()

    # --- Adapter path (manual) ---
    model = load_ref_model(
        model_path=model_path,
        adapter_mod=adapter_module,
        auto_model_cls=AutoModelForSequenceClassification,
    )
    adapter_module.prepare_for_spyre(model)
    _unwrap_compiled_blocks(model)

    with torch.no_grad():
        adapter_logits = hf_common_mod.prefill_sequence_classification(
            adapter_module._run_backbone_forward,
            model,
            encoded["input_ids"],
            encoded["attention_mask"],
            token_type_ids=encoded.get("token_type_ids", None),
        ).float()

    id2label = model.config.id2label
    del model
    gc.collect()

    assert (
        adapter_logits.shape == ref_logits.shape
    ), f"shape mismatch: adapter {adapter_logits.shape} vs ref {ref_logits.shape}"
    assert torch.isfinite(adapter_logits).all()

    # Per-sample cosine similarity over the num_labels dimension
    cos = F.cosine_similarity(adapter_logits, ref_logits, dim=-1)  # [B]
    ref_ids = ref_logits.argmax(dim=-1)
    adapter_ids = adapter_logits.argmax(dim=-1)

    print("\n## Sequence Classification CPU Comparison (manual path)\n")
    print("| Text | HF Label | Adapter Label | Cosine | Match |")
    print("|------|----------|---------------|--------|-------|")
    for text, ref_id, adp_id, c in zip(TEXTS, ref_ids, adapter_ids, cos):
        ref_lbl = id2label.get(ref_id.item(), str(ref_id.item()))
        adp_lbl = id2label.get(adp_id.item(), str(adp_id.item()))
        match = "Yes" if ref_id.item() == adp_id.item() else "No"
        print(f"| {text} | {ref_lbl} | {adp_lbl} | {c.item():.6f} | {match} |")

    assert (
        cos.min().item() >= COSINE_THRESHOLD
    ), f"min per-sample cosine {cos.min().item():.6f} < threshold {COSINE_THRESHOLD}"
    assert torch.equal(
        adapter_ids, ref_ids
    ), f"predicted class mismatch: adapter {adapter_ids.tolist()} vs ref {ref_ids.tolist()}"


@pytest.mark.parametrize(
    "model_path", SEQ_CLASSIFICATION_PATHS, ids=SEQ_CLASSIFICATION_PATHS
)
def test_auto_loader(model_path: str) -> None:
    """Standard sequence-classification forward via auto-loader matches HF reference."""
    adapter_module = resolve_adapter_module_for_test(
        model_path,
        mapping=SEQUENCE_CLASSIFICATION_CONFIG_TO_ADAPTER_MODULE_MAPPING,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # --- HF reference ---
    ref_model = load_ref_model(
        model_path=model_path,
        adapter_mod=adapter_module,
        auto_model_cls=AutoModelForSequenceClassification,
    )
    encoded = tokenizer(
        TEXTS,
        return_tensors="pt",
        padding=True,
        truncation=True,
        padding_side="right",
        return_attention_mask=True,
    )
    with torch.no_grad():
        ref_logits = ref_model(
            **encoded, return_dict=True
        ).logits.float()  # [B, num_labels]
    del ref_model
    gc.collect()

    # --- Auto-loader path ---
    model = AutoSpyreModelForSequenceClassification.from_pretrained(model_path)
    _unwrap_compiled_blocks(model)

    with torch.no_grad():
        adapter_logits = model(**encoded, return_dict=True).logits.float()

    del model
    gc.collect()

    assert (
        adapter_logits.shape == ref_logits.shape
    ), f"shape mismatch: adapter {adapter_logits.shape} vs ref {ref_logits.shape}"
    assert torch.isfinite(adapter_logits).all()

    cos = F.cosine_similarity(adapter_logits, ref_logits, dim=-1)  # [B]
    ref_ids = ref_logits.argmax(dim=-1)
    adapter_ids = adapter_logits.argmax(dim=-1)

    print("\n## Sequence Classification CPU Comparison (auto-loader path)\n")
    print("| Text | HF Label | Adapter Label | Cosine | Match |")
    print("|------|----------|---------------|--------|-------|")
    for text, ref_id, adp_id, c in zip(TEXTS, ref_ids, adapter_ids, cos):
        match = "Yes" if ref_id.item() == adp_id.item() else "No"
        print(
            f"| {text} | {ref_id.item()} | {adp_id.item()} | {c.item():.6f} | {match} |"
        )

    assert (
        cos.min().item() >= COSINE_THRESHOLD
    ), f"min per-sample cosine {cos.min().item():.6f} < threshold {COSINE_THRESHOLD}"
    assert torch.equal(
        adapter_ids, ref_ids
    ), f"predicted class mismatch: adapter {adapter_ids.tolist()} vs ref {ref_ids.tolist()}"
