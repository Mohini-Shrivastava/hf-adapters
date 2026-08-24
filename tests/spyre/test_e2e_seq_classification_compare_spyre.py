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
E2E sequence-classification accuracy: HF stock forward (CPU) vs adapter (Spyre).

For each registered sequence-classification model, loads the model on CPU,
runs a reference forward to get ``[B, num_labels]`` logits, moves the same
model instance to Spyre, runs the adapter forward via
``DistilBertSpyreForSequenceClassification.classify()``, and asserts that:

  - Output shape matches ``[B, num_labels]``
  - Per-sample cosine similarity over the label dimension is >= threshold
  - Predicted class ids match exactly between CPU and Spyre

Usage (on Spyre LPAR)::

    # All registered sequence-classification models
    pytest -s -vvv tests/spyre/test_e2e_seq_classification_compare_spyre.py

    # Just the SST-2 DistilBERT checkpoint
    pytest -s -vvv tests/spyre/test_e2e_seq_classification_compare_spyre.py \\
        -k distilbert
"""

import gc

import pytest
import torch
import torch.nn.functional as F
from conftest import load_ref_model
from model_registry import SEQ_CLASSIFICATION_PATHS
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from hf_adapters import DistilBertSpyreForSequenceClassification
from hf_adapters.auto_spyre_model import (
    SEQUENCE_CLASSIFICATION_CONFIG_TO_ADAPTER_MODULE_MAPPING,
    dtype_for_model_path,
    resolve_adapter_module,
)
from hf_adapters.hf_common import move_model_to_spyre, prefill_encoder

pytestmark = pytest.mark.model_harness("seq_classification")

TEXTS: list[str] = [
    "Hello, my dog is cute.",
    "This movie was absolutely terrible.",
    "The weather is nice today.",
]

# Spyre fp16 backbone vs CPU fp32: cosine over num_labels should be very tight.
COSINE_THRESHOLD: float = 0.99


@pytest.mark.parametrize(
    "model_path", SEQ_CLASSIFICATION_PATHS, ids=SEQ_CLASSIFICATION_PATHS
)
def test_manual_path(model_path: str) -> None:
    """Adapter logits via prepare_for_spyre + prefill_encoder (Spyre) match HF CPU reference."""
    adapter = resolve_adapter_module(
        model_path,
        mapping=SEQUENCE_CLASSIFICATION_CONFIG_TO_ADAPTER_MODULE_MAPPING,
    )
    dtype = dtype_for_model_path(model_path, target_device="spyre")

    print(f"\n{'=' * 70}")
    print(f"  {model_path}  dtype={dtype}")
    print(f"{'=' * 70}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    encoded = tokenizer(
        TEXTS,
        return_tensors="pt",
        padding=True,
        truncation=True,
        padding_side="right",
        return_attention_mask=True,
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    token_type_ids = encoded.get("token_type_ids", None)

    # --- HF reference on CPU ---
    model = load_ref_model(
        model_path=model_path,
        adapter_mod=adapter,
        auto_model_cls=AutoModelForSequenceClassification,
    )
    print("  Running HF reference on CPU ...")
    with torch.no_grad():
        ref_logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        ).logits.float()  # [B, num_labels]
    print(f"  HF logits:\n{ref_logits.tolist()}")
    ref_ids = ref_logits.argmax(dim=-1)

    # --- Adapter on Spyre ---
    move_model_to_spyre(model=model, module=adapter, dtype=dtype)

    print("  Running adapter on Spyre ...")
    with torch.no_grad():
        last_hidden = prefill_encoder(
            adapter._run_backbone_forward,
            model,
            input_ids,
            attention_mask,
            token_type_ids=token_type_ids,
        )
        classifier = model.classifier
        cls_device = next(classifier.parameters()).device
        spyre_logits = classifier(last_hidden.to(cls_device)).float()  # [B, num_labels]
    print(f"  Spyre logits:\n{spyre_logits.tolist()}")

    del model
    gc.collect()

    spyre_ids = spyre_logits.argmax(dim=-1)
    cos = F.cosine_similarity(spyre_logits, ref_logits, dim=-1)  # [B]
    id2label = {i: str(i) for i in range(ref_logits.shape[-1])}

    print("\n## Seq Classification: HF (CPU) vs Adapter (Spyre) — manual path\n")
    print("| Text | HF Label | Spyre Label | Cosine | Match |")
    print("|------|----------|-------------|--------|-------|")
    for text, ref_id, spyre_id, c in zip(TEXTS, ref_ids, spyre_ids, cos):
        match = "Yes" if ref_id.item() == spyre_id.item() else "No"
        print(
            f"| {text} | {id2label.get(ref_id.item(), ref_id.item())} "
            f"| {id2label.get(spyre_id.item(), spyre_id.item())} "
            f"| {c.item():.6f} | {match} |"
        )

    assert (
        spyre_logits.shape == ref_logits.shape
    ), f"shape mismatch: spyre {spyre_logits.shape} vs ref {ref_logits.shape}"
    assert torch.isfinite(spyre_logits).all()
    assert cos.min().item() >= COSINE_THRESHOLD, (
        f"min per-sample cosine {cos.min().item():.6f} < threshold {COSINE_THRESHOLD}\n"
        f"  HF logits    : {ref_logits.tolist()}\n"
        f"  Spyre logits : {spyre_logits.tolist()}"
    )
    assert torch.equal(spyre_ids, ref_ids), (
        f"predicted class mismatch.\n"
        f"  HF ids    : {ref_ids.tolist()}\n"
        f"  Spyre ids : {spyre_ids.tolist()}"
    )


@pytest.mark.parametrize(
    "model_path", SEQ_CLASSIFICATION_PATHS, ids=SEQ_CLASSIFICATION_PATHS
)
def test_auto_loader(model_path: str) -> None:
    """Logits via DistilBertSpyreForSequenceClassification.classify() match HF CPU reference."""
    dtype = dtype_for_model_path(model_path, target_device="cpu")

    print(f"\n{'=' * 70}")
    print(f"  {model_path}  dtype={dtype}")
    print(f"{'=' * 70}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    encoded = tokenizer(
        TEXTS,
        return_tensors="pt",
        padding=True,
        truncation=True,
        padding_side="right",
        return_attention_mask=True,
    )

    # --- HF reference on CPU ---
    print("  Running HF reference on CPU ...")
    ref_model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        dtype=dtype,
        device_map="cpu",
    ).eval()
    with torch.no_grad():
        ref_logits = ref_model(
            **encoded, return_dict=True
        ).logits.float()  # [B, num_labels]
    print(f"  HF logits:\n{ref_logits.tolist()}")
    del ref_model
    gc.collect()

    # --- Auto-loader path on Spyre ---
    print("  Running DistilBertSpyreForSequenceClassification.classify() on Spyre ...")
    model = DistilBertSpyreForSequenceClassification.from_pretrained(
        model_path,
        dtype=dtype_for_model_path(model_path, target_device="spyre"),
    )
    with torch.no_grad():
        spyre_logits = model.classify(tokenizer, TEXTS).float()  # [B, num_labels]
    print(f"  Spyre logits:\n{spyre_logits.tolist()}")

    del model
    gc.collect()

    ref_ids = ref_logits.argmax(dim=-1)
    spyre_ids = spyre_logits.argmax(dim=-1)
    cos = F.cosine_similarity(spyre_logits, ref_logits, dim=-1)  # [B]

    print("\n## Seq Classification: HF (CPU) vs Adapter (Spyre) — auto-loader path\n")
    print("| Text | HF id | Spyre id | Cosine | Match |")
    print("|------|-------|----------|--------|-------|")
    for text, ref_id, spyre_id, c in zip(TEXTS, ref_ids, spyre_ids, cos):
        match = "Yes" if ref_id.item() == spyre_id.item() else "No"
        print(
            f"| {text} | {ref_id.item()} | {spyre_id.item()} | {c.item():.6f} | {match} |"
        )

    assert (
        spyre_logits.shape == ref_logits.shape
    ), f"shape mismatch: spyre {spyre_logits.shape} vs ref {ref_logits.shape}"
    assert torch.isfinite(spyre_logits).all()
    assert cos.min().item() >= COSINE_THRESHOLD, (
        f"min per-sample cosine {cos.min().item():.6f} < threshold {COSINE_THRESHOLD}\n"
        f"  HF logits    : {ref_logits.tolist()}\n"
        f"  Spyre logits : {spyre_logits.tolist()}"
    )
    assert torch.equal(spyre_ids, ref_ids), (
        f"predicted class mismatch.\n"
        f"  HF ids    : {ref_ids.tolist()}\n"
        f"  Spyre ids : {spyre_ids.tolist()}"
    )
