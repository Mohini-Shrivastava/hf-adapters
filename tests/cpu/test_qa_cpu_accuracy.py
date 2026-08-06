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
CPU accuracy test for extractive question-answering models (e.g. deepset/roberta-base-squad2).

Two parametrised cases per registered QA model:

  test_manual_path[<model_path>]
    Loads the model fresh via ``AutoModelForQuestionAnswering`` (stock HF),
    runs a reference forward to get start/end logits, then loads a second copy,
    applies ``prepare_for_spyre``, unwraps compiled blocks (CPU mode), and calls
    ``prefill_qa``.  Asserts:
      - Output shapes match ``[B, L]`` for both start and end logits
      - Logits are within an absolute tolerance of the HF reference
      - The predicted answer span (argmax of start/end logits) is identical

  test_auto_loader[<model_path>]
    Same comparison but the adapter side goes through
    ``AutoSpyreModelForQuestionAnswering.from_pretrained`` and the attached
    ``model.predict()`` method.  Exercises the full end-to-end auto-loading path.

DEVICE is patched to ``"cpu"`` by ``tests/conftest.py``; torch.compile is
unwrapped by ``_unwrap_compiled_blocks`` so blocks run eagerly.
"""

import gc
import sys

import pytest
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from hf_adapters.auto_spyre_model import (
    SEQUENCE_CLASSIFICATION_CONFIG_TO_ADAPTER_MODULE_MAPPING,
    resolve_adapter_module,
)
from hf_adapters.hf_common import prefill_qa
from tests.conftest import get_dtype_for_cpu, load_ref_model
from tests.cpu.conftest import _unwrap_compiled_blocks
from tests.model_registry import QA_PATHS

# Question-context pairs.  The first pair has a clear factual answer so the
# predicted span can be checked by name; the remaining pairs exercise batching
# and ensure the shape/tolerance assertions hold across multiple examples.
QA_PAIRS: list[tuple[str, str]] = [
    (
        "What is the capital city of France?",
        "France is a country in Western Europe. Its capital city is Paris, "
        "which is also the largest city in France.",
    ),
    (
        "Who wrote the play Hamlet?",
        "Hamlet is a tragedy written by William Shakespeare, believed to have "
        "been written around 1600.",
    ),
    (
        "What language is primarily spoken in Brazil?",
        "Brazil is the largest country in South America. Portuguese is the "
        "official and most widely spoken language in Brazil.",
    ),
]

# Absolute logit tolerance — fp16 backbone vs fp32 reference.
LOGIT_ATOL: float = 0.1


def _hf_reference_logits(
    model_path: str,
    pairs: list[tuple[str, str]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run stock HF forward on CPU, return ``(start_logits, end_logits)`` each ``[B, L]``."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    ref_model = load_ref_model(
        model_path=model_path,
        auto_model_cls=AutoModelForQuestionAnswering,
    )
    ref_model.eval()

    encoded = tokenizer(
        pairs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        padding_side="right",
        return_attention_mask=True,
    )
    with torch.no_grad():
        out = ref_model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            return_dict=True,
        )
    return out.start_logits.float(), out.end_logits.float()


def _predicted_span(
    start_logits: torch.Tensor, end_logits: torch.Tensor
) -> list[tuple[int, int]]:
    """Return the (start, end) argmax span for each item in the batch."""
    starts = start_logits.argmax(dim=-1).tolist()
    ends = end_logits.argmax(dim=-1).tolist()
    return list(zip(starts, ends))


@pytest.mark.parametrize("model_path", QA_PATHS, ids=QA_PATHS)
def test_manual_path(model_path: str) -> None:
    """Adapter logits via prepare_for_spyre + prefill_qa match HF reference."""
    adapter_module = resolve_adapter_module(
        model_path,
        mapping=SEQUENCE_CLASSIFICATION_CONFIG_TO_ADAPTER_MODULE_MAPPING,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # --- HF reference ---
    ref_start, ref_end = _hf_reference_logits(model_path, QA_PAIRS)
    gc.collect()

    # --- Adapter path ---
    model = load_ref_model(
        model_path=model_path,
        auto_model_cls=AutoModelForQuestionAnswering,
    )
    adapter_module.prepare_for_spyre(model)
    _unwrap_compiled_blocks(model)

    encoded = tokenizer(
        QA_PAIRS,
        return_tensors="pt",
        padding=True,
        truncation=True,
        padding_side="right",
        return_attention_mask=True,
    )
    with torch.no_grad():
        adapter_start, adapter_end = prefill_qa(
            adapter_module._run_backbone_forward,
            model,
            encoded["input_ids"],
            encoded["attention_mask"],
            token_type_ids=encoded.get("token_type_ids", None),
        )
    adapter_start = adapter_start.float()
    adapter_end = adapter_end.float()

    del model
    gc.collect()

    assert (
        adapter_start.shape == ref_start.shape
    ), f"start_logits shape mismatch: adapter {adapter_start.shape} vs ref {ref_start.shape}"
    assert (
        adapter_end.shape == ref_end.shape
    ), f"end_logits shape mismatch: adapter {adapter_end.shape} vs ref {ref_end.shape}"

    max_start_diff = (adapter_start - ref_start).abs().max().item()
    assert max_start_diff <= LOGIT_ATOL, (
        f"start_logits max absolute diff {max_start_diff:.4f} exceeds {LOGIT_ATOL}.\n"
        f"  ref    = {ref_start.tolist()}\n"
        f"  adapter= {adapter_start.tolist()}"
    )
    max_end_diff = (adapter_end - ref_end).abs().max().item()
    assert max_end_diff <= LOGIT_ATOL, (
        f"end_logits max absolute diff {max_end_diff:.4f} exceeds {LOGIT_ATOL}.\n"
        f"  ref    = {ref_end.tolist()}\n"
        f"  adapter= {adapter_end.tolist()}"
    )

    ref_spans = _predicted_span(ref_start, ref_end)
    adapter_spans = _predicted_span(adapter_start, adapter_end)
    assert (
        ref_spans == adapter_spans
    ), f"predicted spans differ: ref {ref_spans} vs adapter {adapter_spans}"


@pytest.mark.parametrize("model_path", QA_PATHS, ids=QA_PATHS)
def test_auto_loader(model_path: str) -> None:
    """Logits via AutoSpyreModelForQuestionAnswering.predict() match HF reference."""
    auto_spyre_model_mod = sys.modules["hf_adapters.auto_spyre_model"]
    dtype = get_dtype_for_cpu(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # --- HF reference ---
    ref_start, ref_end = _hf_reference_logits(model_path, QA_PAIRS)
    gc.collect()

    # --- Auto-loader path ---
    model = auto_spyre_model_mod.AutoSpyreModelForQuestionAnswering.from_pretrained(
        model_path, dtype=dtype
    )
    _unwrap_compiled_blocks(model)

    with torch.no_grad():
        adapter_start, adapter_end = model.predict(tokenizer, QA_PAIRS)
    adapter_start = adapter_start.float()
    adapter_end = adapter_end.float()

    del model
    gc.collect()

    assert (
        adapter_start.shape == ref_start.shape
    ), f"start_logits shape mismatch: adapter {adapter_start.shape} vs ref {ref_start.shape}"
    assert (
        adapter_end.shape == ref_end.shape
    ), f"end_logits shape mismatch: adapter {adapter_end.shape} vs ref {ref_end.shape}"

    max_start_diff = (adapter_start - ref_start).abs().max().item()
    assert max_start_diff <= LOGIT_ATOL, (
        f"start_logits max absolute diff {max_start_diff:.4f} exceeds {LOGIT_ATOL}.\n"
        f"  ref    = {ref_start.tolist()}\n"
        f"  adapter= {adapter_start.tolist()}"
    )
    max_end_diff = (adapter_end - ref_end).abs().max().item()
    assert max_end_diff <= LOGIT_ATOL, (
        f"end_logits max absolute diff {max_end_diff:.4f} exceeds {LOGIT_ATOL}.\n"
        f"  ref    = {ref_end.tolist()}\n"
        f"  adapter= {adapter_end.tolist()}"
    )

    ref_spans = _predicted_span(ref_start, ref_end)
    adapter_spans = _predicted_span(adapter_start, adapter_end)
    assert (
        ref_spans == adapter_spans
    ), f"predicted spans differ: ref {ref_spans} vs adapter {adapter_spans}"
