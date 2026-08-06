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
E2E extractive QA accuracy: HF stock forward (CPU) vs adapter forward (Spyre).

For each registered QA model, loads the model on CPU, runs a reference forward
to get start/end logits, moves it to Spyre, runs the adapter forward via
``prefill_qa``, and asserts that logits are close and the predicted answer span
is identical.

Usage (on Spyre LPAR)::

    # All registered QA models
    pytest -s -vvv tests/spyre/test_e2e_qa_compare_spyre.py

    # Just roberta-base-squad2
    pytest -s -vvv tests/spyre/test_e2e_qa_compare_spyre.py -k roberta
"""

import pytest
import torch
from conftest import load_ref_model
from model_registry import QA_PATHS
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from hf_adapters.auto_spyre_model import (
    SEQUENCE_CLASSIFICATION_CONFIG_TO_ADAPTER_MODULE_MAPPING,
    resolve_adapter_module,
    torch_dtype_for_model_path,
)
from hf_adapters.hf_common import move_model_to_spyre, prefill_qa

# Question-context pairs that cover a range of answer positions so
# span-position correctness is exercised in addition to logit tolerance.
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

# Absolute logit tolerance. Spyre fp16 introduces small numerical differences
# in the backbone hidden states; the QA head amplifies them slightly.
# 0.5 logit units is generous but the span-match assertion is the primary check.
LOGIT_ATOL: float = 0.5


def _predicted_span(
    start_logits: torch.Tensor, end_logits: torch.Tensor
) -> list[tuple[int, int]]:
    """Return (start, end) argmax span for each item in the batch."""
    starts = start_logits.argmax(dim=-1).tolist()
    ends = end_logits.argmax(dim=-1).tolist()
    return list(zip(starts, ends))


def _run_qa_test(model_path: str) -> dict:
    """Full CPU-vs-Spyre comparison for one QA model.

    Returns a result dict with raw logits and match flags for assertion and
    printing.
    """
    adapter = resolve_adapter_module(
        model_path,
        mapping=SEQUENCE_CLASSIFICATION_CONFIG_TO_ADAPTER_MODULE_MAPPING,
    )
    dtype = torch_dtype_for_model_path(model_path)

    print(f"\n{'=' * 70}")
    print(f"  {model_path}")
    print(f"  dtype: {dtype}")
    print(f"{'=' * 70}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = load_ref_model(
        model_path=model_path,
        adapter_mod=adapter,
        auto_model_cls=AutoModelForQuestionAnswering,
    )

    encoded = tokenizer(
        QA_PAIRS,
        return_tensors="pt",
        padding=True,
        truncation=True,
        padding_side="right",
        return_attention_mask=True,
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    token_type_ids = encoded.get("token_type_ids", None)

    lengths = attention_mask.sum(dim=1).tolist()
    print(
        f"  Inputs: {len(QA_PAIRS)} pairs, padded to {input_ids.shape[1]} tokens"
        f" (real lengths: {lengths})"
    )

    # --- HF reference on CPU ---
    print("  Running HF reference on CPU ...")
    with torch.no_grad():
        ref_out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
    ref_start = ref_out.start_logits.float()
    ref_end = ref_out.end_logits.float()
    ref_spans = _predicted_span(ref_start, ref_end)
    print(f"  HF spans: {ref_spans}")
    for i, (q, _) in enumerate(QA_PAIRS):
        s, e = ref_spans[i]
        tokens = input_ids[i, s : e + 1]
        print(f"  [{i}] Q: {q!r}  A: {tokenizer.decode(tokens)!r}")

    # --- Adapter on Spyre ---
    move_model_to_spyre(model=model, module=adapter, dtype=dtype)

    print("  Running adapter on Spyre ...")
    with torch.no_grad():
        spyre_start, spyre_end = prefill_qa(
            adapter._run_backbone_forward,
            model,
            input_ids,
            attention_mask,
            token_type_ids=token_type_ids,
        )
    spyre_start = spyre_start.float()
    spyre_end = spyre_end.float()
    spyre_spans = _predicted_span(spyre_start, spyre_end)
    print(f"  Spyre spans: {spyre_spans}")

    max_start_diff = (spyre_start - ref_start).abs().max().item()
    max_end_diff = (spyre_end - ref_end).abs().max().item()
    span_match = ref_spans == spyre_spans

    print(
        f"  Max abs diff — start: {max_start_diff:.4f}  end: {max_end_diff:.4f}"
        f"  (threshold: {LOGIT_ATOL})"
    )
    print(f"  Span match: {'OK' if span_match else 'MISMATCH'}")

    return {
        "model_path": model_path,
        "input_ids": input_ids,
        "ref_start": ref_start,
        "ref_end": ref_end,
        "spyre_start": spyre_start,
        "spyre_end": spyre_end,
        "ref_spans": ref_spans,
        "spyre_spans": spyre_spans,
        "max_start_diff": max_start_diff,
        "max_end_diff": max_end_diff,
        "start_match": max_start_diff <= LOGIT_ATOL,
        "end_match": max_end_diff <= LOGIT_ATOL,
        "span_match": span_match,
        "tokenizer": tokenizer,
    }


def _print_result_table(result: dict) -> None:
    """Print a per-pair comparison table."""
    print(f"\n## QA E2E: HF (CPU) vs Adapter (Spyre) — {result['model_path']}\n")
    print("| Pair | HF span | Spyre span | HF answer | Spyre answer | Match |")
    print("|------|---------|------------|-----------|--------------|-------|")
    tok = result["tokenizer"]
    for i, (q, _) in enumerate(QA_PAIRS):
        hs, he = result["ref_spans"][i]
        ss, se = result["spyre_spans"][i]
        hf_ans = tok.decode(result["input_ids"][i, hs : he + 1])
        sp_ans = tok.decode(result["input_ids"][i, ss : se + 1])
        ok = "OK" if (hs, he) == (ss, se) else "FAIL"
        print(f"| {i} | ({hs},{he}) | ({ss},{se}) | {hf_ans!r} | {sp_ans!r} | {ok} |")
    print(
        f"\nMax abs diff — start: {result['max_start_diff']:.4f}  "
        f"end: {result['max_end_diff']:.4f}"
    )
    print(f"All spans match: {'OK' if result['span_match'] else 'MISMATCH'}")


@pytest.mark.parametrize("model_path", QA_PATHS, ids=QA_PATHS)
def test_e2e_qa_compare_spyre(model_path: str) -> None:
    result = _run_qa_test(model_path)
    _print_result_table(result)

    assert result["start_match"], (
        f"start_logits max abs diff {result['max_start_diff']:.4f} exceeds {LOGIT_ATOL}.\n"
        f"  HF spans    : {result['ref_spans']}\n"
        f"  Spyre spans : {result['spyre_spans']}"
    )
    assert result["end_match"], (
        f"end_logits max abs diff {result['max_end_diff']:.4f} exceeds {LOGIT_ATOL}.\n"
        f"  HF spans    : {result['ref_spans']}\n"
        f"  Spyre spans : {result['spyre_spans']}"
    )
    assert result["span_match"], (
        f"Predicted span mismatch.\n"
        f"  HF spans    : {result['ref_spans']}\n"
        f"  Spyre spans : {result['spyre_spans']}"
    )
