"""Fetch the top 1000 generative models from Hugging Face, ranked by downloads."""

import csv
import sys

from huggingface_hub import HfApi


def fetch_top_generative_models(limit=1000, output_csv="top_generative_models.csv"):
    api = HfApi()

    print(f"Fetching top {limit} text-generation models by downloads...")
    models = list(
        api.list_models(
            pipeline_tag="text-generation",
            sort="downloads",
            limit=limit,
            expand=["config", "safetensors"],
        )
    )

    print(f"Retrieved {len(models)} models. Writing to {output_csv}...")

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "rank",
                "model_id",
                "downloads",
                "likes",
                "model_type",
                "architectures",
                "parameters",
                "library",
                "created_at",
            ]
        )

        for rank, m in enumerate(models, start=1):
            model_type = m.config.get("model_type") if m.config else None
            architectures = m.config.get("architectures") if m.config else None
            arch_str = ";".join(architectures) if architectures else None
            param_count = m.safetensors.total if m.safetensors else None
            writer.writerow(
                [
                    rank,
                    m.id,
                    m.downloads,
                    m.likes,
                    model_type,
                    arch_str,
                    param_count,
                    m.library_name,
                    m.created_at.isoformat() if m.created_at else None,
                ]
            )

    print("Done. Top 5 models:")
    for i, m in enumerate(models[:5], start=1):
        print(f"  {i}. {m.id} — {m.downloads:,} downloads")


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    fetch_top_generative_models(limit=limit)
