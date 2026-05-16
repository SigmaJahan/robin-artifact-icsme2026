"""Regenerate the six paper figures from existing JSON outputs.

Reads cached ablation_results.json from outputs/full_seed*/{model}/eval/
and writes the six figures used by the paper directly into
ICSME_Vision_2026_Manuscript/figures/. No model loading or GPU is needed.

Run from anywhere:
    python scripts/regenerate_paper_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from robin.analysis import (
    plot_ablation_lines,
    plot_cross_model_ablation,
    plot_latency_overhead_bar,
    plot_tradeoff_scatter,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
PAPER_FIGURES_DIR = REPO_ROOT / "plots"


# Per the FIGURES_EXPLAINED.md ranking guide:
# BERT seed 0, DistilBERT seed 2, DistilGPT-2 seed 0, GPT-2 seed 0.
PAPER_SEEDS = {
    "bert-base-uncased":      "full_seed0",
    "distilbert-base-uncased": "full_seed2",
    "gpt2":                    "full_seed0",
    "distilgpt2":              "full_seed0",
}


def load_ablation(model_name: str) -> List[Dict]:
    seed_dir = PAPER_SEEDS[model_name]
    path = REPO_ROOT / "outputs" / seed_dir / model_name / "eval" / "ablation_results.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)["results"]


def main() -> None:
    PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    bert_rows        = load_ablation("bert-base-uncased")
    distilbert_rows  = load_ablation("distilbert-base-uncased")
    gpt2_rows        = load_ablation("gpt2")
    distilgpt2_rows  = load_ablation("distilgpt2")

    per_model_rows = {
        "BERT":        bert_rows,
        "DistilBERT":  distilbert_rows,
        "GPT-2":       gpt2_rows,
        "DistilGPT-2": distilgpt2_rows,
    }

    # 1) fig_bert_pareto.pdf -- BERT WinoBias gap vs language-modeling loss
    plot_tradeoff_scatter(
        bert_rows,
        fairness_key="winobias_gap",
        utility_key="utility_ppl",
        out_path=str(PAPER_FIGURES_DIR / "fig_bert_pareto.png"),
        xlabel="WinoBias gap",
        ylabel="Language-modeling loss",
    )

    # 2) fig_distilbert_pareto.pdf -- DistilBERT WinoBias gap vs LM loss
    plot_tradeoff_scatter(
        distilbert_rows,
        fairness_key="winobias_gap",
        utility_key="utility_ppl",
        out_path=str(PAPER_FIGURES_DIR / "fig_distilbert_pareto.png"),
        xlabel="WinoBias gap",
        ylabel="Language-modeling loss",
    )

    # 3) fig_gpt2_quality.pdf -- GPT-2 language-modeling loss vs k
    plot_ablation_lines(
        gpt2_rows,
        metric_key="utility_ppl",
        out_path=str(PAPER_FIGURES_DIR / "fig_gpt2_quality.png"),
        ylabel="Language-modeling loss",
    )

    # 4) fig_latency_overhead.pdf -- BERT latency overhead bar chart
    plot_latency_overhead_bar(
        bert_rows,
        out_path=str(PAPER_FIGURES_DIR / "fig_latency_overhead.png"),
    )

    # 5) fig_cross_model_gap.pdf -- four-model WinoBias gap (ROBIN only)
    plot_cross_model_ablation(
        per_model_rows,
        metric_key="winobias_gap",
        out_path=str(PAPER_FIGURES_DIR / "fig_cross_model_gap.png"),
        method="FairEtell",
        ylabel="WinoBias gap",
    )

    # 6) fig_cross_model_quality.pdf -- four-model language-modeling loss (ROBIN)
    plot_cross_model_ablation(
        per_model_rows,
        metric_key="utility_ppl",
        out_path=str(PAPER_FIGURES_DIR / "fig_cross_model_quality.png"),
        method="FairEtell",
        ylabel="Language-modeling loss",
    )

    print(f"Wrote 6 paper figures to {PAPER_FIGURES_DIR}")


if __name__ == "__main__":
    main()
