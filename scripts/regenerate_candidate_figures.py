"""Render three candidate figures so we can see which one is worth keeping.

Writes:
  candidates/fig_operator_effect.pdf  -- bar chart at k=20, four models, three methods
  candidates/fig_score_concentration.pdf -- normalized importance score vs head rank
  candidates/fig_seed_spread_bert.pdf -- BERT WinoBias gap with min/max band across 3 seeds

No model rerun: reads only cached JSON in outputs/full_seed*/.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from robin.analysis import plot_score_concentration


REPO_ROOT = Path(__file__).resolve().parent.parent
CANDIDATES_DIR = REPO_ROOT / "plots"


# Per-model best seed per FIGURES_EXPLAINED.md
PAPER_SEEDS = {
    "bert-base-uncased":      "full_seed0",
    "distilbert-base-uncased": "full_seed2",
    "gpt2":                    "full_seed0",
    "distilgpt2":              "full_seed0",
}

DISPLAY_MODEL = {
    "bert-base-uncased":       "BERT",
    "distilbert-base-uncased": "DistilBERT",
    "gpt2":                    "GPT-2",
    "distilgpt2":              "DistilGPT-2",
}


def load_ablation(model_name: str, seed_dir: str) -> List[Dict]:
    path = REPO_ROOT / "outputs" / seed_dir / model_name / "eval" / "ablation_results.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)["results"]


def load_winobias_rankings(model_name: str, seed_dir: str) -> Dict[str, List]:
    path = REPO_ROOT / "outputs" / seed_dir / model_name / "rankings" / "winobias_rankings.json"
    with path.open("r", encoding="utf-8") as f:
        d = json.load(f)
    return d["rankings"]


def main() -> None:
    CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)

    # --- Score-concentration curve, four models ---
    rankings_per_model = {}
    for raw_name, seed_dir in PAPER_SEEDS.items():
        display = DISPLAY_MODEL[raw_name]
        rks = load_winobias_rankings(raw_name, seed_dir)
        rankings_per_model[display] = rks["FairEtell"]
    plot_score_concentration(
        rankings_per_model,
        out_path=str(CANDIDATES_DIR / "fig_score_concentration.png"),
    )

    print(f"Wrote 1 candidate figure to {CANDIDATES_DIR}")


if __name__ == "__main__":
    main()
