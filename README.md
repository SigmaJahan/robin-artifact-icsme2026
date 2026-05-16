# ROBIN: Repair Of Bias via INtervention

Replication package for the ICSME 2026 Vision Track paper
**"Toward Localizing and Repairing Bias in Transformer Attention Heads."**

ROBIN is a white-box, head-level fairness debugging method for pretrained
transformer language models. It ranks attention heads by their sensitivity
to fairness probes and removes a small bias subspace from the top-k head
outputs at inference, rather than zeroing the heads.

## Contents

```
robin-artifact-icsme2026/
├── robin/                Python package
│   ├── probes.py           Builds WinoBias / StereoSet pair sets
│   ├── ranking.py          Gradient, SquaredGrad, ROBIN head rankings
│   ├── modeling.py         Head-level head_mask + subspace projection
│   ├── evaluation.py       Bias and language-modeling-loss measurements
│   ├── analysis.py         Plotting routines (matplotlib)
│   ├── run_all.py          End-to-end driver (CLI)
│   └── ...
├── scripts/              Shell + Python entry points
│   ├── local_env.sh        Environment variables (Hugging Face cache, venv)
│   ├── local_smoke.sh      Small end-to-end smoke test (~5 min)
│   ├── local_run.sh        Full single-seed run on all four models
│   ├── local_run_all_seeds.sh   Three-seed driver for paper-grade results
│   ├── regenerate_paper_figures.py    Rebuilds the six paper figures from
│                                       cached outputs/ (no GPU needed)
│   └── regenerate_candidate_figures.py
├── outputs/              Cached numerical results for all 3 seeds
│   ├── full_seed0/
│   ├── full_seed1/
│   └── full_seed2/
├── plots/                The seven figures that appear in the paper
├── requirements.txt      Runtime Python dependencies
├── pyproject.toml        Package metadata
└── README.md             This file
```

The internal method-name string in the cached JSON outputs is still
`"FairEtell"` (the working name used during development). The user-facing
plot labels and prose name are all `ROBIN`. The display mapping happens
inside `robin/analysis.py` via `_DISPLAY_NAME`; leave the JSON keys alone
to keep cached outputs loadable.

## Quick start: reproduce the paper figures without a GPU

The repository already contains the numerical outputs of all three random
seeds, so the six paper figures can be re-rendered in seconds without
re-running any model.

```bash
# 1. Create a Python venv (any Python 3.10+).
python3 -m venv .venv
.venv/bin/pip install matplotlib numpy scipy

# 2. Re-render the seven paper figures into plots/.
PYTHONPATH=. .venv/bin/python scripts/regenerate_paper_figures.py
PYTHONPATH=. .venv/bin/python scripts/regenerate_candidate_figures.py
```

This produces the seven PDFs that appear in the paper:

- `fig_bert_pareto.pdf`, `fig_distilbert_pareto.pdf` (RQ2 Pareto curves)
- `fig_gpt2_quality.pdf` (GPT-2 language-modeling loss collapse)
- `fig_latency_overhead.pdf` (RQ3 latency overhead bars)
- `fig_cross_model_gap.pdf`, `fig_cross_model_quality.pdf` (cross-model)
- `fig_score_concentration.pdf` (head-importance concentration)

## Full reproduction from scratch (requires a GPU)

To re-run the full evaluation on all four models for all three seeds, you
need a CUDA-capable GPU and roughly 1-2 hours of wall time per seed.

```bash
# 1. Install dependencies (PyTorch needs to match your CUDA version).
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install torch torchvision torchaudio   # adjust per your CUDA

# 2. (Optional) point HF_HOME at a shared Hugging Face cache.
export HF_HOME=/path/to/hf-cache

# 3. Run all three seeds back-to-back.
bash scripts/local_run_all_seeds.sh
```

Outputs land in `outputs/full_seed{0,1,2}/`. The figure-regeneration scripts
then consume those JSON files exactly as described in the Quick Start.

## Smoke test (a few minutes on a small GPU)

```bash
bash scripts/local_smoke.sh
```

Runs a 2-pair, 4-text mini-evaluation on DistilBERT to confirm the pipeline
end-to-end without spending an hour.

## Pipeline overview

For each pretrained transformer model the pipeline:

1. Builds WinoBias and StereoSet probe pairs from Hugging Face datasets.
2. Computes three attention-head rankings:
   - **Gradient** — per-head mean attention-gradient magnitude
   - **SquaredGrad** — per-head mean *squared* gradient magnitude
     (empirical-Fisher style, applied at the head level)
   - **ROBIN** — `z(Gradient) + z(SquaredGrad)` (rank fusion)
3. Selects the top-k WinoBias-ranked heads for each method.
4. **Intervenes** on the selected heads:
   - Gradient and SquaredGrad **zero** the heads (whole-head removal).
   - ROBIN estimates a 4-dimensional bias subspace per head from paired
     activation differences (top-r right singular vectors) and **projects
     it out** at inference, leaving the orthogonal complement intact.
5. Measures, at every k in {1, 2, 3, 5, 8, 10, 15, 20}:
   - WinoBias gap (primary fairness metric, lower is better)
   - StereoSet stereotype score (secondary, value of 50 = no preference)
   - Language-modeling loss on WikiText-2 (perplexity for decoders;
     masked-token pseudo-perplexity for encoders; lower is better)
   - Forward-pass latency (mean over 50 measured forwards)

## Models evaluated

- **Encoders**: BERT (`bert-base-uncased`), DistilBERT (`distilbert-base-uncased`)
- **Decoders**: GPT-2 (`gpt2`), DistilGPT-2 (`distilgpt2`)

All weights are loaded via Hugging Face Transformers.

## Datasets

- **WinoBias** (Zhao et al., 2018) — gender pronoun resolution probes.
  Pairs are constructed by replacing every gendered pronoun with its
  opposite, after filtering pairs unchanged by the replacement.
- **StereoSet** (Nadeem et al., 2021) — intrasentence stereotype vs
  anti-stereotype pairs across gender, race, profession, and religion.
- **WikiText-2** (Merity et al., 2017) — general text for language-modeling
  loss.

## Reproducibility notes

- Three seeds (0, 1, 2) shipped in `outputs/`. Per-seed differences in
  `outputs/full_seed*/*/eval/ablation_results.json` reflect only
  language-modeling-loss variation; WinoBias gap is deterministic across
  seeds for these models and probes.
- The per-model figure-selection mapping (BERT seed 0, DistilBERT seed 2,
  DistilGPT-2 seed 0, GPT-2 seed 0) lives at the top of
  `scripts/regenerate_paper_figures.py`.
- Latency numbers reported in the paper are from a single GPU and a single
  batch shape; absolute values will differ on other hardware, but the
  *relative* overhead ROBIN adds (around 7-11% at k=20) is structural.

## Citation

If you use ROBIN, please cite the paper:

```
@inproceedings{robin2026,
  author    = {Anonymous},
  title     = {Toward Localizing and Repairing Bias in Transformer Attention Heads},
  booktitle = {Proc. ICSME (Vision Track)},
  year      = {2026}
}
```
