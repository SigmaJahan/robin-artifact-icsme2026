from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


_RC_INITIALIZED = False


def _init_publication_style() -> None:
    """Apply publication-quality matplotlib defaults once per process."""
    global _RC_INITIALIZED
    if _RC_INITIALIZED:
        return
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times", "serif"],
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "lines.linewidth": 1.2,
        "lines.markersize": 4.0,
        "axes.grid": True,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.4,
        "grid.linestyle": "--",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    _RC_INITIALIZED = True


_METHOD_COLORS = {
    "Baseline": "#000000",   # black
    "Gradient": "#0072B2",   # Okabe-Ito blue
    "Curvature": "#D55E00",  # Okabe-Ito vermillion
    "FairEtell": "#009E73",  # Okabe-Ito bluish green
}
_METHOD_MARKERS = {
    "Baseline": "s",
    "Gradient": "o",
    "Curvature": "^",
    "FairEtell": "D",
}

_DISPLAY_NAME = {
    "Baseline":  "Baseline",
    "Gradient":  "Gradient",
    "Curvature": "SquaredGrad",
    "FairEtell": "ROBIN",
}


def _display(method: str) -> str:
    return _DISPLAY_NAME.get(method, method)


def _save_fig(fig, out_path: str) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(p), dpi=300)
    fig.savefig(str(p.with_suffix(".pdf")))
    plt.close(fig)


def topk_overlap(r1: List[Tuple[str, float]], r2: List[Tuple[str, float]], k: int = 20) -> int:
    s1 = set([h for h, _ in r1[:k]])
    s2 = set([h for h, _ in r2[:k]])
    return len(s1.intersection(s2))


def spearman_rank_correlation(r1: List[Tuple[str, float]], r2: List[Tuple[str, float]]) -> float:
    # map head -> rank
    rank1 = {h: i for i, (h, _) in enumerate(r1)}
    rank2 = {h: i for i, (h, _) in enumerate(r2)}
    heads = sorted(set(rank1.keys()).union(rank2.keys()))
    a = [rank1.get(h, len(heads)) for h in heads]
    b = [rank2.get(h, len(heads)) for h in heads]
    rho, _ = spearmanr(a, b)
    return float(0.0 if np.isnan(rho) else rho)


def compute_ranking_stats(rankings: Dict[str, List[Tuple[str, float]]], k: int = 20) -> Dict:
    methods = ["Gradient", "Curvature", "FairEtell"]
    out = {
        "k": k,
        "overlap_topk": {},
        "spearman": {},
    }
    for i, m1 in enumerate(methods):
        for m2 in methods[i + 1 :]:
            out["overlap_topk"][f"{m1}_vs_{m2}"] = topk_overlap(rankings[m1], rankings[m2], k=k)
            out["spearman"][f"{m1}_vs_{m2}"] = spearman_rank_correlation(rankings[m1], rankings[m2])
    return out


def compute_cross_benchmark_stability(
    rankings_a: Dict[str, List[Tuple[str, float]]],
    rankings_b: Dict[str, List[Tuple[str, float]]],
    k: int = 20,
) -> Dict[str, int]:
    methods = ["Gradient", "Curvature", "FairEtell"]
    return {m: topk_overlap(rankings_a[m], rankings_b[m], k=k) for m in methods}


def _ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def plot_overlap_bar(stats: Dict, out_path: str, title: str = "") -> None:
    _init_publication_style()
    pairs = list(stats["overlap_topk"].keys())
    def _pretty(p: str) -> str:
        left, right = p.split("_vs_")
        return f"{_display(left)} vs. {_display(right)}"
    pretty = [_pretty(p) for p in pairs]
    vals = [stats["overlap_topk"][p] for p in pairs]

    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    ax.bar(range(len(pairs)), vals, color="#0072B2", edgecolor="black", linewidth=0.4, width=0.6)
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels(pretty, rotation=20, ha="right")
    ax.set_ylabel(f"Top-{stats['k']} overlap")
    ax.set_ylim(0, max(stats["k"], max(vals) + 1))
    _save_fig(fig, out_path)


def plot_stability_bar(stability: Dict[str, int], k: int, out_path: str, title: str = "") -> None:
    _init_publication_style()
    methods = list(stability.keys())
    vals = [stability[m] for m in methods]
    colors = [_METHOD_COLORS.get(m, "#0072B2") for m in methods]

    fig, ax = plt.subplots(figsize=(3.0, 2.2))
    ax.bar(range(len(methods)), vals, color=colors, edgecolor="black", linewidth=0.4, width=0.6)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([_display(m) for m in methods])
    ax.set_ylabel(f"Top-{k} overlap")
    ax.set_ylim(0, max(k, max(vals) + 1))
    _save_fig(fig, out_path)


def _ci_yerr(ys: List[float], cis: List) -> Optional[np.ndarray]:
    if not cis or any(c is None for c in cis):
        return None
    lower = np.array([y - c[0] for y, c in zip(ys, cis)], dtype=float)
    upper = np.array([c[1] - y for y, c in zip(ys, cis)], dtype=float)
    lower = np.clip(lower, 0.0, None)
    upper = np.clip(upper, 0.0, None)
    return np.vstack([lower, upper])


def plot_ablation_lines(
    rows: List[Dict],
    metric_key: str,
    out_path: str,
    title: str = "",
    ylabel: str = "",
    ci_key: Optional[str] = None,
) -> None:
    _init_publication_style()
    methods = ["Baseline", "Gradient", "Curvature", "FairEtell"]
    if ci_key is None:
        ci_key = metric_key + "_ci"

    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for m in methods:
        sub = [r for r in rows if r["method"] == m]
        sub = sorted(sub, key=lambda x: x["k"])
        if not sub:
            continue
        xs = [r["k"] for r in sub]
        ys = [r[metric_key] for r in sub]
        cis = [r.get(ci_key) for r in sub]
        yerr = _ci_yerr(ys, cis)
        color = _METHOD_COLORS.get(m, None)
        marker = _METHOD_MARKERS.get(m, "o")
        if yerr is not None:
            ax.errorbar(
                xs, ys, yerr=yerr, label=_display(m), color=color, marker=marker,
                capsize=2.0, elinewidth=0.6, capthick=0.6,
            )
        else:
            ax.plot(xs, ys, label=_display(m), color=color, marker=marker)

    ax.set_xlabel("k (modified heads)")
    ax.set_ylabel(ylabel)
    ax.legend(loc="best")
    _save_fig(fig, out_path)


def plot_latency_lines(rows: List[Dict], out_path: str, title: str = "") -> None:
    """Per-method inference latency (mean + p95 band) vs k."""
    _init_publication_style()
    methods = ["Baseline", "Gradient", "Curvature", "FairEtell"]
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    base_mean = None
    for r in rows:
        if r["method"] == "Baseline" and "latency_mean_ms" in r:
            base_mean = float(r["latency_mean_ms"])
            break
    for m in methods:
        sub = [r for r in rows if r["method"] == m and "latency_mean_ms" in r]
        sub = sorted(sub, key=lambda x: x["k"])
        if not sub:
            continue
        if m == "Baseline":
            xs = [r["k"] for r in rows if r["method"] != "Baseline" and "latency_mean_ms" in r]
            xs = sorted(set(xs))
            if not xs:
                continue
            ys = [sub[0]["latency_mean_ms"]] * len(xs)
            ax.plot(xs, ys, label=_display(m), color=_METHOD_COLORS[m], linestyle="--")
            continue
        xs = [r["k"] for r in sub]
        ys = [r["latency_mean_ms"] for r in sub]
        p95 = [r.get("latency_p95_ms", y) for r, y in zip(sub, ys)]
        std = [r.get("latency_std_ms", 0.0) for r in sub]
        color = _METHOD_COLORS.get(m, None)
        marker = _METHOD_MARKERS.get(m, "o")
        ax.plot(xs, ys, label=_display(m), color=color, marker=marker)
        ax.fill_between(xs, [y - s for y, s in zip(ys, std)], [y + s for y, s in zip(ys, std)],
                        color=color, alpha=0.15, linewidth=0)
    ax.set_xlabel("k (modified heads)")
    ax.set_ylabel("Latency (ms / forward)")
    if title:
        ax.set_title(title)
    ax.legend(loc="best")
    _save_fig(fig, out_path)


def plot_latency_overhead_bar(rows: List[Dict], out_path: str, title: str = "") -> None:
    """Per-method latency overhead vs baseline (%) at each k. Grouped bars."""
    _init_publication_style()
    methods = ["Gradient", "Curvature", "FairEtell"]
    base_mean = None
    for r in rows:
        if r["method"] == "Baseline" and "latency_mean_ms" in r:
            base_mean = float(r["latency_mean_ms"])
            break
    if base_mean is None or base_mean <= 0:
        return
    ks = sorted({r["k"] for r in rows if r["method"] != "Baseline" and "latency_mean_ms" in r})
    if not ks:
        return
    fig, ax = plt.subplots(figsize=(4.0, 2.4))
    width = 0.27
    x_idx = np.arange(len(ks))
    for i, m in enumerate(methods):
        vals = []
        for k in ks:
            row = next((r for r in rows if r["method"] == m and r["k"] == k and "latency_mean_ms" in r), None)
            v = (float(row["latency_mean_ms"]) - base_mean) / base_mean * 100.0 if row else 0.0
            vals.append(v)
        ax.bar(x_idx + (i - 1) * width, vals, width=width, label=_display(m),
               color=_METHOD_COLORS.get(m), edgecolor="white", linewidth=0.4)
    ax.axhline(0.0, color="#444444", linewidth=0.5)
    ax.set_xticks(x_idx)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("k (modified heads)")
    ax.set_ylabel("Latency overhead vs baseline (%)")
    if title:
        ax.set_title(title)
    ax.legend(loc="best")
    _save_fig(fig, out_path)


def plot_fairness_latency_tradeoff(rows: List[Dict], fairness_key: str, out_path: str,
                                   xlabel: str = "Latency (ms / forward)",
                                   ylabel: str = "WinoBias gap (pct points)",
                                   title: str = "") -> None:
    """Pareto-style scatter: latency vs fairness, one marker per (method, k)."""
    _init_publication_style()
    methods = ["Baseline", "Gradient", "Curvature", "FairEtell"]
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    for m in methods:
        sub = [r for r in rows if r["method"] == m and "latency_mean_ms" in r]
        sub = sorted(sub, key=lambda x: x["k"])
        if not sub:
            continue
        xs = [r["latency_mean_ms"] for r in sub]
        ys = [r[fairness_key] for r in sub]
        color = _METHOD_COLORS.get(m, None)
        marker = _METHOD_MARKERS.get(m, "o")
        ax.plot(xs, ys, label=_display(m), color=color, marker=marker, linewidth=0.9)
        # Annotate baseline with k=0 marker
        if m == "Baseline":
            ax.annotate("k=0", (xs[0], ys[0]), textcoords="offset points", xytext=(5, 0), fontsize=7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(loc="best")
    _save_fig(fig, out_path)


def write_json(path: str, obj: Dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def write_csv(path: str, rows: List[Dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _parse_head_name(h: str) -> Tuple[int, int]:
    # Expected format: L{layer}H{head}
    try:
        l, hh = h.replace("L", "").split("H")
        return int(l), int(hh)
    except Exception:
        return 0, 0


def plot_head_heatmap(
    ranking: List[Tuple[str, float]],
    n_layers: int,
    n_heads: int,
    out_path: str,
    title: str = "",
) -> None:
    _init_publication_style()
    mat = np.zeros((n_layers, n_heads), dtype=np.float64)
    for head, score in ranking:
        l, h = _parse_head_name(head)
        if 0 <= l < n_layers and 0 <= h < n_heads:
            mat[l, h] = float(score)

    fig_w = max(2.6, n_heads * 0.22 + 1.0)
    fig_h = max(1.8, n_layers * 0.20 + 0.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.grid(False)
    im = ax.imshow(mat, aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_xticks(range(n_heads))
    ax.set_yticks(range(n_layers))
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.ax.tick_params(labelsize=7)
    cbar.outline.set_linewidth(0.4)
    _save_fig(fig, out_path)


def plot_tradeoff_scatter(
    rows: List[Dict],
    fairness_key: str,
    utility_key: str,
    out_path: str,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
) -> None:
    _init_publication_style()
    methods = ["Baseline", "Gradient", "Curvature", "FairEtell"]

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for m in methods:
        sub = [r for r in rows if r["method"] == m]
        sub = sorted(sub, key=lambda x: x["k"])
        if not sub:
            continue
        xs = [r[fairness_key] for r in sub]
        ys = [r[utility_key] for r in sub]
        color = _METHOD_COLORS.get(m, None)
        marker = _METHOD_MARKERS.get(m, "o")
        ax.plot(xs, ys, label=_display(m), color=color, marker=marker, linewidth=1.0)
        # Annotate k=0 baseline
        if m == "Baseline" and xs:
            ax.annotate("k=0", (xs[0], ys[0]), textcoords="offset points",
                        xytext=(5, 5), fontsize=7, color=color)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best")
    _save_fig(fig, out_path)


def plot_spearman_matrix(
    rankings: Dict[str, List[Tuple[str, float]]],
    out_path: str,
    title: str = "",
) -> None:
    """3x3 method-vs-method Spearman correlation matrix on full ranking scores."""
    _init_publication_style()
    methods = ["Gradient", "Curvature", "FairEtell"]

    score_vecs = {}
    for m in methods:
        # Sort by head name to align across methods
        items = sorted(rankings[m], key=lambda x: x[0])
        score_vecs[m] = np.array([s for _, s in items], dtype=np.float64)

    mat = np.zeros((3, 3), dtype=np.float64)
    for i, m1 in enumerate(methods):
        for j, m2 in enumerate(methods):
            rho, _ = spearmanr(score_vecs[m1], score_vecs[m2])
            mat[i, j] = 0.0 if np.isnan(rho) else rho

    fig, ax = plt.subplots(figsize=(2.6, 2.4))
    ax.grid(False)
    im = ax.imshow(mat, aspect="equal", cmap="RdBu_r", vmin=-1.0, vmax=1.0, interpolation="nearest")
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels([_display(m) for m in methods], rotation=20, ha="right")
    ax.set_yticklabels([_display(m) for m in methods])
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                    color="white" if abs(mat[i, j]) > 0.5 else "black", fontsize=7)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.ax.tick_params(labelsize=7)
    cbar.outline.set_linewidth(0.4)
    _save_fig(fig, out_path)


def plot_score_distribution(
    rankings: Dict[str, List[Tuple[str, float]]],
    out_path: str,
    title: str = "",
) -> None:
    """Sorted score curves for the three methods on one axes (head-importance concentration)."""
    _init_publication_style()
    methods = ["Gradient", "Curvature", "FairEtell"]

    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    for m in methods:
        scores = sorted([s for _, s in rankings[m]], reverse=True)
        if not scores:
            continue
        scores = np.array(scores, dtype=np.float64)
        s_max = scores.max() if scores.max() > 0 else 1.0
        ys = scores / s_max
        xs = np.arange(1, len(scores) + 1)
        color = _METHOD_COLORS.get(m, None)
        ax.plot(xs, ys, label=_display(m), color=color, linewidth=1.2)

    ax.set_xlabel("Head rank")
    ax.set_ylabel("Normalized score")
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.legend(loc="best")
    _save_fig(fig, out_path)


def plot_gap_vs_score_scatter(
    per_sample_grad: np.ndarray,
    per_sample_gap: np.ndarray,
    head_names: List[str],
    head_index: int,
    out_path: str,
    title: str = "",
) -> None:
    """Scatter of per-pair fairness gap vs per-pair head gradient norm for one head.

    This is the mechanistic justification for the FairETell formula:
    Spearman(g_h, gap) is the correlation factor in fairetell = curvature * |corr|.
    """
    _init_publication_style()
    if per_sample_grad is None or per_sample_gap is None:
        return
    if head_index < 0 or head_index >= per_sample_grad.shape[1]:
        return

    g = per_sample_grad[:, head_index]
    gap = per_sample_gap
    rho, _ = spearmanr(g, gap)
    if np.isnan(rho):
        rho = 0.0

    fig, ax = plt.subplots(figsize=(3.0, 2.4))
    ax.scatter(g, gap, s=6, alpha=0.5, color="#0072B2", edgecolors="none")
    ax.set_xlabel(f"Head {head_names[head_index]} gradient norm")
    ax.set_ylabel("Probability gap |p_a - p_b|")
    ax.text(0.98, 0.95, rf"$\rho_S$ = {rho:.2f}", transform=ax.transAxes,
            ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="black", lw=0.4))
    _save_fig(fig, out_path)


def plot_cross_model_ablation(
    per_model_rows: Dict[str, List[Dict]],
    metric_key: str,
    out_path: str,
    method: str = "FairEtell",
    title: str = "",
    ylabel: str = "",
    ci_key: Optional[str] = None,
) -> None:
    """One ablation curve per model for a single method (model-generality figure)."""
    _init_publication_style()
    if ci_key is None:
        ci_key = metric_key + "_ci"

    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    # Okabe-Ito colorblind-safe palette
    palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
    for i, (model_name, rows) in enumerate(sorted(per_model_rows.items())):
        sub = [r for r in rows if r["method"] == method]
        sub = sorted(sub, key=lambda x: x["k"])
        if not sub:
            continue
        xs = [r["k"] for r in sub]
        ys = [r[metric_key] for r in sub]
        cis = [r.get(ci_key) for r in sub]
        yerr = _ci_yerr(ys, cis)
        color = palette[i % len(palette)]
        if yerr is not None:
            ax.errorbar(xs, ys, yerr=yerr, label=model_name, color=color,
                        marker="o", capsize=2.0, elinewidth=0.5, capthick=0.5)
        else:
            ax.plot(xs, ys, label=model_name, color=color, marker="o")

    ax.set_xlabel("k (modified heads)")
    ax.set_ylabel(ylabel)
    ax.legend(loc="best")
    _save_fig(fig, out_path)


# ---------------------------------------------------------------------------
# Score-concentration figure (justifies the small-k regime)
# ---------------------------------------------------------------------------


def plot_operator_effect_bars(
    per_model_rows: Dict[str, List[Dict]],
    k_target: int,
    out_path: str,
) -> None:
    """Grouped bar chart at fixed k: per-model RELATIVE loss for three methods.

    Bars show language-modeling loss divided by the model's own baseline loss,
    so all four models share a comparable y-axis. A dashed line at y=1 marks
    the baseline. The vertical gap between Gradient/SquaredGrad bars and the
    ROBIN bar (rankings near-identical) is the operator effect alone.
    """
    _init_publication_style()
    methods = ["Gradient", "Curvature", "FairEtell"]
    models = list(per_model_rows.keys())

    fig, ax = plt.subplots(figsize=(3.8, 2.6))
    width = 0.26
    x_idx = np.arange(len(models))

    # Compute baseline per model
    baselines = {}
    for model_name in models:
        base = next(
            (r for r in per_model_rows[model_name]
             if r["method"] == "Baseline"),
            None,
        )
        baselines[model_name] = float(base["utility_ppl"]) if base else 1.0

    for i, m in enumerate(methods):
        vals = []
        for model_name in models:
            row = next(
                (r for r in per_model_rows[model_name]
                 if r["method"] == m and r["k"] == k_target),
                None,
            )
            b = baselines[model_name] or 1.0
            vals.append(float(row["utility_ppl"]) / b if row else 0.0)
        ax.bar(
            x_idx + (i - 1) * width, vals, width=width,
            label=_display(m), color=_METHOD_COLORS.get(m),
            edgecolor="white", linewidth=0.4,
        )

    ax.axhline(1.0, color=_METHOD_COLORS["Baseline"], linewidth=0.8,
               linestyle="--", label="Baseline")
    ax.set_xticks(x_idx)
    ax.set_xticklabels(models)
    ax.set_ylabel(f"Loss relative to baseline (k={k_target})")
    ax.legend(loc="upper left", ncol=2)
    _save_fig(fig, out_path)


def plot_operator_effect_dumbbell(
    per_model_rows: Dict[str, List[Dict]],
    k_target: int,
    out_path: str,
) -> None:
    """Horizontal dumbbell plot of relative LM loss at fixed k.

    One row per model. On each row, three markers show the relative
    language-modeling loss (loss / baseline_loss) for Gradient, SquaredGrad,
    and ROBIN. A thin gray line connects them within each row so the eye
    follows the operator transition. A dashed vertical at x=1 marks baseline.
    """
    _init_publication_style()
    methods = ["Gradient", "Curvature", "FairEtell"]
    models = list(per_model_rows.keys())

    # Compute relative loss per (model, method)
    baselines = {}
    for model_name in models:
        base = next(
            (r for r in per_model_rows[model_name]
             if r["method"] == "Baseline"),
            None,
        )
        baselines[model_name] = float(base["utility_ppl"]) if base else 1.0

    rel: Dict[str, Dict[str, float]] = {}
    for model_name in models:
        b = baselines[model_name] or 1.0
        rel[model_name] = {}
        for m in methods:
            row = next(
                (r for r in per_model_rows[model_name]
                 if r["method"] == m and r["k"] == k_target),
                None,
            )
            rel[model_name][m] = float(row["utility_ppl"]) / b if row else 0.0

    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    y_idx = np.arange(len(models))

    # Connector line per model
    for j, model_name in enumerate(models):
        xs = [rel[model_name][m] for m in methods]
        ax.plot([min(xs), max(xs)], [y_idx[j], y_idx[j]],
                color="#bbbbbb", linewidth=0.8, zorder=1)

    # Marker per (model, method)
    for m in methods:
        xs = [rel[model_name][m] for model_name in models]
        ax.scatter(xs, y_idx,
                   color=_METHOD_COLORS.get(m),
                   marker=_METHOD_MARKERS.get(m, "o"),
                   s=42, edgecolor="white", linewidth=0.6,
                   label=_display(m), zorder=3)

    # Baseline reference
    ax.axvline(1.0, color=_METHOD_COLORS["Baseline"],
               linewidth=0.8, linestyle="--", zorder=0,
               label="Baseline")

    ax.set_yticks(y_idx)
    ax.set_yticklabels(models)
    ax.set_xlabel(f"Loss relative to baseline (k={k_target})")
    max_val = max(rel[m_n][m] for m_n in models for m in methods)
    ax.set_xlim(
        left=min(0.95, min(rel[m_n][m] for m_n in models for m in methods) - 0.05),
        right=max_val + 0.55,
    )
    # Add headroom above the top row so the legend has a clean place to sit
    ax.set_ylim(top=-0.7, bottom=len(models) - 0.5)
    ax.legend(loc="upper right", ncol=2, frameon=False)
    _save_fig(fig, out_path)


def plot_score_concentration(
    rankings_per_model: Dict[str, List[Tuple[str, float]]],
    out_path: str,
    top_n: int = 30,
) -> None:
    """Sorted importance scores per model on one axes.

    Shows only the top-n highest-scoring heads. A steep early drop visualizes
    that bias-relevant signal lives in a small number of heads. Use to justify
    the small-k regime.
    """
    _init_publication_style()
    # Okabe-Ito colorblind-safe palette
    palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
               "#E69F00", "#56B4E9"]

    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for i, (model_name, scores) in enumerate(sorted(rankings_per_model.items())):
        s = sorted([v for _, v in scores], reverse=True)
        if not s:
            continue
        s_arr = np.array(s, dtype=np.float64)
        s_max = s_arr.max() if s_arr.max() > 0 else 1.0
        ys = s_arr / s_max
        xs = np.arange(1, len(ys) + 1)
        ys = ys[:top_n]
        xs = xs[:top_n]
        ax.plot(xs, ys, label=model_name, color=palette[i % len(palette)],
                linewidth=1.2, marker="o", markersize=3.0)

    # Mark common-budget guide at k=20
    ax.axvline(20, color="#666666", linewidth=0.6, linestyle=":", zorder=0)
    ax.text(20, ax.get_ylim()[1] * 0.95, "k=20", ha="right", va="top",
            fontsize=7, color="#666666")

    ax.set_xlabel("Head rank")
    ax.set_ylabel("Normalized importance score")
    ax.set_ylim(bottom=0)
    ax.legend(loc="best")
    _save_fig(fig, out_path)


def plot_seed_spread(
    per_seed_rows: Dict[int, List[Dict]],
    metric_key: str,
    out_path: str,
    ylabel: str,
    method: str = "FairEtell",
) -> None:
    """Ablation curve for one method across seeds, with min/max band.

    per_seed_rows: {seed_int: ablation_rows} for a single model. Solid line is
    the across-seed mean; shaded band is min-to-max range.
    """
    _init_publication_style()
    seeds = sorted(per_seed_rows.keys())
    if not seeds:
        return

    # Collect (k -> [values across seeds])
    k_to_vals: Dict[int, List[float]] = {}
    for s in seeds:
        for r in per_seed_rows[s]:
            if r.get("method") != method:
                continue
            k = r["k"]
            k_to_vals.setdefault(k, []).append(float(r[metric_key]))

    ks = sorted(k_to_vals.keys())
    means = np.array([np.mean(k_to_vals[k]) for k in ks])
    mins  = np.array([np.min(k_to_vals[k])  for k in ks])
    maxs  = np.array([np.max(k_to_vals[k])  for k in ks])

    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    color = _METHOD_COLORS.get(method, "#009E73")
    ax.fill_between(ks, mins, maxs, color=color, alpha=0.20, linewidth=0,
                    label="seed min-max")
    ax.plot(ks, means, color=color,
            marker=_METHOD_MARKERS.get(method, "D"),
            label=f"{_display(method)} mean")
    ax.set_xlabel("k (modified heads)")
    ax.set_ylabel(ylabel)
    ax.legend(loc="best")
    _save_fig(fig, out_path)
