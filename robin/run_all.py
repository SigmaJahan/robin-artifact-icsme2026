from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch

from .analysis import (
    compute_cross_benchmark_stability,
    compute_ranking_stats,
    plot_ablation_lines,
    plot_cross_model_ablation,
    plot_fairness_latency_tradeoff,
    plot_gap_vs_score_scatter,
    plot_head_heatmap,
    plot_latency_lines,
    plot_latency_overhead_bar,
    plot_overlap_bar,
    plot_score_distribution,
    plot_spearman_matrix,
    plot_stability_bar,
    plot_tradeoff_scatter,
    write_csv,
    write_json,
)
from .evaluation import evaluate_ablation, save_json
from .modeling import load_model, num_layers_heads
from .probes import build_stereoset_pairs, build_winobias_pairs, write_pairs_csv
from .ranking import compute_rankings, compute_utility_head_curvature, save_rankings_json


def _safe_name(model_name: str) -> str:
    return model_name.replace('/', '__')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', type=str, default='outputs')
    ap.add_argument('--run_id', type=str, default='run')
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--models', nargs='+', required=True)
    ap.add_argument('--seed', type=int, default=0)

    ap.add_argument('--max_length', type=int, default=128)

    ap.add_argument('--max_pairs_winobias', type=int, default=0, help='0 means all (ranking)')
    ap.add_argument('--max_pairs_stereoset', type=int, default=0, help='0 means all (ranking)')
    ap.add_argument('--eval_max_pairs_winobias', type=int, default=0, help='0 means all (evaluation only)')
    ap.add_argument('--eval_max_pairs_stereoset', type=int, default=800, help='0 means all (evaluation only)')
    ap.add_argument('--utility_max_texts', type=int, default=2000)
    ap.add_argument('--utility_batch_size', type=int, default=8, help='Batch size for MLM utility evaluation')
    ap.add_argument('--utility_datasets', nargs='+', default=['wikitext2'],
                    help='Utility corpora to evaluate (first is primary). Choices: wikitext2, wikitext103. Note: wt2 and wt103 share the same test split, so adding wt103 yields identical numbers.')
    ap.add_argument('--k_values', nargs='+', type=int, default=[5, 10, 15])
    ap.add_argument('--winobias_types', nargs='+', default=['type_1', 'type_2'],
                    help='WinoBias subset types to include (type_1 knowledge, type_2 syntactic)')
    ap.add_argument('--util_curv_max_texts', type=int, default=500,
                    help='Number of wikitext-2 train texts used to compute the utility-curvature signal for FairEtell. 0 disables the bias-specificity term.')
    ap.add_argument('--fairetell_n_dirs', type=int, default=4,
                    help='Number of bias directions per head used by FairEtell subspace ablation (top-k SVD of paired activation differences). Set to 0 to fall back to whole-head zeroing for FairEtell as well.')
    ap.add_argument('--fairetell_dir_max_pairs', type=int, default=400,
                    help='Number of WinoBias pairs used to estimate per-head bias subspaces.')

    args = ap.parse_args()

    torch.manual_seed(args.seed)

    device = args.device
    if device.startswith('cuda') and not torch.cuda.is_available():
        device = 'cpu'

    run_root = Path(args.outdir) / args.run_id
    run_root.mkdir(parents=True, exist_ok=True)

    config_path = run_root / 'config.json'
    with config_path.open('w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=2)

    global_rows: List[Dict] = []
    per_model_ablation_rows: Dict[str, List[Dict]] = {}

    for model_name in args.models:
        print(f"\n==== Model: {model_name} ====")
        model_dir = run_root / _safe_name(model_name)
        (model_dir / 'probes').mkdir(parents=True, exist_ok=True)
        (model_dir / 'rankings').mkdir(parents=True, exist_ok=True)
        (model_dir / 'eval').mkdir(parents=True, exist_ok=True)
        (model_dir / 'analysis').mkdir(parents=True, exist_ok=True)
        (model_dir / 'plots').mkdir(parents=True, exist_ok=True)

        loaded = load_model(model_name, device=device)

        # --- Build probes
        winobias_pairs = build_winobias_pairs(split='test', max_pairs=0, type_filter=tuple(args.winobias_types))
        stereoset_pairs = build_stereoset_pairs(config='intrasentence', split='validation', bias_types=None, max_pairs=0)

        write_pairs_csv(winobias_pairs, str(model_dir / 'probes' / 'probes_winobias.csv'))
        write_pairs_csv(stereoset_pairs, str(model_dir / 'probes' / 'probes_stereoset.csv'))

        # --- Utility-curvature signal (used by FairEtell to penalize heads that
        # are also important for clean-text language modeling). Computed once
        # per model from a wikitext-2 sample.
        util_curv = None
        if args.util_curv_max_texts and args.util_curv_max_texts > 0:
            util_curv = compute_utility_head_curvature(
                loaded,
                device=device,
                max_length=args.max_length,
                max_texts=args.util_curv_max_texts,
                seed=args.seed,
            )

        # --- Rankings (WinoBias)
        wb_rank = compute_rankings(
            loaded,
            probe_name='winobias',
            pairs=winobias_pairs,
            device=device,
            max_length=args.max_length,
            max_pairs=args.max_pairs_winobias,
            util_curvature=util_curv,
        )
        wb_path = model_dir / 'rankings' / 'winobias_rankings.json'
        save_rankings_json(wb_rank, str(wb_path))

        # --- Rankings (StereoSet)
        ss_rank = compute_rankings(
            loaded,
            probe_name='stereoset',
            pairs=stereoset_pairs,
            device=device,
            max_length=args.max_length,
            max_pairs=args.max_pairs_stereoset,
            util_curvature=util_curv,
        )
        ss_path = model_dir / 'rankings' / 'stereoset_rankings.json'
        save_rankings_json(ss_rank, str(ss_path))

        # --- Ranking divergence + stability
        wb_stats = compute_ranking_stats(wb_rank.rankings, k=20)
        ss_stats = compute_ranking_stats(ss_rank.rankings, k=20)
        stability = compute_cross_benchmark_stability(wb_rank.rankings, ss_rank.rankings, k=20)

        write_json(str(model_dir / 'analysis' / 'ranking_stats_winobias.json'), wb_stats)
        write_json(str(model_dir / 'analysis' / 'ranking_stats_stereoset.json'), ss_stats)
        write_json(str(model_dir / 'analysis' / 'stability_top20.json'), stability)

        plot_overlap_bar(wb_stats, str(model_dir / 'plots' / 'ranking_overlap_winobias.png'))
        plot_overlap_bar(ss_stats, str(model_dir / 'plots' / 'ranking_overlap_stereoset.png'))
        plot_stability_bar(stability, k=20, out_path=str(model_dir / 'plots' / 'stability_top20.png'))

        # Spearman correlation matrix between methods (full ranking, not just top-k)
        plot_spearman_matrix(wb_rank.rankings, str(model_dir / 'plots' / 'spearman_matrix_winobias.png'))
        plot_spearman_matrix(ss_rank.rankings, str(model_dir / 'plots' / 'spearman_matrix_stereoset.png'))

        # Sorted-score distribution (concentration of importance across heads)
        plot_score_distribution(wb_rank.rankings, str(model_dir / 'plots' / 'score_distribution_winobias.png'))
        plot_score_distribution(ss_rank.rankings, str(model_dir / 'plots' / 'score_distribution_stereoset.png'))

        # Per-pair gap-vs-score scatter for the top FairEtell head (mechanistic justification)
        if wb_rank.per_sample_grad is not None and wb_rank.head_names is not None:
            top_head_name = wb_rank.rankings['FairEtell'][0][0]
            try:
                top_idx = wb_rank.head_names.index(top_head_name)
                plot_gap_vs_score_scatter(
                    wb_rank.per_sample_grad,
                    wb_rank.per_sample_gap,
                    wb_rank.head_names,
                    top_idx,
                    str(model_dir / 'plots' / 'gap_vs_score_top_winobias.png'),
                )
            except ValueError:
                pass
        if ss_rank.per_sample_grad is not None and ss_rank.head_names is not None:
            top_head_name = ss_rank.rankings['FairEtell'][0][0]
            try:
                top_idx = ss_rank.head_names.index(top_head_name)
                plot_gap_vs_score_scatter(
                    ss_rank.per_sample_grad,
                    ss_rank.per_sample_gap,
                    ss_rank.head_names,
                    top_idx,
                    str(model_dir / 'plots' / 'gap_vs_score_top_stereoset.png'),
                )
            except ValueError:
                pass

        # --- Head heatmaps (scores reshaped to layers x heads)
        n_layers, n_heads = num_layers_heads(loaded.model)
        for m in ["Gradient", "Curvature", "FairEtell"]:
            plot_head_heatmap(wb_rank.rankings[m], n_layers, n_heads, str(model_dir / "plots" / f"heatmap_winobias_{m}.png"))
            plot_head_heatmap(ss_rank.rankings[m], n_layers, n_heads, str(model_dir / "plots" / f"heatmap_stereoset_{m}.png"))

        # --- Ablation evaluation (rankings derived from WinoBias)
        ablation = evaluate_ablation(
            loaded,
            winobias_pairs=winobias_pairs,
            stereoset_pairs=stereoset_pairs,
            rankings_winobias=wb_rank.rankings,
            device=device,
            max_length=args.max_length,
            k_values=list(args.k_values),
            eval_max_pairs_winobias=args.eval_max_pairs_winobias,
            eval_max_pairs_stereoset=args.eval_max_pairs_stereoset,
            utility_max_texts=args.utility_max_texts,
            utility_batch_size=args.utility_batch_size,
            seed=args.seed,
            utility_datasets=list(args.utility_datasets),
            fairetell_n_dirs=args.fairetell_n_dirs,
            fairetell_dir_max_pairs=args.fairetell_dir_max_pairs,
        )
        save_json(ablation, str(model_dir / 'eval' / 'ablation_results.json'))

        rows = ablation['results']
        plot_ablation_lines(rows, 'winobias_gap', str(model_dir / 'plots' / 'ablation_wb_gap.png'), ylabel='WinoBias gap')
        plot_ablation_lines(rows, 'stereoset_ss', str(model_dir / 'plots' / 'ablation_stereoset_ss.png'), ylabel='StereoSet stereotype score')
        plot_ablation_lines(rows, 'utility_ppl', str(model_dir / 'plots' / 'ablation_utility.png'), ylabel='Language-modeling loss')
        # Per-utility-dataset ablation curves
        for ds_key in ablation.get('utility_datasets', []):
            ppl_key = f'utility_{ds_key}_ppl'
            if any(ppl_key in r for r in rows):
                plot_ablation_lines(
                    rows, ppl_key, str(model_dir / 'plots' / f'ablation_utility_{ds_key}.png'),
                    ylabel='Language-modeling loss', ci_key=f'utility_{ds_key}_ci',
                )

        # Tradeoff (fairness vs utility) plots
        plot_tradeoff_scatter(
            rows,
            fairness_key="winobias_gap",
            utility_key="utility_ppl",
            out_path=str(model_dir / "plots" / "tradeoff_wb_gap_vs_utility.png"),
            xlabel="WinoBias gap",
            ylabel="Language-modeling loss",
        )
        plot_tradeoff_scatter(
            rows,
            fairness_key="stereoset_ss",
            utility_key="utility_ppl",
            out_path=str(model_dir / "plots" / "tradeoff_stereoset_ss_vs_utility.png"),
            xlabel="StereoSet stereotype score",
            ylabel="Language-modeling loss",
        )

        # Latency plots (paper-grade: per-method overhead vs baseline)
        plot_latency_lines(rows, str(model_dir / "plots" / "latency_per_method.png"))
        plot_latency_overhead_bar(rows, str(model_dir / "plots" / "latency_overhead_pct.png"))
        plot_fairness_latency_tradeoff(
            rows, fairness_key="winobias_gap",
            out_path=str(model_dir / "plots" / "tradeoff_wb_gap_vs_latency.png"),
            ylabel="WinoBias gap",
        )
        plot_fairness_latency_tradeoff(
            rows, fairness_key="stereoset_ss",
            out_path=str(model_dir / "plots" / "tradeoff_stereoset_ss_vs_latency.png"),
            ylabel="StereoSet stereotype score",
        )


        per_model_ablation_rows[model_name] = rows

        # --- Local summary row (baseline + a few ranking stats)
        baseline = [r for r in rows if r['method'] == 'Baseline' and r['k'] == 0][0]
        global_rows.append(
            {
                'model_name': model_name,
                'kind': loaded.kind,
                'wb_gap_baseline': baseline['winobias_gap'],
                'ss_baseline': baseline['stereoset_ss'],
                'utility_baseline': baseline['utility_ppl'],
                'wb_overlap_GvC': wb_stats['overlap_topk']['Gradient_vs_Curvature'],
                'wb_overlap_GvF': wb_stats['overlap_topk']['Gradient_vs_FairEtell'],
                'wb_overlap_CvF': wb_stats['overlap_topk']['Curvature_vs_FairEtell'],
                'stability_top20_Gradient': stability['Gradient'],
                'stability_top20_Curvature': stability['Curvature'],
                'stability_top20_FairEtell': stability['FairEtell'],
            }
        )

        # Also write per-model summary files
        write_json(str(model_dir / 'analysis' / 'summary.json'), global_rows[-1])

    # Global summary
    write_json(str(run_root / 'GLOBAL_SUMMARY.json'), {'rows': global_rows})
    write_csv(str(run_root / 'GLOBAL_SUMMARY.csv'), global_rows)

    # Cross-model comparison plots (FairEtell on every model, one figure per metric)
    if len(per_model_ablation_rows) > 1:
        global_plot_dir = run_root / 'plots'
        global_plot_dir.mkdir(parents=True, exist_ok=True)
        plot_cross_model_ablation(
            per_model_ablation_rows, 'winobias_gap',
            str(global_plot_dir / 'cross_model_wb_gap_ROBIN.png'),
            method='FairEtell', ylabel='WinoBias gap',
        )
        plot_cross_model_ablation(
            per_model_ablation_rows, 'stereoset_ss',
            str(global_plot_dir / 'cross_model_stereoset_ss_ROBIN.png'),
            method='FairEtell', ylabel='StereoSet stereotype score',
        )
        plot_cross_model_ablation(
            per_model_ablation_rows, 'utility_ppl',
            str(global_plot_dir / 'cross_model_utility_ROBIN.png'),
            method='FairEtell', ylabel='Language-modeling loss',
        )

    print("\n==== DONE ====")
    print(f"Run directory: {run_root}")


if __name__ == '__main__':
    main()
