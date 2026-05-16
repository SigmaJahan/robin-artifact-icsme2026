from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import DataCollatorForLanguageModeling

from .modeling import LoadedHFModel, apply_head_mask, build_head_mask
from .ranking import _build_clm_tensors, _build_mlm_tensors, _diff_indices, _special_token_count
from .types import ProbePair


def measure_inference_latency(
    loaded: LoadedHFModel,
    device: str,
    head_mask: Optional[torch.Tensor],
    bias_dirs: Optional[Dict] = None,
    batch_size: int = 8,
    seq_len: int = 128,
    warmup: int = 5,
    iters: int = 50,
    seed: int = 0,
) -> Dict[str, float]:
    """Measure forward-pass latency in ms on a fixed synthetic batch.

    Reports mean / p50 / p95 / std. Uses CUDA events for accurate GPU timing
    (falls back to perf_counter on CPU). Returns ms per forward.
    """
    import time

    model = loaded.model
    tok = loaded.tokenizer
    model.eval()

    torch.manual_seed(seed)
    vocab = getattr(tok, "vocab_size", None) or model.config.vocab_size
    input_ids = torch.randint(0, vocab, (batch_size, seq_len), dtype=torch.long, device=device)
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=device)
    use_cuda = (device == "cuda" and torch.cuda.is_available())

    def _forward():
        if loaded.kind == "clm":
            return model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        return model(input_ids=input_ids, attention_mask=attention_mask)

    times_ms: List[float] = []
    with torch.no_grad(), apply_head_mask(model, head_mask, bias_dirs=bias_dirs):
        for _ in range(warmup):
            _forward()
        if use_cuda:
            torch.cuda.synchronize()
        for _ in range(iters):
            if use_cuda:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                _forward()
                end.record()
                torch.cuda.synchronize()
                times_ms.append(start.elapsed_time(end))
            else:
                t0 = time.perf_counter()
                _forward()
                times_ms.append((time.perf_counter() - t0) * 1000.0)

    arr = np.asarray(times_ms, dtype=np.float64)
    return {
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.median(arr)),
        "p95_ms": float(np.quantile(arr, 0.95)),
        "std_ms": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "iters": int(arr.size),
        "batch_size": batch_size,
        "seq_len": seq_len,
    }


def bootstrap_mean_ci(values: List[float], seed: int = 0, n_boot: int = 500, alpha: float = 0.05) -> Tuple[float, Tuple[float, float]]:
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0, (0.0, 0.0)
    mean = float(arr.mean())
    boots = []
    n = arr.size
    for _ in range(n_boot):
        sample = rng.choice(arr, size=n, replace=True)
        boots.append(sample.mean())
    boots = np.sort(np.asarray(boots, dtype=np.float64))
    lo = float(np.quantile(boots, alpha / 2.0))
    hi = float(np.quantile(boots, 1.0 - alpha / 2.0))
    return mean, (lo, hi)


def bootstrap_exp_mean_loss_ci(values: List[float], seed: int = 0, n_boot: int = 500, alpha: float = 0.05) -> Tuple[float, Tuple[float, float]]:
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0, (0.0, 0.0)
    mean = float(np.exp(arr.mean()))
    boots = []
    n = arr.size
    for _ in range(n_boot):
        sample = rng.choice(arr, size=n, replace=True)
        boots.append(float(np.exp(sample.mean())))
    boots = np.sort(np.asarray(boots, dtype=np.float64))
    lo = float(np.quantile(boots, alpha / 2.0))
    hi = float(np.quantile(boots, 1.0 - alpha / 2.0))
    return mean, (lo, hi)


def bootstrap_weighted_ppl_ci(
    nlls: List[float],
    token_counts: List[int],
    seed: int = 0,
    n_boot: int = 500,
    alpha: float = 0.05,
) -> Tuple[float, Tuple[float, float]]:
    if not nlls or not token_counts:
        return 0.0, (0.0, 0.0)
    nll_arr = np.asarray(nlls, dtype=np.float64)
    count_arr = np.asarray(token_counts, dtype=np.float64)
    mean = float(np.exp(nll_arr.sum() / max(1.0, count_arr.sum())))
    rng = np.random.default_rng(seed)
    boots = []
    n = nll_arr.size
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        boots.append(float(np.exp(nll_arr[idx].sum() / max(1.0, count_arr[idx].sum()))))
    boots = np.sort(np.asarray(boots, dtype=np.float64))
    lo = float(np.quantile(boots, alpha / 2.0))
    hi = float(np.quantile(boots, 1.0 - alpha / 2.0))
    return mean, (lo, hi)


def _pair_prob_score(
    loaded: LoadedHFModel,
    text_a: str,
    text_b: str,
    head_mask: Optional[torch.Tensor],
    device: str,
    max_length: int,
    bias_dirs: Optional[Dict] = None,
) -> Tuple[float, float]:
    """Return (prob_a, prob_b) where prob is exp(mean logprob) over differing tokens."""
    model = loaded.model
    tok = loaded.tokenizer

    # Compute diff on truncated tokens
    if loaded.kind == "mlm":
        special = _special_token_count(tok)
        base_max = max(1, max_length - special)
    else:
        base_max = max_length

    ids_a = tok(text_a, add_special_tokens=False, truncation=True, max_length=base_max)["input_ids"]
    ids_b = tok(text_b, add_special_tokens=False, truncation=True, max_length=base_max)["input_ids"]
    diff_a, diff_b = _diff_indices(ids_a, ids_b)

    if loaded.kind == "mlm":
        in_a, am_a, lab_a, n_mask_a = _build_mlm_tensors(tok, text_a, diff_a, max_length)
        in_b, am_b, lab_b, n_mask_b = _build_mlm_tensors(tok, text_b, diff_b, max_length)
        if n_mask_a == 0 or n_mask_b == 0:
            return 0.0, 0.0

        in_a = in_a.to(device)
        am_a = am_a.to(device)
        lab_a = lab_a.to(device)
        in_b = in_b.to(device)
        am_b = am_b.to(device)
        lab_b = lab_b.to(device)

        with apply_head_mask(model, head_mask, bias_dirs=bias_dirs):
            out_a = model(input_ids=in_a, attention_mask=am_a, labels=lab_a)
            out_b = model(input_ids=in_b, attention_mask=am_b, labels=lab_b)

        prob_a = float(torch.exp(-out_a.loss.detach()).cpu().item())
        prob_b = float(torch.exp(-out_b.loss.detach()).cpu().item())
        return prob_a, prob_b

    # clm
    in_a, am_a, orig_len_a = _build_clm_tensors(tok, text_a, max_length)
    in_b, am_b, orig_len_b = _build_clm_tensors(tok, text_b, max_length)

    in_a = in_a.to(device)
    am_a = am_a.to(device)
    in_b = in_b.to(device)
    am_b = am_b.to(device)

    with apply_head_mask(model, head_mask, bias_dirs=bias_dirs):
        out_a = model(input_ids=in_a, attention_mask=am_a, use_cache=False)
        out_b = model(input_ids=in_b, attention_mask=am_b, use_cache=False)

    # score only differing token positions
    idxs_a = [i for i in diff_a if 0 < i < orig_len_a]
    idxs_b = [i for i in diff_b if 0 < i < orig_len_b]
    if len(idxs_a) == 0 or len(idxs_b) == 0:
        return 0.0, 0.0

    def prob_from_logits(logits, input_ids, idxs):
        idxs_t = torch.tensor(idxs, device=device, dtype=torch.long)
        logp = torch.log_softmax(logits[:, :-1, :], dim=-1)
        target_ids = input_ids[0, idxs_t]
        picked = logp[0, idxs_t - 1, :].gather(1, target_ids.unsqueeze(1)).squeeze(1)
        loss = -picked.mean()
        return float(torch.exp(-loss.detach()).cpu().item())

    return prob_from_logits(out_a.logits, in_a, idxs_a), prob_from_logits(out_b.logits, in_b, idxs_b)


@dataclass
class FairnessMetrics:
    winobias_gap_mean: float
    winobias_gap_ci: Tuple[float, float]
    stereoset_ss_mean: float
    stereoset_ss_ci: Tuple[float, float]


def evaluate_fairness(
    loaded: LoadedHFModel,
    winobias_pairs: List[ProbePair],
    stereoset_pairs: List[ProbePair],
    device: str,
    head_mask: Optional[torch.Tensor],
    max_length: int,
    eval_max_pairs_winobias: int = 0,
    eval_max_pairs_stereoset: int = 0,
    seed: int = 0,
    bias_dirs: Optional[Dict] = None,
) -> FairnessMetrics:
    model = loaded.model
    model.eval()

    # --- WinoBias gap: mean |p(a)-p(b)| * 100
    use_wb = winobias_pairs[:eval_max_pairs_winobias] if (eval_max_pairs_winobias and eval_max_pairs_winobias > 0) else winobias_pairs
    gaps: List[float] = []
    with torch.no_grad():
        for p in tqdm(use_wb, desc=f"eval_wb:{loaded.name}", ncols=100):
            pa, pb = _pair_prob_score(loaded, p.text_a, p.text_b, head_mask, device, max_length, bias_dirs=bias_dirs)
            gaps.append(abs(pa - pb) * 100.0)

    wb_mean, wb_ci = bootstrap_mean_ci(gaps, seed=seed)

    # --- StereoSet stereotype score: % of pairs where stereotype > anti
    use_ss = stereoset_pairs[:eval_max_pairs_stereoset] if (eval_max_pairs_stereoset and eval_max_pairs_stereoset > 0) else stereoset_pairs
    flags: List[float] = []
    with torch.no_grad():
        for p in tqdm(use_ss, desc=f"eval_ss:{loaded.name}", ncols=100):
            pa, pb = _pair_prob_score(loaded, p.text_a, p.text_b, head_mask, device, max_length, bias_dirs=bias_dirs)
            flags.append(1.0 if pa > pb else 0.0)

    ss_values = [x * 100.0 for x in flags]
    ss_mean, ss_ci = bootstrap_mean_ci(ss_values, seed=seed)

    return FairnessMetrics(
        winobias_gap_mean=wb_mean,
        winobias_gap_ci=wb_ci,
        stereoset_ss_mean=ss_mean,
        stereoset_ss_ci=ss_ci,
    )


@dataclass
class UtilityMetrics:
    utility_ppl: float
    utility_ci: Tuple[float, float]
    utility_metric: str


UTILITY_DATASETS: Dict[str, Dict] = {
    "wikitext2": {
        "load": ("wikitext", "wikitext-2-raw-v1", "test"),
        "text_field": "text",
        "label_clm": "wikitext2_ppl",
        "label_mlm": "wikitext2_mlm_ppl_proxy",
    },
    "wikitext103": {
        "load": ("wikitext", "wikitext-103-raw-v1", "test"),
        "text_field": "text",
        "label_clm": "wikitext103_ppl",
        "label_mlm": "wikitext103_mlm_ppl_proxy",
    },
}


def _load_utility_texts(dataset_key: str) -> Tuple[List[str], Dict]:
    if dataset_key not in UTILITY_DATASETS:
        raise ValueError(f"Unknown utility dataset: {dataset_key}")
    spec = UTILITY_DATASETS[dataset_key]
    name, config, split = spec["load"]
    if config is None:
        ds = load_dataset(name, split=split)
    else:
        ds = load_dataset(name, config, split=split)
    field = spec["text_field"]
    texts = [t for t in ds[field] if isinstance(t, str) and t.strip()]
    return texts, spec


def evaluate_utility(
    loaded: LoadedHFModel,
    device: str,
    head_mask: Optional[torch.Tensor],
    max_length: int,
    utility_max_texts: int = 2000,
    batch_size: int = 8,
    seed: int = 0,
    dataset_key: str = "wikitext2",
    bias_dirs: Optional[Dict] = None,
) -> UtilityMetrics:
    """Utility metric on a generic LM corpus.

    Supported corpora: wikitext-2, PTB (ptb_text_only). Both are evaluated as:
    - clm: standard perplexity (exp(cross-entropy))
    - mlm: masked-token perplexity proxy (exp(masked loss))
    """
    model = loaded.model
    tok = loaded.tokenizer
    model.eval()

    texts, spec = _load_utility_texts(dataset_key)
    if utility_max_texts and utility_max_texts > 0:
        texts = texts[:utility_max_texts]

    label_clm = spec["label_clm"]
    label_mlm = spec["label_mlm"]

    if loaded.kind == "clm":
        # Concatenate all token ids for efficiency
        all_ids: List[int] = []
        eos = tok.eos_token_id if tok.eos_token_id is not None else None
        for t in texts:
            ids = tok(t, add_special_tokens=False)["input_ids"]
            if eos is not None:
                ids = ids + [eos]
            all_ids.extend(ids)

        if len(all_ids) < 2:
            return UtilityMetrics(utility_ppl=0.0, utility_ci=(0.0, 0.0), utility_metric=label_clm)

        # Build blocks
        block = max_length
        nlls: List[float] = []
        token_counts: List[int] = []

        with torch.no_grad(), apply_head_mask(model, head_mask, bias_dirs=bias_dirs):
            for i in tqdm(range(0, len(all_ids), block), desc=f"util_clm:{loaded.name}:{dataset_key}", ncols=100):
                chunk = all_ids[i : i + block]
                if len(chunk) < 2:
                    continue
                input_ids = torch.tensor([chunk], dtype=torch.long, device=device)
                labels = input_ids.clone()
                out = model(input_ids=input_ids, labels=labels, use_cache=False)
                # out.loss is mean over predicted tokens
                n_pred = int(input_ids.shape[1] - 1)
                nlls.append(float(out.loss.detach().cpu().item()) * n_pred)
                token_counts.append(n_pred)

        ppl, ci = bootstrap_weighted_ppl_ci(nlls, token_counts, seed=seed)
        return UtilityMetrics(utility_ppl=ppl, utility_ci=ci, utility_metric=label_clm)

    # mlm utility: masked-token perplexity proxy
    torch.manual_seed(seed)

    collator = DataCollatorForLanguageModeling(
        tokenizer=tok,
        mlm=True,
        mlm_probability=0.15,
    )

    # tokenize into fixed-length chunks
    encodings = tok(texts, add_special_tokens=True, truncation=True, max_length=max_length)
    input_ids_list = encodings["input_ids"]

    # simple batching
    masked_losses: List[float] = []

    with torch.no_grad(), apply_head_mask(model, head_mask, bias_dirs=bias_dirs):
        for i in tqdm(range(0, len(input_ids_list), batch_size), desc=f"util_mlm:{loaded.name}:{dataset_key}", ncols=100):
            batch = [{"input_ids": ids} for ids in input_ids_list[i : i + batch_size]]
            batch_t = collator(batch)
            input_ids = batch_t["input_ids"].to(device)
            labels = batch_t["labels"].to(device)
            attention_mask = (input_ids != tok.pad_token_id).long() if tok.pad_token_id is not None else None
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            masked_losses.append(float(out.loss.detach().cpu().item()))

    ppl_proxy, ci = bootstrap_exp_mean_loss_ci(masked_losses, seed=seed)
    return UtilityMetrics(utility_ppl=ppl_proxy, utility_ci=ci, utility_metric=label_mlm)


def evaluate_utility_wikitext2(
    loaded: LoadedHFModel,
    device: str,
    head_mask: Optional[torch.Tensor],
    max_length: int,
    utility_max_texts: int = 2000,
    batch_size: int = 8,
    seed: int = 0,
) -> UtilityMetrics:
    return evaluate_utility(
        loaded,
        device=device,
        head_mask=head_mask,
        max_length=max_length,
        utility_max_texts=utility_max_texts,
        batch_size=batch_size,
        seed=seed,
        dataset_key="wikitext2",
    )


@dataclass
class AblationRow:
    method: str
    k: int
    winobias_gap: float
    winobias_gap_ci: Tuple[float, float]
    stereoset_ss: float
    stereoset_ss_ci: Tuple[float, float]
    utility_ppl: float
    utility_ci: Tuple[float, float]
    utility_metric: str


def evaluate_ablation(
    loaded: LoadedHFModel,
    winobias_pairs: List[ProbePair],
    stereoset_pairs: List[ProbePair],
    rankings_winobias: Dict[str, List[Tuple[str, float]]],
    device: str,
    max_length: int,
    k_values: List[int],
    eval_max_pairs_winobias: int,
    eval_max_pairs_stereoset: int,
    utility_max_texts: int,
    utility_batch_size: int = 8,
    seed: int = 0,
    utility_datasets: Optional[List[str]] = None,
    fairetell_n_dirs: int = 2,
    fairetell_dir_max_pairs: int = 400,
) -> Dict:
    """Evaluate baseline + ablations using WinoBias-derived rankings."""

    if utility_datasets is None:
        utility_datasets = ["wikitext2"]
    primary_util = utility_datasets[0]

    def _run_utility(head_mask, bias_dirs=None):
        out: Dict[str, UtilityMetrics] = {}
        for ds_key in utility_datasets:
            out[ds_key] = evaluate_utility(
                loaded,
                device=device,
                head_mask=head_mask,
                max_length=max_length,
                utility_max_texts=utility_max_texts,
                batch_size=utility_batch_size,
                seed=seed,
                dataset_key=ds_key,
                bias_dirs=bias_dirs,
            )
        return out

    def _row(method: str, k: int, fairness, util_map: Dict[str, UtilityMetrics], latency: Optional[Dict] = None) -> Dict:
        primary = util_map[primary_util]
        row = {
            "method": method,
            "k": k,
            "winobias_gap": fairness.winobias_gap_mean,
            "winobias_gap_ci": fairness.winobias_gap_ci,
            "stereoset_ss": fairness.stereoset_ss_mean,
            "stereoset_ss_ci": fairness.stereoset_ss_ci,
            "utility_ppl": primary.utility_ppl,
            "utility_ci": primary.utility_ci,
            "utility_metric": primary.utility_metric,
        }
        for ds_key, um in util_map.items():
            row[f"utility_{ds_key}_ppl"] = um.utility_ppl
            row[f"utility_{ds_key}_ci"] = um.utility_ci
            row[f"utility_{ds_key}_metric"] = um.utility_metric
        if latency is not None:
            row["latency_mean_ms"] = latency["mean_ms"]
            row["latency_p50_ms"] = latency["p50_ms"]
            row["latency_p95_ms"] = latency["p95_ms"]
            row["latency_std_ms"] = latency["std_ms"]
            row["latency_iters"] = latency["iters"]
            row["latency_batch_size"] = latency["batch_size"]
            row["latency_seq_len"] = latency["seq_len"]
        return row

    out_rows: List[Dict] = []

    fairness = evaluate_fairness(
        loaded,
        winobias_pairs,
        stereoset_pairs,
        device=device,
        head_mask=None,
        max_length=max_length,
        eval_max_pairs_winobias=eval_max_pairs_winobias,
        eval_max_pairs_stereoset=eval_max_pairs_stereoset,
        seed=seed,
    )
    util_baseline = _run_utility(None)
    lat_baseline = measure_inference_latency(
        loaded, device=device, head_mask=None, bias_dirs=None,
        batch_size=8, seq_len=max_length, warmup=5, iters=50, seed=seed,
    )
    out_rows.append(_row("Baseline", 0, fairness, util_baseline, latency=lat_baseline))

    # Precompute bias directions once for the union of all FairEtell heads we'll
    # touch across k values. FairEtell's ablation operator is subspace projection
    # (top-n_dirs SVD of paired activation differences), not whole-head zeroing.
    fe_max_k = max(k_values)
    fe_top_heads = [name for name, _ in rankings_winobias["FairEtell"][:fe_max_k]]
    fe_target_heads = set()
    for hn in fe_top_heads:
        l, h = hn.replace("L", "").split("H")
        fe_target_heads.add((int(l), int(h)))
    if fairetell_n_dirs and fairetell_n_dirs > 0:
        from .modeling import compute_head_bias_directions
        fe_bias_dirs = compute_head_bias_directions(
            loaded, winobias_pairs, fe_target_heads,
            device=device, max_length=max_length,
            max_pairs=fairetell_dir_max_pairs, n_dirs=fairetell_n_dirs,
        )
    else:
        fe_bias_dirs = None

    for method in ["Gradient", "Curvature", "FairEtell"]:
        for k in k_values:
            if k <= 0:
                continue
            heads = [name for name, _ in rankings_winobias[method][:k]]
            heads_to_ablate = set()
            for hn in heads:
                l, h = hn.replace("L", "").split("H")
                heads_to_ablate.add((int(l), int(h)))

            head_mask = build_head_mask(loaded.model, heads_to_ablate, device=device)
            bias_dirs_for_method = fe_bias_dirs if method == "FairEtell" else None

            fairness_k = evaluate_fairness(
                loaded,
                winobias_pairs,
                stereoset_pairs,
                device=device,
                head_mask=head_mask,
                max_length=max_length,
                eval_max_pairs_winobias=eval_max_pairs_winobias,
                eval_max_pairs_stereoset=eval_max_pairs_stereoset,
                seed=seed,
                bias_dirs=bias_dirs_for_method,
            )
            util_k = _run_utility(head_mask, bias_dirs=bias_dirs_for_method)
            lat_k = measure_inference_latency(
                loaded, device=device, head_mask=head_mask, bias_dirs=bias_dirs_for_method,
                batch_size=8, seq_len=max_length, warmup=5, iters=50, seed=seed,
            )
            out_rows.append(_row(method, int(k), fairness_k, util_k, latency=lat_k))

    return {
        "model_name": loaded.name,
        "kind": loaded.kind,
        "rank_source": "winobias",
        "k_values": k_values,
        "utility_datasets": utility_datasets,
        "seed": seed,
        "results": out_rows,
    }


def save_json(obj: Dict, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
