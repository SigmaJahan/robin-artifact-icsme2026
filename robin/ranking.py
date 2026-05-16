from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy.stats import spearmanr
from tqdm import tqdm

from .modeling import LoadedHFModel, num_layers_heads
from .types import ProbePair


def compute_utility_head_curvature(
    loaded: LoadedHFModel,
    device: str,
    max_length: int = 128,
    max_texts: int = 500,
    seed: int = 0,
) -> np.ndarray:
    """Per-head Fisher-diag (squared gradient norm of attention weights) on
    a clean wikitext-2 sample. Used by FairEtell as a 'general utility'
    importance signal so we can down-weight heads that are not bias-specific.
    """
    from datasets import load_dataset
    from transformers import DataCollatorForLanguageModeling

    model = loaded.model
    tok = loaded.tokenizer
    n_layers, n_heads = num_layers_heads(model)
    n_total = n_layers * n_heads

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [t for t in ds["text"] if isinstance(t, str) and t.strip()]
    if max_texts and max_texts > 0:
        texts = texts[:max_texts]

    sum_g2 = np.zeros((n_total,), dtype=np.float64)
    n_seen = 0

    if loaded.kind == "mlm":
        torch.manual_seed(seed)
        collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=True, mlm_probability=0.15)
        encodings = tok(texts, add_special_tokens=True, truncation=True, max_length=max_length)
        ids_list = encodings["input_ids"]
        for i in tqdm(range(len(ids_list)), desc=f"util_curv:{loaded.name}", ncols=100):
            batch = collator([{"input_ids": ids_list[i]}])
            in_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            am = (in_ids != tok.pad_token_id).long() if tok.pad_token_id is not None else None
            model.zero_grad(set_to_none=True)
            out = model(input_ids=in_ids, attention_mask=am, labels=labels, output_attentions=True)
            if out.attentions is None:
                continue
            for t in out.attentions:
                if t is not None:
                    t.retain_grad()
            out.loss.backward()
            sum_g2 += _head_grad_norms(out.attentions) ** 2
            n_seen += 1
    else:
        eos = tok.eos_token_id
        for t in tqdm(texts, desc=f"util_curv:{loaded.name}", ncols=100):
            ids = tok(t, add_special_tokens=False, truncation=True, max_length=max_length)["input_ids"]
            if eos is not None:
                ids = ids + [eos]
            if len(ids) < 2:
                continue
            in_ids = torch.tensor([ids], dtype=torch.long, device=device)
            model.zero_grad(set_to_none=True)
            out = model(input_ids=in_ids, labels=in_ids, output_attentions=True, use_cache=False)
            if out.attentions is None:
                continue
            for layer_attn in out.attentions:
                if layer_attn is not None:
                    layer_attn.retain_grad()
            out.loss.backward()
            sum_g2 += _head_grad_norms(out.attentions) ** 2
            n_seen += 1

    if n_seen == 0:
        return np.zeros((n_total,), dtype=np.float64)
    return sum_g2 / n_seen


def _diff_indices(a: List[int], b: List[int]) -> Tuple[List[int], List[int]]:
    """Return indices that differ between two token sequences (aligned)."""
    import difflib

    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    idx_a = []
    idx_b = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        idx_a.extend(range(i1, i2))
        idx_b.extend(range(j1, j2))
    # uniq + sorted
    return sorted(set(idx_a)), sorted(set(idx_b))


def _special_token_count(tokenizer) -> int:
    try:
        return int(tokenizer.num_special_tokens_to_add(pair=False))
    except Exception:
        try:
            return len(_build_with_special_tokens(tokenizer, []))
        except Exception:
            return 0


def _build_with_special_tokens(tokenizer, base_ids: List[int]) -> List[int]:
    try:
        return tokenizer.build_inputs_with_special_tokens(base_ids)
    except AttributeError:
        pass
    try:
        return tokenizer.prepare_for_model(
            base_ids,
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=False,
        )["input_ids"]
    except AttributeError:
        pass
    if getattr(tokenizer, "cls_token_id", None) is not None and getattr(tokenizer, "sep_token_id", None) is not None:
        return [tokenizer.cls_token_id] + list(base_ids) + [tokenizer.sep_token_id]
    if getattr(tokenizer, "bos_token_id", None) is not None and getattr(tokenizer, "eos_token_id", None) is not None:
        return [tokenizer.bos_token_id] + list(base_ids) + [tokenizer.eos_token_id]
    return list(base_ids)


def _build_mlm_tensors(tokenizer, text: str, diff_token_positions: List[int], max_length: int):
    special = _special_token_count(tokenizer)
    base_max = max(1, max_length - special)

    enc = tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=base_max,
    )
    base_ids: List[int] = enc["input_ids"]

    built = _build_with_special_tokens(tokenizer, base_ids)

    # Infer offset of base_ids inside built.
    offset = 0
    if len(built) >= len(base_ids) and built[: len(base_ids)] == base_ids:
        offset = 0
    elif len(built) >= len(base_ids) + 1 and built[1 : 1 + len(base_ids)] == base_ids:
        offset = 1

    labels = [-100] * len(built)
    masked = list(built)

    if tokenizer.mask_token_id is None:
        raise ValueError("Tokenizer has no mask_token_id; MLM scoring not supported")

    for i in diff_token_positions:
        if i < 0 or i >= len(base_ids):
            continue
        pos = offset + i
        if pos <= 0 or pos >= len(built) - 1:
            # avoid masking special tokens
            continue
        labels[pos] = built[pos]
        masked[pos] = tokenizer.mask_token_id

    # Pad to max_length
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = 0

    attn = [1] * len(masked)
    if len(masked) > max_length:
        masked = masked[:max_length]
        labels = labels[:max_length]
        attn = attn[:max_length]
    elif len(masked) < max_length:
        pad_len = max_length - len(masked)
        masked = masked + [pad_id] * pad_len
        labels = labels + [-100] * pad_len
        attn = attn + [0] * pad_len

    input_ids = torch.tensor([masked], dtype=torch.long)
    attention_mask = torch.tensor([attn], dtype=torch.long)
    labels_t = torch.tensor([labels], dtype=torch.long)

    n_masked = int(sum(1 for x in labels if x != -100))
    return input_ids, attention_mask, labels_t, n_masked


def _build_clm_tensors(tokenizer, text: str, max_length: int):
    enc = tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )
    ids: List[int] = enc["input_ids"]

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    attn = [1] * len(ids)
    if len(ids) > max_length:
        ids = ids[:max_length]
        attn = attn[:max_length]
    elif len(ids) < max_length:
        pad_len = max_length - len(ids)
        ids = ids + [pad_id] * pad_len
        attn = attn + [0] * pad_len

    input_ids = torch.tensor([ids], dtype=torch.long)
    attention_mask = torch.tensor([attn], dtype=torch.long)
    orig_len = int(sum(attn))
    return input_ids, attention_mask, orig_len


def _head_grad_norms(attentions) -> np.ndarray:
    """Return flattened per-head gradient norms as numpy array."""
    norms: List[float] = []
    for layer_attn in attentions:
        if layer_attn is None:
            continue
        grad = getattr(layer_attn, "grad", None)
        if grad is None:
            # no grad -> zeros
            b, h, s1, s2 = layer_attn.shape
            norms.extend([0.0] * int(h))
            continue
        # grad: (B, H, S, S)
        g2 = grad.detach().float().pow(2).sum(dim=(0, 2, 3))
        g = torch.sqrt(g2 + 1e-12)
        norms.extend(g.cpu().tolist())
    return np.asarray(norms, dtype=np.float64)


@dataclass
class RankingOutput:
    model_name: str
    kind: str
    probe_name: str
    num_pairs: int
    max_length: int
    rankings: Dict[str, List[Tuple[str, float]]]
    per_sample_grad: np.ndarray = None
    per_sample_gap: np.ndarray = None
    head_names: List[str] = None
    score_correlations: Dict[str, np.ndarray] = None


def compute_rankings(
    loaded: LoadedHFModel,
    probe_name: str,
    pairs: List[ProbePair],
    device: str,
    max_length: int = 128,
    max_pairs: int = 0,
    util_curvature: np.ndarray = None,
) -> RankingOutput:
    """Compute Gradient, Curvature, and FairEtell rankings for a model on a probe set."""

    model = loaded.model
    tok = loaded.tokenizer

    n_layers, n_heads = num_layers_heads(model)
    head_names = [f"L{l}H{h}" for l in range(n_layers) for h in range(n_heads)]
    n_total = len(head_names)

    sum_g = np.zeros((n_total,), dtype=np.float64)
    sum_g2 = np.zeros((n_total,), dtype=np.float64)

    sample_g: List[np.ndarray] = []
    sample_gap: List[float] = []

    use_pairs = pairs[:max_pairs] if (max_pairs and max_pairs > 0) else pairs

    pbar = tqdm(use_pairs, desc=f"rank:{loaded.name}:{probe_name}", ncols=100)

    for pair in pbar:
        # Tokenize without special tokens for diffing.
        special = _special_token_count(tok)
        base_max = max(1, max_length - special) if loaded.kind == "mlm" else max_length

        ids_a = tok(pair.text_a, add_special_tokens=False, truncation=True, max_length=base_max)["input_ids"]
        ids_b = tok(pair.text_b, add_special_tokens=False, truncation=True, max_length=base_max)["input_ids"]

        diff_a, diff_b = _diff_indices(ids_a, ids_b)
        if len(diff_a) == 0 and len(diff_b) == 0:
            continue

        # --- text_a pass ---
        model.zero_grad(set_to_none=True)
        if loaded.kind == "mlm":
            in_a, am_a, lab_a, n_mask_a = _build_mlm_tensors(tok, pair.text_a, diff_a, max_length)
            if n_mask_a == 0:
                continue
            in_a = in_a.to(device)
            am_a = am_a.to(device)
            lab_a = lab_a.to(device)
            out_a = model(input_ids=in_a, attention_mask=am_a, labels=lab_a, output_attentions=True)
            # geometric mean prob over masked tokens
            score_a_prob = float(torch.exp(-out_a.loss.detach()).cpu().item())
            if out_a.attentions is None:
                raise RuntimeError("Model did not return attentions; cannot compute head rankings")
            for t in out_a.attentions:
                if t is not None:
                    t.retain_grad()
            out_a.loss.backward()
            g_a = _head_grad_norms(out_a.attentions)
        else:
            in_a, am_a, orig_len_a = _build_clm_tensors(tok, pair.text_a, max_length)
            in_a = in_a.to(device)
            am_a = am_a.to(device)
            out_a = model(input_ids=in_a, attention_mask=am_a, output_attentions=True, use_cache=False)
            if out_a.attentions is None:
                raise RuntimeError("Model did not return attentions; cannot compute head rankings")
            for t in out_a.attentions:
                if t is not None:
                    t.retain_grad()

            # diff indices that are within the original (unpadded) length
            idxs = [i for i in diff_a if 0 < i < orig_len_a]
            if len(idxs) == 0:
                continue
            idxs_t = torch.tensor(idxs, device=device, dtype=torch.long)
            logits = out_a.logits  # (1, L, V)
            logp = torch.log_softmax(logits[:, :-1, :], dim=-1)  # predict token i from position i-1
            target_ids = in_a[0, idxs_t]
            picked = logp[0, idxs_t - 1, :].gather(1, target_ids.unsqueeze(1)).squeeze(1)
            loss_a = -picked.mean()
            score_a_prob = float(torch.exp(-loss_a.detach()).cpu().item())
            loss_a.backward()
            g_a = _head_grad_norms(out_a.attentions)

        # --- text_b pass ---
        model.zero_grad(set_to_none=True)
        if loaded.kind == "mlm":
            in_b, am_b, lab_b, n_mask_b = _build_mlm_tensors(tok, pair.text_b, diff_b, max_length)
            if n_mask_b == 0:
                continue
            in_b = in_b.to(device)
            am_b = am_b.to(device)
            lab_b = lab_b.to(device)
            out_b = model(input_ids=in_b, attention_mask=am_b, labels=lab_b, output_attentions=True)
            score_b_prob = float(torch.exp(-out_b.loss.detach()).cpu().item())
            if out_b.attentions is None:
                raise RuntimeError("Model did not return attentions; cannot compute head rankings")
            for t in out_b.attentions:
                if t is not None:
                    t.retain_grad()
            out_b.loss.backward()
            g_b = _head_grad_norms(out_b.attentions)
        else:
            in_b, am_b, orig_len_b = _build_clm_tensors(tok, pair.text_b, max_length)
            in_b = in_b.to(device)
            am_b = am_b.to(device)
            out_b = model(input_ids=in_b, attention_mask=am_b, output_attentions=True, use_cache=False)
            if out_b.attentions is None:
                raise RuntimeError("Model did not return attentions; cannot compute head rankings")
            for t in out_b.attentions:
                if t is not None:
                    t.retain_grad()

            idxs = [i for i in diff_b if 0 < i < orig_len_b]
            if len(idxs) == 0:
                continue
            idxs_t = torch.tensor(idxs, device=device, dtype=torch.long)
            logits = out_b.logits
            logp = torch.log_softmax(logits[:, :-1, :], dim=-1)
            target_ids = in_b[0, idxs_t]
            picked = logp[0, idxs_t - 1, :].gather(1, target_ids.unsqueeze(1)).squeeze(1)
            loss_b = -picked.mean()
            score_b_prob = float(torch.exp(-loss_b.detach()).cpu().item())
            loss_b.backward()
            g_b = _head_grad_norms(out_b.attentions)

        if g_a.shape[0] != n_total or g_b.shape[0] != n_total:
            raise RuntimeError(
                f"Unexpected head count: got {g_a.shape[0]} / {g_b.shape[0]} but expected {n_total}"
            )

        gap = abs(score_a_prob - score_b_prob)

        # Accumulate scores
        sum_g += g_a + g_b
        sum_g2 += (g_a ** 2) + (g_b ** 2)
        sample_g.append(g_a)
        sample_gap.append(gap)
        sample_g.append(g_b)
        sample_gap.append(gap)

    n_samples = len(sample_gap)
    if n_samples == 0:
        raise RuntimeError("No usable probe pairs were processed (n_samples=0)")

    grad_mean = sum_g / n_samples
    fisher_diag = sum_g2 / n_samples

    g_mat = np.stack(sample_g, axis=0)  # (n_samples=2*N_pairs, n_heads); even=text_a, odd=text_b
    gaps = np.asarray(sample_gap, dtype=np.float64)

    # Spearman correlation kept for backward compatibility / diagnostics
    corr = np.zeros((n_total,), dtype=np.float64)
    for i in range(n_total):
        c, _ = spearmanr(g_mat[:, i], gaps)
        if np.isnan(c):
            c = 0.0
        corr[i] = c

    # Per-head paired gradient asymmetry: bias-causing heads should respond
    # asymmetrically to text_a vs text_b within each probe pair.
    g_a = g_mat[0::2]
    g_b = g_mat[1::2]
    asym = np.mean(np.abs(g_a - g_b), axis=0) if g_a.shape[0] == g_b.shape[0] else np.zeros(n_total)

    def _zscore(x: np.ndarray) -> np.ndarray:
        s = float(x.std())
        if s < 1e-12:
            return np.zeros_like(x)
        return (x - float(x.mean())) / s

    # util_curvature is captured for diagnostics but the FairEtell ranking
    # itself is the rank-fusion of gradient + bias-curvature. FairEtell's
    # differentiator vs Gradient/Curvature is the *ablation operator*
    # (subspace projection inside each head, see fairetell.modeling.apply_head_mask
    # with bias_dirs), not the ranking.
    if util_curvature is not None and util_curvature.shape == fisher_diag.shape:
        eps = float(np.median(util_curvature)) * 1e-3 + 1e-12
        spec_ratio = fisher_diag / (util_curvature + eps)
        spec_signal = np.log1p(spec_ratio)
    else:
        spec_signal = np.zeros(n_total, dtype=np.float64)

    fairetell = _zscore(grad_mean) + _zscore(fisher_diag)

    rankings: Dict[str, List[Tuple[str, float]]] = {}
    rankings["Gradient"] = sorted(zip(head_names, grad_mean.tolist()), key=lambda x: x[1], reverse=True)
    rankings["Curvature"] = sorted(zip(head_names, fisher_diag.tolist()), key=lambda x: x[1], reverse=True)
    rankings["FairEtell"] = sorted(zip(head_names, fairetell.tolist()), key=lambda x: x[1], reverse=True)

    score_correlations = {
        "Gradient": grad_mean,
        "Curvature": fisher_diag,
        "FairEtell": fairetell,
        "spearman_gap": corr,
        "paired_asym": asym,
        "util_curvature": util_curvature if util_curvature is not None else np.zeros(n_total),
        "spec_signal": spec_signal,
    }

    return RankingOutput(
        model_name=loaded.name,
        kind=loaded.kind,
        probe_name=probe_name,
        num_pairs=len(use_pairs),
        max_length=max_length,
        rankings=rankings,
        per_sample_grad=g_mat,
        per_sample_gap=gaps,
        head_names=list(head_names),
        score_correlations=score_correlations,
    )


def save_rankings_json(out: RankingOutput, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_name": out.model_name,
        "kind": out.kind,
        "probe_name": out.probe_name,
        "num_pairs": out.num_pairs,
        "max_length": out.max_length,
        "rankings": {
            k: [(h, float(s)) for (h, s) in v]
            for k, v in out.rankings.items()
        },
    }
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    if out.per_sample_grad is not None and out.per_sample_gap is not None:
        np.savez_compressed(
            str(p.with_suffix(".npz")),
            per_sample_grad=out.per_sample_grad,
            per_sample_gap=out.per_sample_gap,
            head_names=np.array(out.head_names, dtype=object),
            grad_mean=out.score_correlations["Gradient"],
            curvature=out.score_correlations["Curvature"],
            fairetell=out.score_correlations["FairEtell"],
            spearman_gap=out.score_correlations["spearman_gap"],
            paired_asym=out.score_correlations.get("paired_asym", np.zeros_like(out.score_correlations["Curvature"])),
            util_curvature=out.score_correlations.get("util_curvature", np.zeros_like(out.score_correlations["Curvature"])),
            spec_signal=out.score_correlations.get("spec_signal", np.zeros_like(out.score_correlations["Curvature"])),
        )
