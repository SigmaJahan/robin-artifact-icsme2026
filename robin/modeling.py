from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
)

ModelKind = Literal["mlm", "clm"]


@dataclass
class LoadedHFModel:
    name: str
    kind: ModelKind
    tokenizer: any
    model: torch.nn.Module


def _load_hf_model(model_cls, model_name: str):
    try:
        return model_cls.from_pretrained(model_name, attn_implementation="eager")
    except TypeError:
        return model_cls.from_pretrained(model_name)


def infer_kind(model_name: str) -> ModelKind:
    n = model_name.lower()
    if "gpt2" in n:
        return "clm"
    return "mlm"


def load_model(
    model_name: str,
    kind: Optional[ModelKind] = None,
    device: str = "cuda",
) -> LoadedHFModel:
    kind = kind or infer_kind(model_name)

    tok = AutoTokenizer.from_pretrained(model_name)

    if kind == "clm":
        model = _load_hf_model(AutoModelForCausalLM, model_name)
        # GPT2-style tokenizers often have no pad token by default.
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model.config.pad_token_id = tok.pad_token_id
        # Avoid KV-cache during scoring/grad passes.
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    else:
        model = _load_hf_model(AutoModelForMaskedLM, model_name)

    model.to(device)
    model.eval()  # deterministic (dropout off), gradients still enabled when needed

    return LoadedHFModel(name=model_name, kind=kind, tokenizer=tok, model=model)


def num_layers_heads(model) -> Tuple[int, int]:
    cfg = model.config

    n_layers = (
        getattr(cfg, "num_hidden_layers", None)
        or getattr(cfg, "n_layer", None)
        or getattr(cfg, "n_layers", None)
    )
    n_heads = (
        getattr(cfg, "num_attention_heads", None)
        or getattr(cfg, "n_head", None)
        or getattr(cfg, "n_heads", None)
    )

    if n_layers is None or n_heads is None:
        raise ValueError(
            f"Could not infer layers/heads from config: {cfg.__class__.__name__}"
        )

    return int(n_layers), int(n_heads)


def build_head_mask(model, heads_to_ablate: set[tuple[int, int]], device: str):
    """Create a head_mask tensor (num_layers, num_heads) with 0 for ablated heads.

    Note: in transformers >= 5.x the `head_mask` kwarg has been removed from
    DistilBERT / GPT-2 / BERT modeling code. Pass the returned mask to
    `apply_head_mask(model, mask)` (a context manager) instead — it installs
    forward pre-hooks on each layer's output projection that zero out the
    head-slice of the attention output before projection.
    """
    n_layers, n_heads = num_layers_heads(model)
    mask = torch.ones((n_layers, n_heads), dtype=torch.float32, device=device)
    for (l, h) in heads_to_ablate:
        if 0 <= l < n_layers and 0 <= h < n_heads:
            mask[l, h] = 0.0
    return mask


def _attention_layers(model) -> list:
    """Return a list of (layer_idx, attn_module) pairs, ordered by layer index.

    The attn_module exposes:
      - .out_proj_module : the linear/Conv1D applied to the concatenated head
        outputs of shape (batch, seq, num_heads * head_dim). Zeroing the
        head-slice of *its input* is equivalent to head_mask=0 for that head.
      - .num_heads, .head_dim
    """
    base = model.base_model if hasattr(model, "base_model") else model
    # DistilBERT path: base.transformer.layer[i].attention.{q_lin,k_lin,v_lin,out_lin}
    if hasattr(base, "transformer") and hasattr(base.transformer, "layer"):
        out = []
        for i, blk in enumerate(base.transformer.layer):
            attn = blk.attention
            attn._fairetell_out_proj = attn.out_lin
            attn._fairetell_num_heads = attn.n_heads
            attn._fairetell_head_dim = attn.attention_head_size
            out.append((i, attn))
        return out
    # GPT-2 path: base.h[i].attn.{c_attn, c_proj} ; num_heads/head_dim attributes already present
    if hasattr(base, "h"):
        out = []
        for i, blk in enumerate(base.h):
            attn = blk.attn
            attn._fairetell_out_proj = attn.c_proj
            attn._fairetell_num_heads = attn.num_heads
            attn._fairetell_head_dim = attn.head_dim
            out.append((i, attn))
        return out
    # BERT path: base.encoder.layer[i].attention.{self, output.dense}
    if hasattr(base, "encoder") and hasattr(base.encoder, "layer"):
        out = []
        for i, blk in enumerate(base.encoder.layer):
            attn_self = blk.attention.self
            attn_self._fairetell_out_proj = blk.attention.output.dense
            attn_self._fairetell_num_heads = attn_self.num_attention_heads
            attn_self._fairetell_head_dim = attn_self.attention_head_size
            out.append((i, attn_self))
        return out
    raise ValueError(
        f"Could not locate attention layers on {model.__class__.__name__}; "
        "supported backbones: DistilBERT, GPT-2, BERT."
    )


@contextmanager
def _capture_attn_outputs(model, layer_indices: set):
    """Context manager: registers forward hooks on each requested layer's
    attention output-projection input (the concatenated head outputs of shape
    (..., num_heads * head_dim)). Captures the LAST batch passed through.

    Yields a dict {layer_idx: tensor of shape (..., num_heads*head_dim)}.
    """
    layers = _attention_layers(model)
    captured: dict = {}
    handles = []
    try:
        for li, attn in layers:
            if li not in layer_indices:
                continue
            out_proj = attn._fairetell_out_proj

            def _make_hook(idx):
                def _hook(_mod, inputs):
                    captured[idx] = inputs[0].detach()
                return _hook

            handles.append(out_proj.register_forward_pre_hook(_make_hook(li)))
        yield captured
    finally:
        for h in handles:
            h.remove()


def compute_head_bias_directions(
    loaded, pairs, target_heads, device,
    max_length=128, max_pairs=400, n_dirs: int = 1,
):
    """For each (layer, head) in target_heads, compute a (n_dirs, head_dim) matrix
    of orthonormal bias directions in head-output space.

    Per pair, we capture the head's output activation at diff-token positions for
    text_a (stereotype) and text_b (anti-stereotype) and form the per-pair
    difference vector. Stacking those across pairs gives a (N_pairs, head_dim)
    matrix; the top-`n_dirs` right-singular vectors of that matrix are the
    bias subspace. n_dirs=1 reduces to the Bolukbasi mean-difference direction.

    Returns: dict (layer, head) -> torch.Tensor of shape (n_dirs, head_dim),
    rows are unit-norm and pairwise orthogonal.
    """
    import difflib

    model = loaded.model
    tok = loaded.tokenizer
    n_layers, n_heads = num_layers_heads(model)
    layers = _attention_layers(model)
    head_dims = {li: attn._fairetell_head_dim for li, attn in layers}

    target_heads = list(target_heads)
    target_layers = {l for (l, _) in target_heads}

    # collect per-pair difference vectors per (layer, head)
    diffs_per_head: dict = {(l, h): [] for (l, h) in target_heads}

    use_pairs = pairs[:max_pairs] if (max_pairs and max_pairs > 0) else pairs

    model.eval()
    with torch.no_grad():
        for pair in use_pairs:
            ids_a = tok(pair.text_a, add_special_tokens=False, truncation=True, max_length=max_length)["input_ids"]
            ids_b = tok(pair.text_b, add_special_tokens=False, truncation=True, max_length=max_length)["input_ids"]
            sm = difflib.SequenceMatcher(a=ids_a, b=ids_b, autojunk=False)
            diff_a, diff_b = [], []
            for tag, i1, i2, j1, j2 in sm.get_opcodes():
                if tag == "equal":
                    continue
                diff_a.extend(range(i1, i2))
                diff_b.extend(range(j1, j2))
            if not diff_a or not diff_b:
                continue

            cap_per_text: dict = {}
            for label, text, ids in [("a", pair.text_a, ids_a), ("b", pair.text_b, ids_b)]:
                enc = tok(text, return_tensors="pt", truncation=True, max_length=max_length)
                input_ids = enc["input_ids"].to(device)
                attention_mask = enc.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                with _capture_attn_outputs(model, target_layers) as cap:
                    if attention_mask is not None:
                        model(input_ids=input_ids, attention_mask=attention_mask)
                    else:
                        model(input_ids=input_ids)
                full = input_ids[0].tolist()
                offset = 0
                if len(full) >= len(ids) and full[:len(ids)] == ids:
                    offset = 0
                elif len(full) >= len(ids) + 1 and full[1:1 + len(ids)] == ids:
                    offset = 1
                cap_per_text[label] = (cap, offset, full, ids)

            cap_a, off_a, full_a, _ = cap_per_text["a"]
            cap_b, off_b, full_b, _ = cap_per_text["b"]
            pos_a = [off_a + i for i in diff_a if 0 <= off_a + i < len(full_a)]
            pos_b = [off_b + i for i in diff_b if 0 <= off_b + i < len(full_b)]
            if not pos_a or not pos_b:
                continue

            for (l, h) in target_heads:
                if l not in cap_a or l not in cap_b:
                    continue
                head_dim = head_dims[l]
                start = h * head_dim
                end = start + head_dim
                slab_a = cap_a[l][0, pos_a, start:end].mean(dim=0)
                slab_b = cap_b[l][0, pos_b, start:end].mean(dim=0)
                diffs_per_head[(l, h)].append((slab_a - slab_b).to(torch.float32).cpu())

    bias_dirs: dict = {}
    for (l, h) in target_heads:
        diffs = diffs_per_head[(l, h)]
        if not diffs:
            continue
        D = torch.stack(diffs, dim=0)  # (N, head_dim)
        if D.shape[0] < 1:
            continue
        # Top-n_dirs right singular vectors of D capture the dominant bias subspace.
        try:
            U, S, Vh = torch.linalg.svd(D, full_matrices=False)
        except Exception:
            # numerical fallback: use mean-diff direction
            v = D.mean(dim=0)
            nrm = float(v.norm())
            if nrm < 1e-8:
                continue
            bias_dirs[(l, h)] = (v / nrm).unsqueeze(0)
            continue
        k_use = min(n_dirs, Vh.shape[0])
        V = Vh[:k_use]  # (k_use, head_dim)
        # rows of V are orthonormal already (SVD); just be safe
        norms = V.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        bias_dirs[(l, h)] = V / norms
    return bias_dirs


@contextmanager
def apply_head_mask(model, head_mask: Optional[torch.Tensor], bias_dirs: "Optional[dict]" = None):
    """Temporarily ablate attention heads via forward pre-hooks on the output
    projection of each layer.

    `head_mask` is a (num_layers, num_heads) tensor: 0 means "ablate". Pass
    None for the unablated baseline (no hooks are installed).

    `bias_dirs` is an optional dict mapping (layer_idx, head_idx) -> 1-D
    torch.Tensor of length head_dim (unit norm). When provided, ablated heads
    have ONLY their bias direction projected out instead of being zeroed
    completely. Heads marked for ablation but missing from bias_dirs fall
    back to full-zero ablation.

    On exit, all hooks are removed even if the wrapped block raises.
    """
    if head_mask is None:
        yield
        return

    layers = _attention_layers(model)
    handles = []
    try:
        for layer_idx, attn in layers:
            mask_row = head_mask[layer_idx]  # shape (num_heads,)
            # Skip layers where every head is kept — saves hook overhead.
            if torch.all(mask_row > 0.5):
                continue
            n_heads = attn._fairetell_num_heads
            head_dim = attn._fairetell_head_dim
            out_proj = attn._fairetell_out_proj
            zero_heads = [h for h in range(n_heads) if float(mask_row[h].item()) == 0.0]
            if not zero_heads:
                continue

            # Partition into subspace-ablated vs zero-ablated heads. Each
            # subspace entry is a (k, head_dim) tensor of orthonormal rows.
            subspace_dirs: dict = {}
            full_zero: list = []
            for h in zero_heads:
                key = (layer_idx, h)
                if bias_dirs is not None and key in bias_dirs:
                    V = bias_dirs[key].detach()
                    if V.dim() == 1:
                        V = V.unsqueeze(0)
                    if V.dim() == 2 and V.shape[-1] == head_dim:
                        subspace_dirs[h] = V.to(dtype=torch.float32)
                        continue
                full_zero.append(h)

            def _make_hook(full_zero_local: list, subspace_dirs_local: dict, head_dim_local: int):
                def _hook(_module, inputs):
                    x = inputs[0]
                    x = x.clone()
                    for h in full_zero_local:
                        start = h * head_dim_local
                        end = start + head_dim_local
                        x[..., start:end] = 0
                    for h, V in subspace_dirs_local.items():
                        start = h * head_dim_local
                        end = start + head_dim_local
                        slab = x[..., start:end]  # (..., head_dim)
                        V_dev = V.to(device=slab.device, dtype=slab.dtype)  # (k, head_dim)
                        # Orthogonal projection onto the subspace and subtract:
                        # coefs = slab @ V^T  -> (..., k); proj = coefs @ V -> (..., head_dim)
                        coefs = torch.matmul(slab, V_dev.T)
                        proj = torch.matmul(coefs, V_dev)
                        x[..., start:end] = slab - proj
                    new_inputs = (x,) + tuple(inputs[1:])
                    return new_inputs
                return _hook

            handle = out_proj.register_forward_pre_hook(_make_hook(full_zero, subspace_dirs, head_dim))
            handles.append(handle)
        yield
    finally:
        for h in handles:
            h.remove()
