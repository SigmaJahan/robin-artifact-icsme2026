from __future__ import annotations

import csv
import re
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List, Optional

from datasets import load_dataset

from .types import ProbePair


_PRONOUN_SWAP = {
    "he": "she",
    "she": "he",
    "him": "her",
    "her": "him",
    "his": "her",
    "hers": "his",
    "himself": "herself",
    "herself": "himself",
}

_PRONOUN_RE = re.compile(
    r"\b(he|she|him|her|his|hers|himself|herself)\b",
    flags=re.IGNORECASE,
)


def _swap_pronouns(text: str) -> str:
    """Swap gendered pronouns using word-boundary regex replacement."""

    def repl(m: re.Match) -> str:
        w = m.group(0)
        swapped = _PRONOUN_SWAP.get(w.lower(), w)
        if w[:1].isupper():
            return swapped[:1].upper() + swapped[1:]
        return swapped

    return _PRONOUN_RE.sub(repl, text)


def build_winobias_pairs(
    split: str = "test",
    max_pairs: int = 0,
    type_filter: Optional[object] = ("type_1", "type_2"),
    polarity_filter: Optional[str] = None,
) -> List[ProbePair]:
    """Build WinoBias probe pairs by pronoun-swapping sentences.

    Source dataset: Elfsong/Wino_Bias

    type_filter accepts a single type string ("type_1") or an iterable of types
    (e.g. ("type_1", "type_2")). Default keeps both types so the ranking and
    eval sees the full WinoBias gender-coreference distribution.
    """
    ds = load_dataset("Elfsong/Wino_Bias", split=split)

    if type_filter is not None and "type" in ds.column_names:
        if isinstance(type_filter, str):
            allowed = {type_filter}
        else:
            allowed = set(type_filter)
        ds = ds.filter(lambda ex: ex["type"] in allowed)
    if polarity_filter is not None and "polarity" in ds.column_names:
        ds = ds.filter(lambda ex: ex["polarity"] == polarity_filter)

    pairs: List[ProbePair] = []
    for i, row in enumerate(ds):
        base = str(row.get("input", ""))
        base = base.replace("[", "").replace("]", "").strip()
        swapped = _swap_pronouns(base)
        if swapped == base:
            continue
        pairs.append(
            ProbePair(
                pair_id=f"winobias_{split}_{i}",
                text_a=base,
                text_b=swapped,
                meta={
                    "source": "Elfsong/Wino_Bias",
                    "split": split,
                    "type": row.get("type"),
                    "polarity": row.get("polarity"),
                    "type_filter": list(allowed) if type_filter is not None else None,
                    "polarity_filter": polarity_filter,
                },
            )
        )
        if max_pairs and len(pairs) >= max_pairs:
            break

    return pairs


def build_stereoset_pairs(
    config: str = "intrasentence",
    split: str = "validation",
    bias_types: Optional[List[str]] = None,
    max_pairs: int = 0,
) -> List[ProbePair]:
    """Build StereoSet paired probes (stereotype vs anti-stereotype).

    Source dataset: McGill-NLP/stereoset

    For each example, we locate the sentence with gold_label==1 (stereotype)
    and gold_label==0 (anti-stereotype), then create a pair.
    """
    ds = load_dataset("McGill-NLP/stereoset", config, split=split)

    pairs: List[ProbePair] = []
    for ex in ds:
        bt = ex.get("bias_type", None)
        if bias_types is not None and bt not in bias_types:
            continue

        context = str(ex.get("context", "")).strip()
        sents = ex.get("sentences", {}).get("sentence", [])
        labels = ex.get("sentences", {}).get("gold_label", [])

        stereo_sent = None
        anti_sent = None
        for sent, gl in zip(sents, labels):
            if gl == 1:
                stereo_sent = str(sent).strip()
            elif gl == 0:
                anti_sent = str(sent).strip()

        if not stereo_sent or not anti_sent:
            continue

        text_a = f"{context} {stereo_sent}".strip()
        text_b = f"{context} {anti_sent}".strip()

        ex_id = ex.get("id", "")
        pairs.append(
            ProbePair(
                pair_id=f"stereoset_{config}_{ex_id}",
                text_a=text_a,
                text_b=text_b,
                meta={
                    "source": "McGill-NLP/stereoset",
                    "config": config,
                    "split": split,
                    "bias_type": bt,
                },
            )
        )

        if max_pairs and len(pairs) >= max_pairs:
            break

    return pairs


def write_pairs_csv(pairs: Iterable[ProbePair], path: str) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "text_a", "text_b"])  # no labels; probes only
        for p in pairs:
            w.writerow([p.pair_id, p.text_a, p.text_b])


def write_pairs_jsonl(pairs: Iterable[ProbePair], path: str) -> None:
    import json

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")
