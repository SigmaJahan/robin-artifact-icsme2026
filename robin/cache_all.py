from __future__ import annotations

import argparse

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer


def cache_models(models: list[str]) -> None:
    for m in models:
        print(f"[cache:model] {m}")
        tok = AutoTokenizer.from_pretrained(m)
        n = m.lower()
        if "gpt2" in n:
            model = AutoModelForCausalLM.from_pretrained(m)
        else:
            model = AutoModelForMaskedLM.from_pretrained(m)
        # touch attributes to ensure fully cached
        _ = len(tok)
        _ = model.config.to_dict()


def cache_datasets() -> None:
    print("[cache:dataset] Elfsong/Wino_Bias (test)")
    load_dataset("Elfsong/Wino_Bias", split="test")

    print("[cache:dataset] McGill-NLP/stereoset (intrasentence, validation)")
    load_dataset("McGill-NLP/stereoset", "intrasentence", split="validation")

    print("[cache:dataset] wikitext (wikitext-2-raw-v1, test)")
    load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    print("[cache:dataset] wikitext (wikitext-103-raw-v1, test)")
    load_dataset("wikitext", "wikitext-103-raw-v1", split="test")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Model names to download and cache",
    )
    args = ap.parse_args()

    cache_models(args.models)
    cache_datasets()
    print("[cache] Done")


if __name__ == "__main__":
    main()
