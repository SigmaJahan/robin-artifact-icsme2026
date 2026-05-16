#!/usr/bin/env bash
# Source this from other scripts to set the local environment.
# Reviewers: override VENV_PY and HF_HOME for your machine if the defaults
# do not apply. See README.md for setup steps.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PROJECT_ROOT

# Default to a project-local venv at .venv/. Override with VENV_PY=/abs/path/python.
VENV_PY="${VENV_PY:-$PROJECT_ROOT/.venv/bin/python}"
if [[ ! -x "$VENV_PY" ]]; then
  echo "[env:error] venv python missing: $VENV_PY" >&2
  echo "Create one with: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt torch" >&2
  return 1 2>/dev/null || exit 1
fi
export VENV_PY

# HF cache: defaults to project-local hf-cache/. Override with HF_HOME=/abs/path
# if you already have a shared Hugging Face cache.
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/hf-cache}"
export HF_HUB_CACHE="$HF_HOME"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg

mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"
