from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class ProbePair:
    """A paired probe (x, x') differing only in a sensitive attribute."""

    pair_id: str
    text_a: str
    text_b: str
    meta: Optional[Dict[str, Any]] = None
