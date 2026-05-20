from __future__ import annotations

import re
from typing import List, Tuple


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def polygon_to_bbox(poly: List[List[float]]) -> Tuple[int, int, int, int]:
    xs = [float(p[0]) for p in poly]
    ys = [float(p[1]) for p in poly]
    return (
        int(round(min(xs))),
        int(round(min(ys))),
        int(round(max(xs))),
        int(round(max(ys))),
    )


def normalize_bbox_values(
    x1: float, y1: float, x2: float, y2: float
) -> Tuple[int, int, int, int]:
    return (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))


def count_noise(text: str) -> int:
    """Cuenta caracteres que no son alfanuméricos ni puntuación estándar."""
    pattern = r"[^a-zA-Z0-9\s\.,;:\-\(\)#\|]"
    return len(re.findall(pattern, text))
