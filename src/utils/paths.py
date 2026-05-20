from __future__ import annotations

from pathlib import Path
from typing import Optional


def find_project_root(start: Optional[Path] = None) -> Path:
    """
    Sube desde `start` buscando marcadores de raíz de proyecto.
    Extraído y centralizado desde los 6 notebooks donde aparecía duplicado.
    """
    start = (start or Path.cwd()).resolve()
    for base in [start, *start.parents]:
        markers = [
            ("data", "notebooks"),
            ("src", "data"),
            ("pyproject.toml",),
            ("requirements.txt",),
        ]
        if any(all((base / m).exists() for m in group) for group in markers):
            return base
    return start


def resolve_run_dir(project_root: Path, run_name: str) -> Path:
    run_dir = project_root / "data" / "outputs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
