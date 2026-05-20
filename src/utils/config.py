from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: Path) -> "PipelineConfig":
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return PipelineConfig.from_dict(raw)


class _Namespace:
    """Convierte un dict anidado en atributos accesibles por punto."""
    def __init__(self, data: Dict[str, Any]):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, _Namespace(value))
            else:
                setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


class PipelineConfig(_Namespace):
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineConfig":
        return cls(data)
