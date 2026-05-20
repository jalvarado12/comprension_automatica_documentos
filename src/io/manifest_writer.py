from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List

import pandas as pd

from src.models.region import Region

logger = logging.getLogger(__name__)


class ManifestWriter:
    def save(self, regions: List[Region], output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "page_num": r.page_num,
                "kind": r.kind,
                "bbox": list(r.bbox),
                "score": r.score,
                "source": r.source,
                "meta": json.dumps(r.meta) if r.meta else None,
            }
            for r in regions
        ]
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        logger.info(f"Manifest guardado: {output_path} ({len(regions)} regiones)")

    def save_jsonl(self, regions: List[Region], output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for r in regions:
                f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
        logger.info(f"JSONL guardado: {output_path}")

    def load(self, path: Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        logger.info(f"Manifest cargado: {path} ({len(df)} filas)")
        return df
