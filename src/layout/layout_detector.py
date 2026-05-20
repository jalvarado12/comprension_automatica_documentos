from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

import numpy as np

from src.models.region import Region

logger = logging.getLogger(__name__)


class LayoutDetector:
    def __init__(self, config):
        self.model_name: str = config.model_name
        self.threshold: float = config.threshold
        self._engine = None

    def load(self) -> None:
        from paddleocr import PPStructure
        self._engine = PPStructure(
            table=False,
            ocr=False,
            show_log=False,
        )
        logger.info(f"LayoutDetector cargado: {self.model_name}")

    def unload(self) -> None:
        self._engine = None
        logger.info("LayoutDetector descargado")

    def detect(self, page_bgr: np.ndarray, page_num: int) -> List[Region]:
        if self._engine is None:
            raise RuntimeError("Llama a load() antes de detect()")

        h, w = page_bgr.shape[:2]
        result = self._engine(page_bgr)

        if not result:
            return []

        regions: List[Region] = []
        for item in result:
            try:
                item_dict = self._to_dict(item)
            except TypeError as e:
                logger.warning(f"Página {page_num}: no se pudo convertir item de layout: {e}")
                continue

            region_type = item_dict.get("type", "unknown")
            bbox = item_dict.get("bbox", None)

            if bbox is None or len(bbox) != 4:
                continue

            x1, y1, x2, y2 = map(int, bbox)
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))

            if x2 <= x1 or y2 <= y1:
                continue

            score = float(item_dict.get("score", 1.0))
            if score < self.threshold:
                continue

            regions.append(Region(
                page_num=page_num,
                kind=region_type,
                bbox=(x1, y1, x2, y2),
                score=score,
                source="ppstructure",
            ))

        logger.debug(f"Página {page_num}: {len(regions)} regiones de layout")
        return regions

    @staticmethod
    def _to_dict(res: Any) -> Dict[str, Any]:
        if isinstance(res, dict):
            return res
        if hasattr(res, "json"):
            value = res.json
            if isinstance(value, dict):
                return value
            if callable(value):
                out = value()
                if isinstance(out, dict):
                    return out
                if isinstance(out, str):
                    return json.loads(out)
        if hasattr(res, "to_json") and callable(res.to_json):
            maybe = res.to_json()
            if isinstance(maybe, str):
                return json.loads(maybe)
            if isinstance(maybe, dict):
                return maybe
        if hasattr(res, "res"):
            return {"res": res.res}
        raise TypeError(f"No se pudo convertir tipo: {type(res)}")
