from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.utils.geometry import bgr_to_rgb
from src.utils.text_utils import clean_text

logger = logging.getLogger(__name__)


class OCREngine:
    """
    OCR con PaddleOCR sobre crops de bloques de texto.
    Guarda resultados estructurados en JSON y texto plano.
    """
    def __init__(self, config):
        self.lang: str = config.lang
        self.use_angle_cls: bool = config.use_angle_cls
        self.min_score: float = config.min_score
        self.use_gpu: bool = config.use_gpu
        self._engine = None

    def load(self) -> None:
        from paddleocr import PaddleOCR
        self._engine = PaddleOCR(
            use_angle_cls=self.use_angle_cls,
            lang=self.lang,
            show_log=False,
            use_gpu=self.use_gpu,
        )
        logger.info("OCREngine cargado")

    def unload(self) -> None:
        self._engine = None
        logger.info("OCREngine descargado")

    def run(self, crop_bgr: np.ndarray) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Ejecuta OCR sobre un crop BGR.
        Retorna (texto_completo, lista_de_líneas).
        """
        if self._engine is None:
            raise RuntimeError("Llama a load() antes de run()")

        if crop_bgr is None or crop_bgr.size == 0:
            return "", []

        result = self._engine.ocr(bgr_to_rgb(crop_bgr), cls=self.use_angle_cls)

        if not result or not result[0]:
            return "", []

        lines = []
        texts = []
        for line in result[0]:
            box, (txt, score) = line
            score = float(score)
            if score >= self.min_score:
                lines.append({"box": box, "text": clean_text(txt), "score": score})
                texts.append(clean_text(txt))

        return "\n".join(texts).strip(), lines

    def process_region(
        self,
        crop_bgr: np.ndarray,
        page_num: int,
        region_idx: int,
        region_bbox: List[int],
        ocr_json_dir: Path,
        ocr_txt_dir: Path,
    ) -> Optional[Dict]:
        """
        Ejecuta OCR y guarda los resultados en disco.
        Formato JSON compatible con el loader del NB3.
        """
        text, lines = self.run(crop_bgr)

        if not lines:
            return None

        payload = {
            "page_num": page_num,
            "region": {"bbox": region_bbox},
            "ocr_lines": lines,
            "full_text": text,
        }

        stem = f"page{page_num:03d}_region{region_idx:04d}"

        json_path = ocr_json_dir / f"{stem}.json"
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        txt_path = ocr_txt_dir / f"{stem}.txt"
        txt_path.write_text(text, encoding="utf-8")

        return payload
