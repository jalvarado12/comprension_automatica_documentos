from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd
from tqdm.auto import tqdm

from src.captioning.florence_model import FlorenceModel
from src.captioning.ocr_context_builder import OCRContextBuilder

logger = logging.getLogger(__name__)


class ContextualCaptioner:
    """
    Genera descripciones contextuales para figuras combinando:
    - Caption visual base de Florence-2
    - Contexto textual OCR de la misma página
    """
    def __init__(self, florence: FlorenceModel, context_builder: OCRContextBuilder):
        self.florence = florence
        self.context_builder = context_builder

    def caption_figure(self, figure: Dict, ocr_blocks_by_page: Dict) -> Dict:
        image = figure["image"]
        page_num = figure["page_num"]
        bbox = tuple(figure["bbox"])

        florence_caption = ""
        try:
            florence_caption = self.florence.caption(image)
        except Exception as e:
            logger.warning(f"Florence falló para {figure['figure_id']}: {e}")

        context_text = self.context_builder.build(bbox, page_num, ocr_blocks_by_page)

        parts = [p for p in [florence_caption, context_text] if p]
        contextual_description = " | ".join(parts) if parts else ""

        return {
            **figure,
            "florence_caption": florence_caption,
            "context_text": context_text,
            "contextual_description": contextual_description,
            "image": None,  # no serializar PIL Image al JSON
        }

    def caption_all(
        self,
        figures: List[Dict],
        ocr_blocks_by_page: Dict,
        output_dir: Path,
    ) -> List[Dict]:
        output_dir.mkdir(parents=True, exist_ok=True)
        results = []

        for figure in tqdm(figures, desc="Captioning figuras"):
            result = self.caption_figure(figure, ocr_blocks_by_page)
            results.append(result)

        # Guardar JSON
        json_path = output_dir / "contextual_image2text_results.json"
        serializable = [
            {k: v for k, v in r.items() if k != "image"}
            for r in results
        ]
        json_path.write_text(
            json.dumps(serializable, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # Guardar CSV
        csv_path = output_dir / "contextual_image2text_results.csv"
        pd.DataFrame(serializable).to_csv(csv_path, index=False)

        logger.info(f"Captions guardados: {len(results)} figuras → {output_dir}")
        return results
