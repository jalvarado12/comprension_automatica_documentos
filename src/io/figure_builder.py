from __future__ import annotations

import ast
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

logger = logging.getLogger(__name__)


class FigureBuilder:
    def load_page_images(self, page_images_dir: Path) -> Dict[int, Image.Image]:
        import re
        page_images: Dict[int, Image.Image] = {}
        for img_path in sorted(page_images_dir.glob("page_*.png")):
            m = re.search(r"page_(\d+)", img_path.stem, flags=re.IGNORECASE)
            if not m:
                continue
            page_num = int(m.group(1))
            page_images[page_num] = Image.open(img_path).convert("RGB")
        logger.info(f"Imágenes de página cargadas: {len(page_images)}")
        return page_images

    def build_figures_from_detections(
        self,
        detections_df: pd.DataFrame,
        page_images_by_page: Dict[int, Image.Image],
        output_dir: Path,
    ) -> List[Dict]:
        output_dir.mkdir(parents=True, exist_ok=True)

        if "kind" in detections_df.columns:
            figures_df = detections_df[
                detections_df["kind"].astype(str).str.lower() == "figure"
            ].copy()
        else:
            figures_df = detections_df.copy()

        figures = []
        for idx, row in figures_df.iterrows():
            try:
                page_num = int(row["page_num"])
                bbox = self._parse_bbox(row["bbox"])
                score = float(row["score"]) if "score" in row and row["score"] is not None else 1.0
                source = row.get("source", "unknown")

                if page_num not in page_images_by_page:
                    logger.warning(f"Sin imagen para página {page_num}, fila {idx}")
                    continue

                page_img = page_images_by_page[page_num]
                crop = self._crop(page_img, bbox)

                figure_id = f"fig_p{page_num}_{idx}"
                image_path = output_dir / f"{figure_id}.png"
                crop.save(image_path)

                figures.append({
                    "figure_id": figure_id,
                    "page_num": page_num,
                    "bbox": list(bbox),
                    "score": score,
                    "source": source,
                    "image": crop,
                    "image_path": str(image_path),
                })

            except Exception as e:
                logger.warning(f"Fila {idx}: {e}")

        logger.info(f"Figuras construidas: {len(figures)}")
        return figures

    @staticmethod
    def _parse_bbox(value) -> Tuple[int, int, int, int]:
        if isinstance(value, (list, tuple)) and len(value) == 4:
            return tuple(int(v) for v in value)
        if isinstance(value, str):
            try:
                parsed = ast.literal_eval(value)
            except Exception:
                parsed = json.loads(value)
            return tuple(int(v) for v in parsed)
        raise ValueError(f"Formato de bbox no soportado: {value}")

    @staticmethod
    def _crop(page_img: Image.Image, bbox: Tuple[int, int, int, int]) -> Image.Image:
        x1, y1, x2, y2 = bbox
        return page_img.crop((x1, y1, x2, y2)).convert("RGB")
