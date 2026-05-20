from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.utils.text_utils import clean_text, polygon_to_bbox

logger = logging.getLogger(__name__)


class OCRContextBuilder:
    """
    Construye el contexto textual para una figura a partir de los
    bloques OCR cercanos en la misma página.
    """
    def __init__(self, config):
        self.max_vertical_gap: int = config.max_vertical_gap
        self.max_horizontal_expand: int = config.max_horizontal_expand
        self.max_nearby_blocks: int = config.max_nearby_blocks
        self.max_mention_blocks: int = config.max_mention_blocks

    def build(
        self,
        figure_bbox: Tuple[int, int, int, int],
        page_num: int,
        ocr_blocks_by_page: Dict[int, List[Dict]],
    ) -> str:
        page_blocks = ocr_blocks_by_page.get(page_num, [])
        if not page_blocks:
            return ""

        nearby = self._get_nearby_blocks(figure_bbox, page_blocks)
        nearby_text = " ".join(b["text"] for b in nearby if b.get("text"))
        return clean_text(nearby_text)

    def _get_nearby_blocks(
        self,
        figure_bbox: Tuple[int, int, int, int],
        page_blocks: List[Dict],
    ) -> List[Dict]:
        fx1, fy1, fx2, fy2 = figure_bbox
        result = []

        for block in page_blocks:
            if block.get("level") == "region":
                continue
            bx1, by1, bx2, by2 = block["bbox"]

            vertical_ok = (
                abs(by1 - fy2) <= self.max_vertical_gap
                or abs(fy1 - by2) <= self.max_vertical_gap
            )
            horizontal_ok = not (
                bx2 < fx1 - self.max_horizontal_expand
                or bx1 > fx2 + self.max_horizontal_expand
            )

            if vertical_ok and horizontal_ok:
                result.append(block)

            if len(result) >= self.max_nearby_blocks:
                break

        return result

    @staticmethod
    def load_ocr_blocks_from_json(
        ocr_json_dir: Path,
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        Reconstruye bloques OCR por página desde los JSONs de 08_ocr_json.
        Las coordenadas de cada línea son locales a la región —
        se convierten a coordenadas de página con el offset de region.bbox.
        """
        ocr_blocks_by_page: Dict[int, List[Dict]] = {}
        skipped = []

        for json_file in sorted(ocr_json_dir.glob("*.json")):
            try:
                payload = json.loads(json_file.read_text(encoding="utf-8"))
            except Exception as e:
                skipped.append((json_file.name, f"json_error: {e}"))
                continue

            page_num = payload.get("page_num")
            region = payload.get("region", {}) or {}
            region_bbox = region.get("bbox")
            ocr_lines = payload.get("ocr_lines", []) or []

            if not isinstance(page_num, int) or page_num < 1:
                skipped.append((json_file.name, "page_num inválido"))
                continue
            if not isinstance(region_bbox, list) or len(region_bbox) != 4:
                skipped.append((json_file.name, "region.bbox inválido"))
                continue
            if not ocr_lines:
                skipped.append((json_file.name, "sin ocr_lines"))
                continue

            rx1, ry1 = int(round(region_bbox[0])), int(round(region_bbox[1]))

            page_blocks = []
            full_parts = []
            scores = []

            for line in ocr_lines:
                text = clean_text(line.get("text", ""))
                poly = line.get("box")
                score = float(line.get("score", 1.0))

                if not text or not isinstance(poly, list) or len(poly) < 4:
                    continue

                try:
                    lx1, ly1, lx2, ly2 = polygon_to_bbox(poly)
                except Exception:
                    continue

                page_blocks.append({
                    "text": text,
                    "bbox": (rx1 + lx1, ry1 + ly1, rx1 + lx2, ry1 + ly2),
                    "score": score,
                    "level": "line",
                })
                full_parts.append(text)
                scores.append(score)

            if page_blocks:
                avg_score = sum(scores) / max(1, len(scores))
                page_blocks.append({
                    "text": clean_text(" ".join(full_parts)),
                    "bbox": (
                        int(round(region_bbox[0])), int(round(region_bbox[1])),
                        int(round(region_bbox[2])), int(round(region_bbox[3])),
                    ),
                    "score": avg_score,
                    "level": "region",
                })
                ocr_blocks_by_page.setdefault(page_num, []).extend(page_blocks)

        for pn in ocr_blocks_by_page:
            ocr_blocks_by_page[pn].sort(
                key=lambda b: (b["bbox"][1], b["bbox"][0], 0 if b["level"] == "line" else 1)
            )

        if skipped:
            logger.warning(f"Archivos OCR omitidos: {len(skipped)}")

        logger.info(
            f"OCR reconstruido: {len(ocr_blocks_by_page)} páginas, "
            f"{sum(len(v) for v in ocr_blocks_by_page.values())} bloques"
        )
        return ocr_blocks_by_page
