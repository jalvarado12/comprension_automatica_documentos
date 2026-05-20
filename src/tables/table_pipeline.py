from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd

from src.tables.cell_ocr import CellOCR
from src.tables.structure_recognizer import TableStructureRecognizer
from src.tables.table_reconstructor import TableReconstructor
from src.utils.geometry import save_bgr

logger = logging.getLogger(__name__)


class TableMarkdownPipeline:
    """
    Pipeline completo: crop de tabla → TATR estructura → OCR celdas → Markdown.
    Orquesta TableStructureRecognizer, CellOCR y TableReconstructor.
    """
    def __init__(self, config, run_dir: Path):
        self.config = config
        self.crop_padding: int = config.crop_padding
        self.upscale_factor: int = config.upscale_factor
        self.tables_dir = run_dir / "21_tables_tatr"
        self.markdown_dir = run_dir / "22_tables_markdown"

        self.tables_dir.mkdir(parents=True, exist_ok=True)
        self.markdown_dir.mkdir(parents=True, exist_ok=True)

        self.recognizer = TableStructureRecognizer(config)
        self.cell_ocr = CellOCR(tesseract_config=config.tesseract_config)
        self.reconstructor: Optional[TableReconstructor] = None

    def load(self) -> None:
        self.recognizer.load()
        self.reconstructor = TableReconstructor(self.cell_ocr, self.recognizer.id2label)
        logger.info("TableMarkdownPipeline cargado")

    def unload(self) -> None:
        self.recognizer.unload()
        logger.info("TableMarkdownPipeline descargado")

    def run(self, tables_df: pd.DataFrame, page_images_dir: Path) -> List[Dict]:
        if self.reconstructor is None:
            raise RuntimeError("Llama a load() antes de run()")

        # Parsear bbox si viene como string desde el CSV
        if tables_df["bbox"].dtype == object:
            tables_df = tables_df.copy()
            tables_df["bbox"] = tables_df["bbox"].apply(self._parse_bbox)

        all_results = []
        for idx, row in tables_df.iterrows():
            result = self._process_one(idx, row, page_images_dir)
            if result:
                all_results.append(result)

        logger.info(f"Tablas procesadas: {len(all_results)}/{len(tables_df)}")
        return all_results

    def _process_one(self, idx, row, page_images_dir: Path) -> Optional[Dict]:
        try:
            page_num = int(row["page_num"])
            bbox = row["bbox"]

            if isinstance(bbox, str):
                bbox = ast.literal_eval(bbox)

            x1, y1, x2, y2 = map(int, bbox[:4])

            page_path = page_images_dir / f"page_{page_num:03d}.png"
            image = cv2.imread(str(page_path))
            if image is None:
                logger.warning(f"No se pudo leer: {page_path}")
                return None

            pad = self.crop_padding
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(image.shape[1], x2 + pad)
            y2 = min(image.shape[0], y2 + pad)

            table_crop = image[y1:y2, x1:x2]
            if table_crop.size == 0:
                logger.warning(f"Tabla {idx}: recorte vacío")
                return None

            # Upscaling + binarización para mejor OCR
            fx = fy = self.upscale_factor
            table_crop = cv2.resize(table_crop, None, fx=fx, fy=fy,
                                    interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(table_crop, cv2.COLOR_BGR2GRAY)
            gray = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2,
            )
            table_crop = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            results = self.recognizer.recognize(table_crop)

            # Guardar debug visual
            debug_path = self.tables_dir / f"table_{idx:04d}.png"
            save_bgr(debug_path, table_crop)

            df_table = self.reconstructor.reconstruct(table_crop, results)
            if df_table.empty:
                logger.warning(f"Tabla {idx}: reconstrucción vacía")
                return None

            markdown = self.reconstructor.to_markdown(df_table)

            md_path = self.markdown_dir / f"table_{idx:04d}.md"
            md_path.write_text(markdown, encoding="utf-8")

            logger.debug(f"Tabla {idx} OK — {df_table.shape}")
            return {"table_id": idx, "page_num": page_num, "markdown": markdown}

        except Exception as e:
            logger.error(f"Error procesando tabla {idx}: {e}")
            return None

    @staticmethod
    def _parse_bbox(value):
        if isinstance(value, (list, tuple)):
            return tuple(int(v) for v in value)
        if isinstance(value, str):
            parsed = ast.literal_eval(value)
            return tuple(int(v) for v in parsed)
        raise ValueError(f"Formato de bbox no soportado: {value}")
