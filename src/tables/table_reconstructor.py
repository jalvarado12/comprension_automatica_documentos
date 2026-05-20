from __future__ import annotations

import logging
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd

from src.tables.cell_ocr import CellOCR

logger = logging.getLogger(__name__)


class TableReconstructor:
    """
    Reconstruye un DataFrame a partir de la estructura detectada por TATR
    y OCR por celda. Convierte a Markdown.
    """
    def __init__(self, cell_ocr: CellOCR, id2label: Dict[int, str]):
        self.cell_ocr = cell_ocr
        self.id2label = id2label

    def reconstruct(self, table_bgr: np.ndarray, results: Dict) -> pd.DataFrame:
        rows: List[List[int]] = []
        cols: List[List[int]] = []

        for score, label, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            label_name = self.id2label[label.item()]
            box = [int(i) for i in box.tolist()]

            if label_name == "table row":
                rows.append(box)
            elif label_name == "table column":
                cols.append(box)

        rows = sorted(rows, key=lambda x: x[1])
        cols = sorted(cols, key=lambda x: x[0])

        logger.debug(f"Filas: {len(rows)}, Columnas: {len(cols)}")

        if not rows or not cols:
            return pd.DataFrame()

        pad = 5
        matrix = []
        for r in rows:
            ry1, ry2 = r[1], r[3]
            row_data = []
            for c in cols:
                cx1, cx2 = c[0], c[2]
                crop = table_bgr[
                    max(0, ry1 + pad): min(table_bgr.shape[0], ry2 - pad),
                    max(0, cx1 + pad): min(table_bgr.shape[1], cx2 - pad),
                ]
                text = self.cell_ocr.run(crop) if crop.size > 0 else ""
                row_data.append(text)
            matrix.append(row_data)

        df = pd.DataFrame(matrix)
        df = self._clean(df)
        return df

    def to_markdown(self, df: pd.DataFrame) -> str:
        if df.empty:
            return ""
        return df.to_markdown(index=False)

    @staticmethod
    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        df = df.replace("", np.nan)
        df = df.dropna(how="all")
        df = df.dropna(axis=1, how="all")
        return df.fillna("")
