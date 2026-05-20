from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CellOCR:
    """
    OCR sobre celdas individuales de tabla usando Tesseract.
    El objeto CLAHE se crea una sola vez en __init__ y se reutiliza
    en cada celda (corrección del anti-patrón del notebook original).
    """
    def __init__(self, tesseract_config: str = "--oem 3 --psm 6 -l spa+eng"):
        self.tesseract_config = tesseract_config
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def run(self, cell_img: np.ndarray) -> str:
        if cell_img is None or cell_img.size == 0:
            return ""

        try:
            import pytesseract

            gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
            gray = cv2.fastNlMeansDenoising(gray)
            gray = self._clahe.apply(gray)
            gray = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31,
                11,
            )
            gray = cv2.copyMakeBorder(
                gray, 10, 10, 10, 10,
                cv2.BORDER_CONSTANT,
                value=255,
            )
            text = pytesseract.image_to_string(gray, config=self.tesseract_config)
            return text.strip()

        except Exception as e:
            logger.warning(f"Error OCR en celda: {e}")
            return ""
