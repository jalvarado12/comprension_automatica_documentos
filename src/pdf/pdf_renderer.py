from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from pdf2image import convert_from_path

from src.utils.geometry import rgb_to_bgr, save_bgr

logger = logging.getLogger(__name__)


class PDFRenderer:
    def __init__(self, config):
        self.dpi: int = config.dpi
        self.poppler_path: Optional[str] = config.poppler_path
        self.use_preprocessing: bool = config.use_preprocessing
        self.clahe_clip_limit: float = config.clahe_clip_limit
        self.clahe_grid_size: tuple = tuple(config.clahe_grid_size)

    def render(self, pdf_path: Path, output_dir: Path) -> List[np.ndarray]:
        """
        Convierte cada página del PDF en imagen BGR.
        Guarda PNGs en output_dir y retorna la lista de arrays para
        procesamiento inmediato sin re-leer desde disco.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        pages = convert_from_path(
            str(pdf_path),
            dpi=self.dpi,
            poppler_path=self.poppler_path,
        )
        result: List[np.ndarray] = []
        for idx, pil_img in enumerate(pages, start=1):
            bgr = rgb_to_bgr(np.array(pil_img.convert("RGB")))
            save_bgr(output_dir / f"page_{idx:03d}.png", bgr)
            result.append(bgr)
            logger.debug(f"Página {idx} renderizada")
        logger.info(f"PDF renderizado: {len(result)} páginas")
        return result

    def preprocess(self, page_bgr: np.ndarray) -> np.ndarray:
        """
        Mejora de contraste con CLAHE en espacio LAB.
        Si use_preprocessing=False retorna copia sin modificar.
        """
        if not self.use_preprocessing:
            return page_bgr.copy()

        lab = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=self.clahe_grid_size,
        )
        l2 = clahe.apply(l)
        merged = cv2.merge([l2, a, b])
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    def render_and_preprocess(
        self, pdf_path: Path, pages_dir: Path, preprocessed_dir: Path
    ) -> List[np.ndarray]:
        """
        Render + preproceso. Guarda ambas versiones en disco.
        Retorna las imágenes preprocesadas para el pipeline.
        """
        preprocessed_dir.mkdir(parents=True, exist_ok=True)
        raw_pages = self.render(pdf_path, pages_dir)
        result = []
        for idx, bgr in enumerate(raw_pages, start=1):
            pre = self.preprocess(bgr)
            save_bgr(preprocessed_dir / f"page_{idx:03d}.png", pre)
            result.append(pre)
        return result
