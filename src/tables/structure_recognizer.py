from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import torch
from PIL import Image

from src.utils.geometry import bgr_to_rgb

logger = logging.getLogger(__name__)


class TableStructureRecognizer:
    """
    Detecta la estructura interna de una tabla ya recortada:
    filas, columnas, headers.
    Usa microsoft/table-transformer-structure-recognition.
    """
    def __init__(self, config):
        self.model_name: str = config.structure_model
        self.threshold: float = config.structure_threshold
        self.device: str = "cpu"
        self._processor = None
        self._model = None
        self.id2label: Dict[int, str] = {}

    def load(self) -> None:
        from transformers import AutoImageProcessor, TableTransformerForObjectDetection
        self._processor = AutoImageProcessor.from_pretrained(self.model_name)
        self._model = TableTransformerForObjectDetection.from_pretrained(self.model_name)
        self._model.eval()
        self._model.to(self.device)
        self.id2label = self._model.config.id2label
        logger.info(f"TableStructureRecognizer cargado: {self.model_name}")

    def unload(self) -> None:
        self._model = None
        self._processor = None
        logger.info("TableStructureRecognizer descargado")

    def recognize(self, table_bgr: np.ndarray) -> Dict:
        if self._model is None:
            raise RuntimeError("Llama a load() antes de recognize()")

        pil_img = Image.fromarray(bgr_to_rgb(table_bgr))
        encoding = self._processor(images=pil_img, return_tensors="pt")
        encoding = {k: v.to(self.device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = self._model(**encoding)

        target_sizes = torch.tensor([pil_img.size[::-1]]).to(self.device)
        results = self._processor.post_process_object_detection(
            outputs,
            threshold=self.threshold,
            target_sizes=target_sizes,
        )[0]

        return results
