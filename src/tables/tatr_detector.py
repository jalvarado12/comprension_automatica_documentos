from __future__ import annotations

import logging
from typing import List

import numpy as np
import torch
from PIL import Image

from src.models.region import Region
from src.utils.geometry import bgr_to_rgb, clamp_box

logger = logging.getLogger(__name__)


class TATRDetector:
    """
    Detecta dónde hay tablas en una página completa.
    Usa microsoft/table-transformer-detection.
    """
    def __init__(self, config):
        self.model_name: str = config.detection_model
        self.threshold: float = config.detection_threshold
        self.device: str = "cpu"
        self._processor = None
        self._model = None

    def load(self) -> None:
        from transformers import DetrImageProcessor, TableTransformerForObjectDetection
        self._processor = DetrImageProcessor.from_pretrained(self.model_name)
        self._model = TableTransformerForObjectDetection.from_pretrained(self.model_name)
        self._model.eval()
        self._model.to(self.device)
        logger.info(f"TATRDetector cargado: {self.model_name}")

    def unload(self) -> None:
        self._model = None
        self._processor = None
        logger.info("TATRDetector descargado")

    def detect(self, page_bgr: np.ndarray, page_num: int) -> List[Region]:
        if self._model is None:
            raise RuntimeError("Llama a load() antes de detect()")

        h, w = page_bgr.shape[:2]
        pil_img = Image.fromarray(bgr_to_rgb(page_bgr))
        inputs = self._processor(images=pil_img, return_tensors="pt")

        with torch.no_grad():
            outputs = self._model(**inputs)

        target_sizes = torch.tensor([pil_img.size[::-1]])
        results = self._processor.post_process_object_detection(
            outputs,
            threshold=self.threshold,
            target_sizes=target_sizes,
        )[0]

        regions: List[Region] = []
        for score, label, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            x1, y1, x2, y2 = clamp_box(box.tolist(), w, h)
            regions.append(Region(
                page_num=page_num,
                kind="table",
                bbox=(x1, y1, x2, y2),
                score=float(score.item()),
                source="tatr",
                meta={"label": int(label.item())},
            ))

        logger.debug(f"Página {page_num}: {len(regions)} tablas TATR")
        return regions
