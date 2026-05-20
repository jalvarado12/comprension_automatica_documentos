from __future__ import annotations

import logging
from typing import List

from src.models.region import Region
from src.utils.geometry import compute_iou

logger = logging.getLogger(__name__)


class RegionMerger:
    """
    Fusiona regiones de layout (PPStructure) con regiones de TATR.
    Cuando se solapan, TATR tiene prioridad para tablas.
    """
    def __init__(self, config):
        self.iou_table_priority: float = config.iou_table_priority
        self.iou_drop_figure_threshold: float = config.iou_drop_figure_threshold

    def merge(
        self,
        layout_regions: List[Region],
        tatr_regions: List[Region],
    ) -> List[Region]:
        """
        - Las tablas TATR reemplazan tablas de layout si IoU > iou_table_priority.
        - Las figuras de layout se eliminan si se solapan con tablas TATR
          con IoU > iou_drop_figure_threshold.
        """
        result: List[Region] = list(tatr_regions)
        tatr_boxes = [r.bbox for r in tatr_regions]

        for region in layout_regions:
            if region.kind == "table":
                # Ya cubierto por TATR si hay solapamiento
                overlaps = any(
                    compute_iou(region.bbox, tb) >= self.iou_table_priority
                    for tb in tatr_boxes
                )
                if not overlaps:
                    result.append(region)

            elif region.kind in ("figure", "image"):
                # Descartar si cae sobre una tabla TATR
                covered = any(
                    compute_iou(region.bbox, tb) >= self.iou_drop_figure_threshold
                    for tb in tatr_boxes
                )
                if not covered:
                    result.append(region)

            else:
                result.append(region)

        logger.debug(
            f"Merge: {len(layout_regions)} layout + {len(tatr_regions)} TATR "
            f"→ {len(result)} regiones"
        )
        return result
