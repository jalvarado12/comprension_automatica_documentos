from __future__ import annotations

from typing import Any, Dict


def parse_ocr_block(block: Dict[str, Any]) -> Dict[str, Any]:
    region = block.get("region", {}) or {}
    bbox = region.get("bbox", [0, 0, 0, 0])
    ocr_lines = block.get("ocr_lines", []) or []
    full_text = "\n".join(
        line["text"] for line in ocr_lines if line.get("text")
    )
    return {
        "page": block.get("page_num", -1),
        "type": "text",
        "bbox": bbox,
        "content": full_text,
        "source": "ocr_json",
    }


def parse_caption_block(block: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "page": block.get("page_num", -1),
        "type": "figure",
        "bbox": block.get("bbox", [0, 0, 0, 0]),
        "content": block.get("contextual_description", ""),
        "source": "florence2",
    }


def parse_table_block(block: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "page": block.get("page_num", -1),
        "type": "table",
        "bbox": block.get("bbox", [0, 0, 0, 0]),
        "content": block.get("markdown", "") or "",
        "source": "table_transformer",
        "source_file": block.get("source_file", ""),
    }
