from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from src.document.loaders import DocumentLoader
from src.document.parsers import parse_caption_block, parse_ocr_block, parse_table_block

logger = logging.getLogger(__name__)


class MultimodalDocumentBuilder:
    def __init__(self, run_name: str):
        self.run_name = run_name

    def build(
        self,
        ocr_blocks: List[Dict],
        caption_blocks: List[Dict],
        table_blocks: List[Dict],
    ) -> Dict[str, Any]:
        parsed_ocr = [parse_ocr_block(b) for b in ocr_blocks]
        parsed_captions = [parse_caption_block(b) for b in caption_blocks]
        parsed_tables = [parse_table_block(b) for b in table_blocks]

        all_blocks = self.sort_blocks(parsed_ocr + parsed_captions + parsed_tables)

        pages = defaultdict(list)
        for block in all_blocks:
            pages[block["page"]].append(block)

        document = {
            "run_name": self.run_name,
            "pages": [
                {"page": pn, "blocks": pages[pn]}
                for pn in sorted(pages.keys())
            ],
        }
        logger.info(f"Documento construido: {len(document['pages'])} páginas, {len(all_blocks)} bloques")
        return document

    def to_markdown(self, document: Dict[str, Any]) -> str:
        lines = []
        for page in document["pages"]:
            lines.append(f"\n\n# Página {page['page']}\n")
            for block in page["blocks"]:
                lines.append(self._block_to_markdown(block))
                lines.append("\n")
        return "".join(lines)

    def save_json(self, document: Dict[str, Any], output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(document, ensure_ascii=False, indent=4),
            encoding="utf-8",
        )
        logger.info(f"JSON multimodal guardado: {output_path}")

    def save_markdown(self, markdown: str, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown, encoding="utf-8")
        logger.info(f"Markdown multimodal guardado: {output_path}")

    @staticmethod
    def sort_blocks(blocks: List[Dict]) -> List[Dict]:
        return sorted(
            blocks,
            key=lambda b: (b["page"], b["bbox"][1], b["bbox"][0]),
        )

    @staticmethod
    def _block_to_markdown(block: Dict) -> str:
        content = (block.get("content") or "").strip()
        if not content:
            return ""
        btype = block.get("type", "text")
        if btype == "text":
            return content
        elif btype == "table":
            return f"\n\n## Tabla\n\n{content}\n"
        elif btype == "figure":
            return f"\n\n## Figura\n\n{content}\n"
        return content
