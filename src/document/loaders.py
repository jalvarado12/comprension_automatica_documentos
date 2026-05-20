from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class DocumentLoader:
    def load_ocr_blocks(self, ocr_json_dir: Path) -> List[Dict[str, Any]]:
        data = []
        files = sorted(ocr_json_dir.glob("*.json"))
        logger.info(f"JSON OCR encontrados: {len(files)}")
        for file in files:
            try:
                content = json.loads(file.read_text(encoding="utf-8"))
                if isinstance(content, list):
                    data.extend(content)
                else:
                    data.append(content)
            except Exception as e:
                logger.warning(f"{file.name}: {e}")
        return data

    def load_captions(self, captions_json_path: Path) -> List[Dict[str, Any]]:
        if not captions_json_path.exists():
            logger.warning(f"No existe: {captions_json_path}")
            return []
        data = json.loads(captions_json_path.read_text(encoding="utf-8"))
        logger.info(f"Captions cargados: {len(data)}")
        return data

    def load_tables(self, tables_markdown_dir: Path) -> List[Dict[str, Any]]:
        tables = []
        files = sorted(tables_markdown_dir.glob("*.md"))
        logger.info(f"Tablas markdown encontradas: {len(files)}")
        for md_file in files:
            markdown = md_file.read_text(encoding="utf-8")
            m = re.search(r"page[_-](\d+)", md_file.stem)
            page_num = int(m.group(1)) if m else -1
            tables.append({
                "page_num": page_num,
                "bbox": [0, 0, 0, 0],
                "markdown": markdown,
                "source_file": md_file.name,
            })
        return tables
