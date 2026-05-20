from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict

from src.llm.gemini_client import GeminiClient
from src.llm.prompts import build_ocr_correction_prompt, build_summary_prompt

logger = logging.getLogger(__name__)

MAX_CHARS_DEFAULT = 50_000


class LLMCorrector:
    def __init__(self, client: GeminiClient, config):
        self.client = client
        self.max_chars: int = getattr(config, "max_chars_per_chunk", MAX_CHARS_DEFAULT)
        self.prompts_dir = Path(getattr(config, "prompts_dir", "configs/prompts"))

    def correct(self, document_text: str) -> str:
        chunks = self._chunk(document_text)
        logger.info(f"Corrección OCR: {len(chunks)} chunk(s)")
        corrected_parts = []
        for i, chunk in enumerate(chunks, 1):
            prompt = build_ocr_correction_prompt(chunk, self.prompts_dir)
            result = self.client.generate(prompt)
            corrected_parts.append(result)
            logger.debug(f"Chunk {i}/{len(chunks)} corregido")
        return "\n".join(corrected_parts)

    def summarize(self, document_text: str) -> str:
        # El resumen se hace sobre el documento completo (no se chunka)
        prompt = build_summary_prompt(document_text, self.prompts_dir)
        logger.info("Generando resumen científico")
        return self.client.generate(prompt)

    def run(self, input_path: Path, output_dir: Path) -> Dict[str, str]:
        if not input_path.exists():
            raise FileNotFoundError(f"No existe: {input_path}")

        markdown_dir = output_dir / "markdown"
        json_dir = output_dir / "json"
        markdown_dir.mkdir(parents=True, exist_ok=True)
        json_dir.mkdir(parents=True, exist_ok=True)

        document_text = input_path.read_text(encoding="utf-8", errors="ignore")

        corrected = self.correct(document_text)
        (markdown_dir / "corrected_document.md").write_text(corrected, encoding="utf-8")
        logger.info("Documento corregido guardado")

        summary = self.summarize(corrected)
        (markdown_dir / "final_summary.md").write_text(summary, encoding="utf-8")
        logger.info("Resumen guardado")

        result = {"corrected_document": corrected, "summary": summary}
        (json_dir / "final_output.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return result

    def _chunk(self, text: str) -> list:
        if len(text) <= self.max_chars:
            return [text]
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.max_chars
            # Cortar en salto de línea para no partir oraciones
            if end < len(text):
                newline = text.rfind("\n", start, end)
                if newline > start:
                    end = newline
            chunks.append(text[start:end])
            start = end
        return chunks
