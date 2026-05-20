from __future__ import annotations

from pathlib import Path


def build_ocr_correction_prompt(document_text: str, prompts_dir: Path) -> str:
    template_path = prompts_dir / "ocr_correction.txt"
    if template_path.exists():
        template = template_path.read_text(encoding="utf-8")
        return template.replace("{{DOCUMENT}}", document_text)
    # Fallback inline si no existe el archivo de template
    return f"""Eres un experto en comprensión de papers científicos.

Tu tarea es:
1. Corregir errores OCR contextualmente.
2. Preservar términos científicos.
3. Mantener tablas markdown.
4. Mantener captions de figuras.
5. NO resumir.
6. NO reorganizar.
7. NO inventar contenido.

DOCUMENTO:

{document_text}
"""


def build_summary_prompt(corrected_document: str, prompts_dir: Path) -> str:
    template_path = prompts_dir / "summary.txt"
    if template_path.exists():
        template = template_path.read_text(encoding="utf-8")
        return template.replace("{{DOCUMENT}}", corrected_document)
    return f"""Genera un resumen científico técnico del siguiente documento.

Debe incluir:
1. Objetivo
2. Metodología
3. Resultados principales
4. Conclusiones

El resumen debe ser académico, técnico, coherente,
y basado únicamente en el contenido proporcionado.

DOCUMENTO:

{corrected_document}
"""
