from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from src.evaluation.metrics import (
    compute_bertscore,
    compute_compression_ratio,
    compute_noise_reduction,
    compute_ocr_metrics,
    compute_rouge,
    compute_structural_preservation,
)

logger = logging.getLogger(__name__)


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


class DocumentEvaluator:
    def __init__(self, config):
        self.lang: str = getattr(config, "lang", "en")
        self.use_stemmer: bool = getattr(config, "use_stemmer", True)

    def evaluate_ocr(self, original: str, corrected: str) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        results.update(compute_ocr_metrics(original, corrected))
        results["Compression_Ratio"] = compute_compression_ratio(original, corrected)
        results.update(compute_noise_reduction(original, corrected))
        results.update(compute_structural_preservation(original, corrected))
        return results

    def evaluate_summary(
        self,
        reference: str,
        generated: str,
        include_bertscore: bool = True,
    ) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        results.update(compute_rouge(reference, generated, self.use_stemmer))
        if include_bertscore:
            results.update(compute_bertscore(reference, generated, self.lang))
        return results

    def full_evaluation(
        self,
        run_dir: Path,
        ground_truth_dir: Optional[Path] = None,
    ) -> pd.DataFrame:
        rows = []

        multimodal_path = run_dir / "30_multimodal" / "markdown" / "document_multimodal.md"
        corrected_path = run_dir / "40_llm" / "markdown" / "corrected_document.md"
        summary_path = run_dir / "40_llm" / "markdown" / "final_summary.md"

        row: Dict[str, Any] = {"run": run_dir.name}

        if multimodal_path.exists() and corrected_path.exists():
            original_text = load_text(multimodal_path)
            corrected_text = load_text(corrected_path)
            ocr_metrics = self.evaluate_ocr(original_text, corrected_text)
            row.update(ocr_metrics)
            logger.info("Métricas OCR calculadas")
        else:
            logger.warning("Faltan archivos para métricas OCR")

        #if ground_truth_dir and summary_path.exists():
        #    abstract_path = ground_truth_dir / "abstract.txt"
        #    if abstract_path.exists():
        #        reference = load_text(abstract_path)
        #        generated = load_text(summary_path)
        #        summary_metrics = self.evaluate_summary(reference, generated)
        #        row.update(summary_metrics)
        #        logger.info("Métricas de resumen calculadas")
        #    else:
        #        logger.warning(f"No existe ground truth: {abstract_path}")

        rows.append(row)
        return pd.DataFrame(rows)

    def save_results(self, results: pd.DataFrame, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)

        csv_path = output_dir / "csv" / "metrics.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(csv_path, index=False)

        json_path = output_dir / "json" / "metrics.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(results.to_dict(orient="records"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(f"Métricas guardadas en {output_dir}")
