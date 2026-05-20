#!/usr/bin/env python3
"""
run_pipeline.py — Entrypoint del pipeline completo de comprensión de documentos científicos.

Uso:
    python run_pipeline.py --pdf data/raw/sample.pdf --run-name paper_run_001
    python run_pipeline.py --pdf data/raw/sample.pdf --run-name paper_run_001 --skip-llm
    python run_pipeline.py --pdf data/raw/sample.pdf --run-name paper_run_001 --only-metrics
"""

# Configuración de entorno ANTES de cualquier import de numpy/torch
from src.utils.platform_compat import configure_threading_env, add_windows_dll_dirs
configure_threading_env()
add_windows_dll_dirs()

import argparse
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Vision-Docs pipeline")
    parser.add_argument("--pdf", required=True, type=Path, help="Ruta al PDF de entrada")
    parser.add_argument("--run-name", required=True, type=str, help="Nombre de la corrida")
    parser.add_argument("--config", default="configs/pipeline.yaml", type=Path)
    parser.add_argument("--skip-ocr", action="store_true", help="Saltar etapas 1-4")
    parser.add_argument("--skip-florence", action="store_true", help="Saltar etapa 5")
    parser.add_argument("--skip-llm", action="store_true", help="Saltar etapa 7")
    parser.add_argument("--skip-metrics", action="store_true", help="Saltar etapa 8")
    parser.add_argument("--only-metrics", action="store_true", help="Solo calcular métricas")
    return parser.parse_args()


def main():
    args = parse_args()

    from src.utils.config import load_config
    from src.utils.paths import find_project_root, resolve_run_dir

    cfg = load_config(args.config)
    project_root = find_project_root()
    run_dir = resolve_run_dir(project_root, args.run_name)

    logger.info(f"PROJECT_ROOT: {project_root}")
    logger.info(f"RUN_DIR: {run_dir}")

    # ── ETAPA 1-4: PDF → páginas → layout → OCR → tablas ────────────────────
    if not args.skip_ocr and not args.only_metrics:
        from src.pdf.pdf_renderer import PDFRenderer
        from src.layout.layout_detector import LayoutDetector
        from src.tables.tatr_detector import TATRDetector
        from src.tables.table_pipeline import TableMarkdownPipeline
        from src.ocr.ocr_engine import OCREngine
        from src.pipeline.region_merger import RegionMerger
        from src.io.manifest_writer import ManifestWriter
        from src.utils.geometry import crop_image

        logger.info("=== ETAPA 1: PDF → Páginas ===")
        renderer = PDFRenderer(cfg.pdf)
        page_images = renderer.render_and_preprocess(
            args.pdf,
            pages_dir=run_dir / "01_page_images",
            preprocessed_dir=run_dir / "02_preprocessed_pages",
        )

        logger.info("=== ETAPA 2-3: Layout + TATR + Merge ===")
        layout_detector = LayoutDetector(cfg.layout)
        layout_detector.load()
        tatr_detector = TATRDetector(cfg.tatr)
        tatr_detector.load()
        merger = RegionMerger(cfg.merge)
        manifest_writer = ManifestWriter()

        all_regions = []
        for page_num, page_bgr in enumerate(page_images, start=1):
            layout_regions = layout_detector.detect(page_bgr, page_num)
            tatr_regions = tatr_detector.detect(page_bgr, page_num)
            merged = merger.merge(layout_regions, tatr_regions)
            all_regions.extend(merged)

        layout_detector.unload()
        tatr_detector.unload()

        manifest_path = run_dir / "11_manifests" / "detections.csv"
        manifest_writer.save(all_regions, manifest_path)
        manifest_writer.save_jsonl(all_regions, manifest_path.with_suffix(".jsonl"))
        logger.info(f"Regiones detectadas: {len(all_regions)}")

        logger.info("=== ETAPA 3: OCR sobre bloques de texto ===")
        ocr_engine = OCREngine(cfg.ocr)
        ocr_engine.load()
        ocr_json_dir = run_dir / "08_ocr_json"
        ocr_txt_dir = run_dir / "09_ocr_txt"
        ocr_json_dir.mkdir(parents=True, exist_ok=True)
        ocr_txt_dir.mkdir(parents=True, exist_ok=True)

        text_regions = [r for r in all_regions if r.kind in ("text", "title", "list")]
        for region_idx, region in enumerate(text_regions):
            page_bgr = page_images[region.page_num - 1]
            crop = crop_image(page_bgr, region.bbox)
            ocr_engine.process_region(
                crop, region.page_num, region_idx,
                list(region.bbox), ocr_json_dir, ocr_txt_dir,
            )
        ocr_engine.unload()
        logger.info(f"OCR completado: {len(text_regions)} bloques")

        logger.info("=== ETAPA 4: Tablas → Markdown ===")
        detections_df = manifest_writer.load(manifest_path)
        tables_df = detections_df[detections_df["kind"] == "table"].copy()

        if not tables_df.empty:
            table_pipeline = TableMarkdownPipeline(cfg.tatr, run_dir)
            table_pipeline.load()
            table_pipeline.run(tables_df, run_dir / "01_page_images")
            table_pipeline.unload()
        else:
            logger.info("Sin tablas detectadas")

    # ── ETAPA 5: Florence-2 para figuras ─────────────────────────────────────
    if not args.skip_florence and not args.only_metrics:
        from src.captioning.florence_model import FlorenceModel
        from src.captioning.ocr_context_builder import OCRContextBuilder
        from src.captioning.contextual_captioner import ContextualCaptioner
        from src.io.figure_builder import FigureBuilder
        from src.io.manifest_writer import ManifestWriter

        logger.info("=== ETAPA 5: Florence-2 Captioning ===")
        florence = FlorenceModel(cfg.florence)
        florence.load()

        context_builder = OCRContextBuilder(cfg.captioning)
        captioner = ContextualCaptioner(florence, context_builder)

        figure_builder = FigureBuilder()
        detections_df = ManifestWriter().load(run_dir / "11_manifests" / "detections.csv")
        page_images_by_page = figure_builder.load_page_images(run_dir / "01_page_images")
        figures = figure_builder.build_figures_from_detections(
            detections_df, page_images_by_page, run_dir / "06_figures"
        )

        ocr_blocks = OCRContextBuilder.load_ocr_blocks_from_json(run_dir / "08_ocr_json")
        captioner.caption_all(figures, ocr_blocks, run_dir / "12_image2text")
        florence.unload()

    # ── ETAPA 6: Documento multimodal ────────────────────────────────────────
    if not args.only_metrics:
        from src.document.loaders import DocumentLoader
        from src.document.builder import MultimodalDocumentBuilder

        logger.info("=== ETAPA 6: Documento Multimodal ===")
        loader = DocumentLoader()
        ocr_blocks = loader.load_ocr_blocks(run_dir / "08_ocr_json")
        captions = loader.load_captions(
            run_dir / "12_image2text" / "contextual_image2text_results.json"
        )
        tables = loader.load_tables(run_dir / "22_tables_markdown")

        builder = MultimodalDocumentBuilder(args.run_name)
        document = builder.build(ocr_blocks, captions, tables)
        builder.save_json(document, run_dir / "30_multimodal" / "json" / "document_multimodal.json")
        markdown = builder.to_markdown(document)
        builder.save_markdown(markdown, run_dir / "30_multimodal" / "markdown" / "document_multimodal.md")

    # ── ETAPA 7: Corrección LLM + Resumen ────────────────────────────────────
    if not args.skip_llm and not args.only_metrics:
        from src.llm.gemini_client import GeminiClient
        from src.llm.corrector import LLMCorrector

        logger.info("=== ETAPA 7: LLM Correction + Summary ===")
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY no está definida en .env")

        gemini = GeminiClient(api_key=api_key, model=cfg.llm.model)
        corrector = LLMCorrector(gemini, cfg.llm)
        corrector.run(
            input_path=run_dir / "30_multimodal" / "markdown" / "document_multimodal.md",
            output_dir=run_dir / "40_llm",
        )

    # ── ETAPA 8: Métricas ────────────────────────────────────────────────────
    if not args.skip_metrics:
        from src.evaluation.evaluator import DocumentEvaluator

        logger.info("=== ETAPA 8: Evaluación ===")
        evaluator = DocumentEvaluator(cfg.evaluation)
        results = evaluator.full_evaluation(
            run_dir=run_dir,
            ground_truth_dir=project_root / "data" / "ground_truth",
        )
        evaluator.save_results(results, run_dir / "50_metrics")

    logger.info("=== PIPELINE COMPLETADO ===")


if __name__ == "__main__":
    main()
