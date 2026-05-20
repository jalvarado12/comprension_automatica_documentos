from __future__ import annotations

from typing import Dict

from src.utils.text_utils import count_noise


def compute_ocr_metrics(gt_text: str, pred_text: str) -> Dict[str, float]:
    from jiwer import cer, wer
    if not gt_text:
        return {"WER": None, "CER": None, "error": "empty_ground_truth"}
    return {
        "WER": wer(gt_text, pred_text),
        "CER": cer(gt_text, pred_text),
    }


def compute_rouge(reference: str, generated: str, use_stemmer: bool = True) -> Dict[str, float]:
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=use_stemmer)
    scores = scorer.score(reference, generated)
    return {
        "ROUGE-1": scores["rouge1"].fmeasure,
        "ROUGE-L": scores["rougeL"].fmeasure,
    }


def compute_bertscore(reference: str, generated: str, lang: str = "en") -> Dict[str, float]:
    from bert_score import score as bertscore
    _, _, F1 = bertscore([generated], [reference], lang=lang, verbose=False)
    return {"BERTScore_F1": float(F1.mean())}


def compute_compression_ratio(original: str, corrected: str) -> float:
    if not original:
        return 0.0
    return len(corrected) / len(original)


def compute_structural_preservation(original: str, corrected: str) -> Dict[str, float]:
    original_headers = original.count("#")
    corrected_headers = corrected.count("#")
    original_tables = original.count("|")
    corrected_tables = corrected.count("|")

    return {
        "Header_Preservation": corrected_headers / max(original_headers, 1),
        "Table_Preservation": corrected_tables / max(original_tables, 1),
    }


def compute_noise_reduction(original: str, corrected: str) -> Dict[str, int]:
    noise_orig = count_noise(original)
    noise_corr = count_noise(corrected)
    return {
        "Noise_Original": noise_orig,
        "Noise_Corrected": noise_corr,
        "Noise_Reduction": noise_orig - noise_corr,
    }
