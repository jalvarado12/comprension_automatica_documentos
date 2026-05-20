from src.evaluation.metrics import (
    compute_compression_ratio,
    compute_structural_preservation,
    compute_noise_reduction,
)


def test_compression_ratio_same():
    assert compute_compression_ratio("abc", "abc") == 1.0


def test_compression_ratio_empty_original():
    assert compute_compression_ratio("", "algo") == 0.0


def test_structural_preservation_identical():
    text = "# Header\n| col1 | col2 |"
    result = compute_structural_preservation(text, text)
    assert result["Header_Preservation"] == 1.0
    assert result["Table_Preservation"] == 1.0


def test_noise_reduction():
    original = "hello @@@ world !!!"
    corrected = "hello world"
    result = compute_noise_reduction(original, corrected)
    assert result["Noise_Reduction"] > 0
