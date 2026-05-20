from src.document.parsers import parse_ocr_block, parse_caption_block, parse_table_block


def test_parse_ocr_block_basic():
    block = {
        "page_num": 1,
        "region": {"bbox": [0, 0, 100, 50]},
        "ocr_lines": [{"text": "Hola"}, {"text": "mundo"}],
    }
    result = parse_ocr_block(block)
    assert result["type"] == "text"
    assert result["page"] == 1
    assert "Hola" in result["content"]


def test_parse_ocr_block_missing_region():
    block = {"page_num": 2, "ocr_lines": [{"text": "x"}]}
    result = parse_ocr_block(block)
    assert result["bbox"] == [0, 0, 0, 0]


def test_parse_caption_block():
    block = {
        "page_num": 3,
        "bbox": [10, 20, 200, 150],
        "contextual_description": "Una figura científica",
    }
    result = parse_caption_block(block)
    assert result["type"] == "figure"
    assert result["content"] == "Una figura científica"


def test_parse_table_block_empty_markdown():
    block = {"page_num": 1, "bbox": [0, 0, 0, 0], "markdown": None}
    result = parse_table_block(block)
    assert result["content"] == ""
