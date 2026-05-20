from src.utils.geometry import compute_iou, clamp_box


def test_iou_perfect_overlap():
    box = (0, 0, 100, 100)
    assert compute_iou(box, box) == 1.0


def test_iou_no_overlap():
    assert compute_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_partial():
    iou = compute_iou((0, 0, 10, 10), (5, 5, 15, 15))
    assert 0.0 < iou < 1.0


def test_clamp_box_within_bounds():
    assert clamp_box((10, 10, 50, 50), w=100, h=100) == (10, 10, 50, 50)


def test_clamp_box_out_of_bounds():
    x1, y1, x2, y2 = clamp_box((-5, -5, 200, 200), w=100, h=100)
    assert x1 == 0 and y1 == 0 and x2 == 100 and y2 == 100
