"""
Tests for gui/inference_core.py — the tkinter-free logic behind the
interactive GUI. No display or tkinter/ttkbootstrap import required.
"""

import numpy as np

from gui.inference_core import PostprocessParams, apply_postprocessing


def _paper_with_dot_and_stroke() -> np.ndarray:
    """Synthetic raw output: paper at 233 (typical MSE-sigmoid ceiling),
    a 1px isolated dot, and a 5px vertical stroke."""
    img = np.full((40, 40), 233, dtype=np.uint8)
    img[5, 5] = 0  # isolated dot: should be removed by denoising
    img[20:25, 20] = 0  # real stroke: should survive and get thickened
    return img


def test_postprocess_params_defaults() -> None:
    params = PostprocessParams()
    assert params.white_point_auto is True
    assert params.min_dot_area == 3
    assert params.thicken_amount == 1


def test_apply_postprocessing_auto_white_point_saturates_paper() -> None:
    img = _paper_with_dot_and_stroke()
    params = PostprocessParams(white_point_auto=True, min_dot_area=0, thicken_amount=0)
    result = apply_postprocessing(img, params)
    assert result[0, 0] == 255  # paper saturates to pure white


def test_apply_postprocessing_manual_white_point_is_used_when_auto_disabled() -> None:
    img = _paper_with_dot_and_stroke()
    params_manual = PostprocessParams(
        white_point_auto=False, white_point_value=250, min_dot_area=0, thicken_amount=0
    )
    result = apply_postprocessing(img, params_manual)
    # 233 stretched with white_point=250 does not fully saturate.
    assert result[0, 0] < 255


def test_apply_postprocessing_removes_isolated_dot() -> None:
    img = _paper_with_dot_and_stroke()
    params = PostprocessParams(min_dot_area=3, thicken_amount=0)
    result = apply_postprocessing(img, params)
    assert result[5, 5] == 255


def test_apply_postprocessing_min_dot_area_zero_keeps_isolated_dot() -> None:
    img = _paper_with_dot_and_stroke()
    params = PostprocessParams(min_dot_area=0, thicken_amount=0)
    result = apply_postprocessing(img, params)
    assert result[5, 5] < 255


def test_apply_postprocessing_thicken_amount_zero_is_noop_on_stroke_width() -> None:
    img = _paper_with_dot_and_stroke()
    params_off = PostprocessParams(min_dot_area=0, thicken_amount=0)
    params_on = PostprocessParams(min_dot_area=0, thicken_amount=2)
    result_off = apply_postprocessing(img, params_off)
    result_on = apply_postprocessing(img, params_on)
    assert (result_on < 255).sum() > (result_off < 255).sum()


def test_apply_postprocessing_full_pipeline_returns_uint8_same_shape() -> None:
    img = _paper_with_dot_and_stroke()
    result = apply_postprocessing(img, PostprocessParams())
    assert result.shape == img.shape
    assert result.dtype == np.uint8
