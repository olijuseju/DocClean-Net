"""
gui/inference_core.py
======================
Tkinter-free logic backing gui/inference_gui.py.

Bundles the post-processing parameters and the single function that turns
a cached raw U-Net output into the final preview (white point + speckle
denoise + stroke thickening, same order as scripts/run_pipeline.py).

Kept separate from the GUI module on purpose: tkinter/ttkbootstrap are not
importable in a headless container (no display), so anything that imports
them can't be exercised by the automated test suite — same constraint
gui/digitize_gui.py already has, and the same reason it has no tests. This
module has no such dependency and is fully unit-tested.
"""

from dataclasses import dataclass

import numpy as np

from inference.predict import _apply_white_point, _estimate_white_point
from scripts.thicken_strokes import remove_small_dots, thicken_strokes


@dataclass
class PostprocessParams:
    """Bundle of the GUI's real-time-adjustable parameters.

    Attributes:
        white_point_auto: if True, the white point is estimated per-image
            from the histogram mode (same "auto" logic as predict_image).
            If False, white_point_value is used for every image instead.
        white_point_value: fixed white point in [1, 255], used only when
            white_point_auto is False.
        min_dot_area: components with area (px) strictly below this are
            erased as speckle noise. 0 disables denoising.
        ink_threshold: gray level below which a pixel counts as ink, for
            building the binary mask denoising is computed on.
        thicken_amount: erosion kernel radius in px for stroke thickening.
            0 disables thickening.
    """

    white_point_auto: bool = True
    white_point_value: int = 200
    min_dot_area: int = 3
    ink_threshold: int = 128
    thicken_amount: int = 1


def apply_postprocessing(
    raw_output: np.ndarray, params: PostprocessParams
) -> np.ndarray:
    """Turn a cached raw U-Net output into the final post-processed image.

    Pipeline: white point -> denoise (speckle removal) -> thicken. This
    mirrors scripts/run_pipeline.process_image(), except the white point
    is applied here explicitly (rather than inside predict_image) so the
    GUI can cache the raw network output once per image and recompute
    this cheap final stage instantly whenever a slider moves.

    Args:
        raw_output (np.ndarray): U-Net output BEFORE white-point
            normalisation, shape (H, W), dtype uint8 (i.e. predict_image()
            called with white_point=None).
        params: the current slider/control state.

    Returns:
        np.ndarray: final grayscale image, shape (H, W), dtype uint8,
            0=ink, 255=paper.
    """
    white_point = (
        _estimate_white_point(raw_output)
        if params.white_point_auto
        else params.white_point_value
    )
    result = _apply_white_point(raw_output, white_point)

    if params.min_dot_area > 0:
        result = remove_small_dots(
            result, min_area=params.min_dot_area, ink_threshold=params.ink_threshold
        )
    if params.thicken_amount > 0:
        result = thicken_strokes(result, amount=params.thicken_amount)
    return result


if __name__ == "__main__":
    pass
