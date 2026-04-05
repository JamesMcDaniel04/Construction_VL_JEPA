"""Basic server-side media quality checks for pilot capture flows."""

from __future__ import annotations

import io

import numpy as np


def assess_image_capture(data: bytes) -> list[str]:
    from PIL import Image

    image = Image.open(io.BytesIO(data)).convert("RGB")
    array = np.asarray(image, dtype=np.float32)
    grayscale = (0.299 * array[..., 0]) + (0.587 * array[..., 1]) + (0.114 * array[..., 2])
    brightness = float(grayscale.mean())
    grad_y, grad_x = np.gradient(grayscale)
    sharpness = float(np.var(grad_x) + np.var(grad_y))

    hints: list[str] = []
    if brightness < 45.0:
        hints.append(
            "The panel image is too dark. Move closer or increase lighting before retrying."
        )
    if sharpness < 120.0:
        hints.append(
            "The panel image appears blurry. Hold the camera steady and retake the shot."
        )
    if min(image.size) < 300:
        hints.append(
            "The panel is framed too loosely or at low resolution. Move closer and fill the frame."
        )
    return hints
