"""Image clean-up before OCR: loading, skew estimation/correction, light denoising."""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def load_image(data: bytes, filename: str = "") -> np.ndarray:
    """Decode an uploaded image or the first page of a PDF into an RGB array."""
    if filename.lower().endswith(".pdf") or data[:4] == b"%PDF":
        import fitz  # PyMuPDF

        with fitz.open(stream=data, filetype="pdf") as pdf:
            pix = pdf[0].get_pixmap(dpi=200)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            return cv2.cvtColor(img, cv2.COLOR_RGBA2RGB) if pix.n == 4 else img.copy()
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise ValueError("Could not decode image")
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


def rotate(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate about the centre (degrees, counter-clockwise), keeping size, white border."""
    if abs(angle) < 1e-3:
        return image
    h, w = image.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    border = (255, 255, 255) if image.ndim == 3 else 255
    return cv2.warpAffine(image, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=border)


def estimate_skew(image: np.ndarray, max_angle: float = 3.0) -> float:
    """Projection-profile skew estimate. Returns the angle that *straightens* the page.

    Text rows produce sharp peaks in the horizontal projection only when they are
    level, so we pick the rotation that maximises the profile's 'sharpness'.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    scale = 800 / max(gray.shape)
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray
    # text = 255, background = 0, so rotation padding (0) adds nothing to the profile
    _, binary = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    def score(a: float) -> float:
        h, w = binary.shape
        m = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0)
        rot = cv2.warpAffine(binary, m, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
        return float(np.sum(np.diff(rot.sum(axis=1, dtype=np.float64)) ** 2))

    coarse = np.arange(-max_angle, max_angle + 1e-6, 0.25)
    best = max(coarse, key=score)
    fine = np.arange(best - 0.25, best + 0.25 + 1e-6, 0.05)
    return round(float(max(fine, key=score)), 2)


def deskew(image: np.ndarray) -> Tuple[np.ndarray, float]:
    angle = estimate_skew(image)
    return rotate(image, angle), angle


def denoise(image: np.ndarray) -> np.ndarray:
    """Mild median filter; strong denoising hurts thin table digits more than it helps."""
    return cv2.medianBlur(image, 3)
