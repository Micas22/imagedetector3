import hashlib
import os
import re
import warnings
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import urlparse

import cv2
import numpy as np

from constants import (
    CLASSIFIER_VERSION,
    DEFAULT_TABLE_SCORE_THRESHOLD,
    IMAGE_EXTENSIONS,
    TABLE_URL_HINT_WORDS,
    TABLE_WORDS,
    _ocr_lock,
    _ocr_mkldnn_disabled,
    _thread_local,
)
from database import load_classification_cache, save_classification_cache  # noqa: F401 (re-exported)
from table_transformer_adapter import classify_with_table_transformer

try:
    from paddleocr import PaddleOCR
except Exception:  # Optional dependency.
    PaddleOCR = None


# ---------------------------------------------------------------------------
# Image validation
# ---------------------------------------------------------------------------

def _is_valid_image_bytes(content: bytes) -> bool:
    if not content:
        return False
    try:
        arr = np.frombuffer(content, dtype=np.uint8)
        if arr.size == 0:
            return False
        decoded = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        return decoded is not None
    except Exception:
        return False


def safe_to_cv(image_bytes: bytes) -> Optional[np.ndarray]:
    try:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        if arr.size == 0:
            return None
        decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return decoded
    except Exception:
        return None


# ---------------------------------------------------------------------------
# OCR engine management
# ---------------------------------------------------------------------------

def _get_ocr_engine():
    if PaddleOCR is None:
        return None
    engine = getattr(_thread_local, "ocr_engine", None)
    if engine is None:
        init_variants = [
            {"lang": "en", "enable_mkldnn": False, "use_textline_orientation": True},
            {"lang": "en", "enable_mkldnn": False, "use_angle_cls": True},
            {"lang": "en", "enable_mkldnn": False},
            {"lang": "en", "use_textline_orientation": True},
            {"lang": "en", "use_angle_cls": True},
            {"lang": "en"},
            {},
        ]
        for kwargs in init_variants:
            try:
                engine = PaddleOCR(**kwargs)
                break
            except (TypeError, ValueError):
                engine = None
        _thread_local.ocr_engine = engine
    return engine


def _reset_ocr_engine() -> None:
    _thread_local.ocr_engine = None


def _disable_mkldnn_if_possible() -> None:
    """Disable MKLDNN/oneDNN to avoid Paddle runtime crashes on some Windows setups."""
    global _ocr_mkldnn_disabled
    if _ocr_mkldnn_disabled:
        return
    os.environ["FLAGS_use_mkldnn"] = "0"
    try:
        import paddle
        paddle.set_flags({"FLAGS_use_mkldnn": False})
    except Exception:
        pass
    import constants as constants
    constants._ocr_mkldnn_disabled = True


def _run_ocr(ocr_engine, img: np.ndarray):
    if hasattr(ocr_engine, "predict"):
        return ocr_engine.predict(img)
    if hasattr(ocr_engine, "ocr"):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*Please use `predict` instead\..*",
                category=DeprecationWarning,
            )
            return ocr_engine.ocr(img)
    raise RuntimeError("ocr_engine_missing_methods")


def _is_mkldnn_runtime_error(exc: Exception) -> bool:
    msg = str(exc)
    return (
        "ConvertPirAttribute2RuntimeAttribute" in msg
        or "onednn_instruction.cc" in msg
        or "oneDNN" in msg
    )


# ---------------------------------------------------------------------------
# Geometry / grid helpers
# ---------------------------------------------------------------------------

def _cluster_axis(values: List[float], tolerance: float) -> int:
    if not values:
        return 0
    sorted_values = sorted(values)
    clusters = 1
    anchor = sorted_values[0]
    for value in sorted_values[1:]:
        if abs(value - anchor) > tolerance:
            clusters += 1
            anchor = value
    return clusters


def _detect_table_grid_signal(img: np.ndarray) -> float:
    """Estimate whether image contains a table-like ruled region."""
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except Exception:
        return 0.0

    h, w = gray.shape[:2]
    if h < 120 or w < 120:
        return 0.0

    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4,
    )

    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(12, w // 24), 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(12, h // 24)))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

    intersections = cv2.bitwise_and(horizontal, vertical)
    h_pixels = float(np.count_nonzero(horizontal))
    v_pixels = float(np.count_nonzero(vertical))
    x_pixels = float(np.count_nonzero(intersections))
    image_pixels = float(max(1, h * w))

    h_ratio = h_pixels / image_pixels
    v_ratio = v_pixels / image_pixels
    x_ratio = x_pixels / image_pixels

    h_proj = (np.sum(horizontal > 0, axis=1) > max(6, w * 0.02)).astype(np.uint8)
    v_proj = (np.sum(vertical > 0, axis=0) > max(6, h * 0.02)).astype(np.uint8)
    h_runs = int(np.count_nonzero((h_proj[1:] == 1) & (h_proj[:-1] == 0)))
    v_runs = int(np.count_nonzero((v_proj[1:] == 1) & (v_proj[:-1] == 0)))

    run_signal = min(1.0, (max(0, h_runs - 1) * max(0, v_runs - 1)) / 20.0)
    density_signal = min(1.0, (h_ratio + v_ratio) / 0.07)
    intersection_signal = min(1.0, x_ratio / 0.0035)

    return max(0.0, min(1.0, 0.35 * run_signal + 0.35 * density_signal + 0.30 * intersection_signal))


def alignment_consistency(centers_x, tolerance):
    centers_x = sorted(centers_x)
    clusters = []

    for x in centers_x:
        placed = False
        for c in clusters:
            if abs(x - c[0]) < tolerance:
                c.append(x)
                placed = True
                break
        if not placed:
            clusters.append([x])

    if not clusters:
        return 0.0

    sizes = [len(c) for c in clusters]
    return max(sizes) / sum(sizes)


def column_stability(centers_x, centers_y, col_tolerance, row_tolerance):
    """
    Strong column consistency:
    Checks if columns align at consistent x positions across rows.
    """
    if not centers_x or not centers_y:
        return 0.0

    points = sorted(zip(centers_y, centers_x))
    rows = []
    current = [points[0][1]]
    anchor_y = points[0][0]

    for y, x in points[1:]:
        if abs(y - anchor_y) <= row_tolerance:
            current.append(x)
        else:
            rows.append(current)
            current = [x]
            anchor_y = y

    if current:
        rows.append(current)

    if len(rows) < 3:
        return 0.0

    all_x = sorted(centers_x)
    col_anchors = []

    for x in all_x:
        placed = False
        for c in col_anchors:
            if abs(x - c) <= col_tolerance:
                placed = True
                break
        if not placed:
            col_anchors.append(x)

    if len(col_anchors) < 2:
        return 0.0

    row_hits = []

    for row in rows:
        hits = 0
        for anchor in col_anchors:
            for x in row:
                if abs(x - anchor) <= col_tolerance:
                    hits += 1
                    break
        row_hits.append(hits / len(col_anchors))

    mean = np.mean(row_hits)
    std = np.std(row_hits)

    stability = mean * (1.0 - std)
    return float(max(0.0, min(1.0, stability)))


# ---------------------------------------------------------------------------
# OCR result normalisation
# ---------------------------------------------------------------------------

def _flatten_ocr_lines(raw_ocr) -> List[Tuple[List[List[float]], str, float]]:
    items: List[Tuple[List[List[float]], str, float]] = []
    if not raw_ocr:
        return items

    def _coerce_quad_points(poly) -> Optional[List[List[float]]]:
        if poly is None:
            return None
        pts = poly.tolist() if hasattr(poly, "tolist") else poly
        if not isinstance(pts, (list, tuple)) or len(pts) < 4:
            return None
        out: List[List[float]] = []
        for pt in list(pts)[:4]:
            coords = pt.tolist() if hasattr(pt, "tolist") else pt
            if not isinstance(coords, (list, tuple)) or len(coords) < 2:
                return None
            out.append([float(coords[0]), float(coords[1])])
        return out if len(out) == 4 else None

    chunks = raw_ocr if isinstance(raw_ocr, list) else [raw_ocr]
    for chunk in chunks:
        if not chunk:
            continue
        if isinstance(chunk, dict) or hasattr(chunk, "keys"):
            payload = chunk if isinstance(chunk, dict) else dict(chunk)
            polys = payload.get("rec_polys") or payload.get("dt_polys") or []
            texts = payload.get("rec_texts") or []
            scores = payload.get("rec_scores")
            limit = min(len(polys), len(texts))
            for i in range(limit):
                points = _coerce_quad_points(polys[i])
                text = str(texts[i]).strip()
                if scores is None:
                    conf = 1.0
                else:
                    try:
                        conf = float(scores[i])
                    except (TypeError, ValueError, IndexError):
                        conf = 0.0
                if text and points:
                    items.append((points, text, conf))
            continue

        for line in chunk:
            if not line or len(line) < 2:
                continue
            box, data = line[0], line[1]
            points = _coerce_quad_points(box)
            if not points:
                continue
            if not isinstance(data, (list, tuple)) or len(data) < 2:
                continue
            text = str(data[0]).strip()
            try:
                conf = float(data[1])
            except (TypeError, ValueError):
                conf = 0.0
            if text:
                items.append((points, text, conf))
    return items


# ---------------------------------------------------------------------------
# Main classification entry point
# ---------------------------------------------------------------------------

def classify_image(
    image_bytes: bytes,
    image_url: str = "",
    fast_mode: bool = False,
    turbo_mode: bool = False,
    table_score_threshold: float = DEFAULT_TABLE_SCORE_THRESHOLD,
    flag_uncertain: bool = False,
) -> Tuple[str, float, str]:
    # Route table detection to Microsoft Table Transformer (detection model).
    label, score, reason = classify_with_table_transformer(
        image_bytes=image_bytes,
        image_url=image_url,
        table_score_threshold=table_score_threshold,
    )
    # NOTE: Do not re-gate on score <= table_score_threshold here.
    # classify_with_table_transformer uses structure verification as its primary
    # signal and intentionally returns "table" with low raw detector scores when
    # structure or heuristics confirm it. Re-applying the threshold here would
    # silently kill those structure-verified detections.
    if flag_uncertain and label != "table" and score >= max(0.0, table_score_threshold * 0.9):
        return "uncertain", score, f"{reason}_near_threshold"
    return label, score, reason