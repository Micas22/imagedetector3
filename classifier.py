import csv
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
    CACHE_FILENAME,
    CLASSIFIER_VERSION,
    DEFAULT_TABLE_SCORE_THRESHOLD,
    IMAGE_EXTENSIONS,
    TABLE_URL_HINT_WORDS,
    TABLE_WORDS,
    _ocr_lock,
    _ocr_mkldnn_disabled,
    _thread_local,
)
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

    # Adaptive threshold highlights dark ruling lines on light backgrounds.
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15,
        4,
    )

    # Extract long horizontal and vertical line candidates.
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

    # Count distinct line runs to avoid treating a single box as a table.
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

    # Step 1: cluster rows
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
        return 0.0  # not enough rows to be a table

    # Step 2: build global column anchors
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

    # Step 3: check how consistently rows hit these anchors
    row_hits = []

    for row in rows:
        hits = 0
        for anchor in col_anchors:
            for x in row:
                if abs(x - anchor) <= col_tolerance:
                    hits += 1
                    break
        row_hits.append(hits / len(col_anchors))

    # Step 4: stability = consistency across rows
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
        # Newer PaddleOCR `predict()` often returns dict-like results.
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

        # Legacy PaddleOCR `ocr()` returns list-based line tuples.
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
    # Keep the original function signature so webapp/crawler callers remain compatible.
    label, score, reason = classify_with_table_transformer(
        image_bytes=image_bytes,
        image_url=image_url,
        table_score_threshold=table_score_threshold,
    )
    if label == "table" and score <= table_score_threshold:
        # Preserve adapter-provided table decision even if numeric score is slightly below threshold.
        # The adapter already performs its own structure verification.
        return "normal", score, f"{reason}_guarded_threshold"
    if flag_uncertain and label != "table" and score >= max(0.0, table_score_threshold * 0.9):
        return "uncertain", score, f"{reason}_near_threshold"
    return label, score, reason

    img = safe_to_cv(image_bytes)
    if img is None:
        return "normal", 0.0, "decode_failed"

    h, w = img.shape[:2]
    if h < 120 or w < 120:
        return "normal", 0.05, "too_small"
    aspect_ratio = float(max(h, w)) / float(max(1, min(h, w)))
    if turbo_mode and aspect_ratio >= 6.5 and min(h, w) < 280:
        # Extreme thin banners/icons are very unlikely to be useful tables.
        return "normal", 0.02, "turbo_extreme_aspect_skip"
    if turbo_mode:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if float(np.std(gray)) < 14.0:
            return "normal", 0.02, "turbo_low_variance_skip"
    grid_region_signal = _detect_table_grid_signal(img)

    # Large images are much slower for OCR; downscale while preserving layout.
    if turbo_mode:
        max_ocr_side = 560 if fast_mode else 1000
    else:
        max_ocr_side = 800 if fast_mode else 1600
    max_side = max(h, w)
    if max_side > max_ocr_side:
        scale = float(max_ocr_side) / float(max_side)
        resized_w = max(1, int(w * scale))
        resized_h = max(1, int(h * scale))
        img = cv2.resize(img, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]

    ocr_engine = _get_ocr_engine()
    if ocr_engine is None:
        return "normal", 0.0, "paddleocr_not_installed"

    # PaddleOCR runtime can hang/crash when accessed concurrently across threads.
    # Serialize OCR calls even if image downloading/classification orchestration is parallel.
    with _ocr_lock:
        try:
            ocr_result = _run_ocr(ocr_engine, img)
        except Exception as first_error:
            # Some Paddle builds crash with oneDNN enabled on CPU. Retry once after disabling it.
            if _is_mkldnn_runtime_error(first_error):
                try:
                    _disable_mkldnn_if_possible()
                    _reset_ocr_engine()
                    ocr_engine = _get_ocr_engine()
                    if ocr_engine is None:
                        return "normal", 0.0, "paddleocr_not_installed"
                    ocr_result = _run_ocr(ocr_engine, img)
                except Exception:
                    return "normal", 0.0, "ocr_failed_mkldnn_runtime"
            else:
                return "normal", 0.0, "ocr_failed"

    lines = _flatten_ocr_lines(ocr_result)
    if not lines:
        return "normal", 0.05, "ocr_no_text"

    min_conf = 0.45
    lines = [entry for entry in lines if entry[2] >= min_conf]
    if not lines:
        return "normal", 0.06, "ocr_low_confidence"

    centers_x: List[float] = []
    centers_y: List[float] = []
    alnum_tokens = 0
    numericish_tokens = 0
    table_keyword_hits = 0
    alpha_chars = 0
    uppercase_alpha_chars = 0
    long_alpha_tokens = 0
    for box, text, _conf in lines:
        x_vals = [pt[0] for pt in box]
        y_vals = [pt[1] for pt in box]
        centers_x.append((min(x_vals) + max(x_vals)) * 0.5)
        centers_y.append((min(y_vals) + max(y_vals)) * 0.5)
        text_l = text.lower()
        tokens = re.findall(r"[a-zA-Z0-9$€£%.,:/-]+", text_l)
        for token in tokens:
            alnum_tokens += 1
            if re.search(r"\d", token):
                numericish_tokens += 1
            if token.isalpha() and len(token) >= 6:
                long_alpha_tokens += 1
            if token in TABLE_WORDS:
                table_keyword_hits += 1
        for ch in text:
            if ch.isalpha():
                alpha_chars += 1
                if ch.isupper():
                    uppercase_alpha_chars += 1

    row_tolerance = max(8.0, h * 0.018)
    col_tolerance = max(8.0, w * 0.018)
    row_count = _cluster_axis(centers_y, tolerance=row_tolerance)
    col_count = _cluster_axis(centers_x, tolerance=col_tolerance)
    line_count = len(lines)
    numeric_ratio = numericish_tokens / max(1, alnum_tokens)
    uppercase_ratio = uppercase_alpha_chars / max(1, alpha_chars)
    long_alpha_ratio = long_alpha_tokens / max(1, alnum_tokens)
    alignment_signal = alignment_consistency(centers_x, tolerance=col_tolerance)
    col_stability = column_stability(
        centers_x,
        centers_y,
        col_tolerance,
        row_tolerance
    )

    grid_likeness = min(1.0, (row_count * col_count) / max(1.0, line_count * 1.4))
    structure_signal = min(1.0, (max(0, row_count - 1) * max(0, col_count - 1)) / 24.0)
    density_signal = min(1.0, line_count / 45.0)
    keyword_signal = min(1.0, table_keyword_hits / 3.0)
    numeric_signal = min(1.0, numeric_ratio / 0.55)
    stability_signal = col_stability

    score = (
        0.28 * numeric_signal +
        0.22 * keyword_signal +
        0.18 * structure_signal +
        0.10 * grid_likeness +
        0.10 * alignment_signal +
        0.10 * stability_signal +
        0.02 * density_signal
    )
    score += 0.12 * grid_region_signal
    parsed_image = urlparse(image_url or "")
    image_path_l = (parsed_image.path or "").lower()
    image_name_l = Path(image_path_l).name
    has_table_url_hint = any(
        hint_word in image_name_l or hint_word in image_path_l
        for hint_word in TABLE_URL_HINT_WORDS
    )
    if has_table_url_hint:
        # Strongly prefer "table/tabela"-named assets, but never auto-classify.
        score += 0.22
    score = max(0.0, min(1.0, score))

    # Require stronger geometric/table-like structure before labeling as table.
    has_strong_structure = (
        row_count >= 4
        and col_count >= 3
        and line_count >= 10
        and grid_likeness >= 0.42
        and structure_signal >= 0.22
    )
    has_table_content = (
        numeric_ratio >= 0.18
        or table_keyword_hits >= 2
        or (numeric_ratio >= 0.12 and numericish_tokens >= 4 and col_stability >= 0.6)
        or (has_table_url_hint and numeric_ratio >= 0.08)
    )
    # Allow mixed-content images (table + surrounding paragraph text) to pass when
    # table content signals are clear but full-frame grid structure is diluted.
    has_partial_table_structure = (
        row_count >= 3
        and col_count >= 3
        and line_count >= 7
        and col_stability >= 0.52
        and structure_signal >= 0.08
        and (numeric_ratio >= 0.12 or table_keyword_hits >= 2 or grid_region_signal >= 0.44)
    )
    has_hint_boosted_structure = (
        has_table_url_hint
        and row_count >= 2
        and col_count >= 2
        and line_count >= 6
        and col_stability >= 0.42
        and structure_signal >= 0.05
    )
    looks_like_poster_banner = (
        uppercase_ratio >= 0.68
        and long_alpha_ratio >= 0.30
        and numericish_tokens <= 3
        and table_keyword_hits == 0
        and grid_region_signal < 0.12
        and structure_signal < 0.20
    )
    is_table = (
        score > table_score_threshold
        and has_table_content
        and not (looks_like_poster_banner and not has_strong_structure)
        and (col_stability > (0.25 if has_table_url_hint else 0.35) or grid_region_signal >= 0.58)
        and (has_strong_structure or has_partial_table_structure or has_hint_boosted_structure)
    )
    uncertain_margin = max(0.003, table_score_threshold * 0.08)
    is_uncertain = (
        flag_uncertain
        and not is_table
        and score <= table_score_threshold
        and (table_score_threshold - score) <= uncertain_margin
        and has_table_content
        and col_stability > (0.22 if has_table_url_hint else 0.3)
    )
    reason = (
        f"ocr_lines{line_count}_rows{row_count}_cols{col_count}_"
        f"numratio{numeric_ratio:.3f}_keywords{table_keyword_hits}_"
        f"align{alignment_signal:.3f}_grid{grid_region_signal:.3f}_"
        f"upratio{uppercase_ratio:.3f}_longalpha{long_alpha_ratio:.3f}_"
        f"urlhint{int(has_table_url_hint)}_"
        f"score{score:.4f}_threshold{table_score_threshold:.4f}"
    )
    if is_table:
        return "table", score, reason
    if is_uncertain:
        return "uncertain", score, f"{reason}_near_threshold"
    return "normal", score, reason


# ---------------------------------------------------------------------------
# Table image saving
# ---------------------------------------------------------------------------

def save_table_image(save_dir: Path, image_url: str, image_bytes: bytes) -> Optional[Path]:
    parsed = urlparse(image_url)
    ext = Path(parsed.path).suffix.lower()
    if ext not in IMAGE_EXTENSIONS:
        ext = ".jpg"
    digest = hashlib.sha1(image_url.encode("utf-8")).hexdigest()[:14]
    name = f"table_{digest}{ext}"
    out_path = save_dir / name
    try:
        out_path.write_bytes(image_bytes)
        return out_path
    except OSError:
        return None


# ---------------------------------------------------------------------------
# Classification cache I/O
# ---------------------------------------------------------------------------

def load_classification_cache(
    cache_path: Path,
    fast_mode: bool,
    turbo_mode: bool = False,
    table_score_threshold: float = DEFAULT_TABLE_SCORE_THRESHOLD,
    flag_uncertain: bool = False,
) -> dict:
    cache: dict = {}
    if not cache_path.exists():
        return cache
    try:
        with cache_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row:
                    continue
                if row.get("classifier_version") != CLASSIFIER_VERSION:
                    continue
                if row.get("threshold") != f"{table_score_threshold:.4f}":
                    continue
                if row.get("fast_mode", "0") != ("1" if fast_mode else "0"):
                    continue
                if row.get("turbo_mode", "0") != ("1" if turbo_mode else "0"):
                    continue
                if row.get("flag_uncertain", "0") != ("1" if flag_uncertain else "0"):
                    continue
                image_hash = row.get("image_hash", "")
                if not image_hash:
                    continue
                try:
                    score = float(row.get("score", "0"))
                except ValueError:
                    continue
                cache[image_hash] = (row.get("label", "normal"), score, row.get("reason", "cache_miss"))
    except OSError:
        return {}
    return cache


def save_classification_cache(
    cache_path: Path,
    cache: dict,
    fast_mode: bool,
    turbo_mode: bool = False,
    table_score_threshold: float = DEFAULT_TABLE_SCORE_THRESHOLD,
    flag_uncertain: bool = False,
) -> None:
    try:
        with cache_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "image_hash",
                    "label",
                    "score",
                    "reason",
                    "classifier_version",
                    "threshold",
                    "fast_mode",
                    "turbo_mode",
                    "flag_uncertain",
                ]
            )
            for image_hash, (label, score, reason) in cache.items():
                writer.writerow(
                    [
                        image_hash,
                        label,
                        f"{score:.6f}",
                        reason,
                        CLASSIFIER_VERSION,
                        f"{table_score_threshold:.4f}",
                        "1" if fast_mode else "0",
                        "1" if turbo_mode else "0",
                        "1" if flag_uncertain else "0",
                    ]
                )
    except OSError:
        return
