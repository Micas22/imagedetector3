from __future__ import annotations

import io
import threading
from typing import Optional, Tuple

from PIL import Image

_adapter_lock = threading.Lock()
_processor = None
_model = None
_device = None
_load_error: Optional[str] = None
_structure_processor = None
_structure_model = None
_MIN_IMAGE_SIDE_PX = 120
_MIN_TABLE_AREA_RATIO = 0.30


def _load_model():
    global _processor, _model, _device, _load_error, _structure_processor, _structure_model
    if _processor is not None and _model is not None and _structure_processor is not None and _structure_model is not None:
        return _processor, _model, _device, _structure_processor, _structure_model, None
    if _load_error is not None:
        return None, None, None, None, None, _load_error

    with _adapter_lock:
        if _processor is not None and _model is not None and _structure_processor is not None and _structure_model is not None:
            return _processor, _model, _device, _structure_processor, _structure_model, None
        if _load_error is not None:
            return None, None, None, None, None, _load_error

        try:
            import torch
            from transformers import AutoImageProcessor, TableTransformerForObjectDetection
        except Exception as exc:
            _load_error = f"missing_dependencies_{type(exc).__name__.lower()}"
            return None, None, None, None, None, _load_error

        try:
            model_id = "microsoft/table-transformer-detection"
            _processor = AutoImageProcessor.from_pretrained(model_id)
            _model = TableTransformerForObjectDetection.from_pretrained(model_id)
            structure_model_id = "microsoft/table-transformer-structure-recognition"
            _structure_processor = AutoImageProcessor.from_pretrained(structure_model_id)
            _structure_model = TableTransformerForObjectDetection.from_pretrained(structure_model_id)
            _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            _model.to(_device)
            _model.eval()
            _structure_model.to(_device)
            _structure_model.eval()
            return _processor, _model, _device, _structure_processor, _structure_model, None
        except Exception as exc:
            _load_error = f"model_load_failed_{type(exc).__name__.lower()}"
            return None, None, None, None, None, _load_error


def _decode_image(image_bytes: bytes) -> Optional[Image.Image]:
    if not image_bytes:
        return None
    try:
        image = Image.open(io.BytesIO(image_bytes))
        return image.convert("RGB")
    except Exception:
        return None


def _looks_like_bar_chart(image: Image.Image) -> bool:
    try:
        import cv2
        import numpy as np
    except Exception:
        return False

    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    if h < 120 or w < 120:
        return False

    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=80,
        minLineLength=int(w * 0.4), maxLineGap=10,
    )
    if lines is None:
        return False

    baseline_y = None
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        if dx > w * 0.5 and dy < 5 and y1 > h * 0.6:
            baseline_y = y1
            break

    if baseline_y is None:
        return False

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bar_count = 0
    for cnt in contours:
        x, y, ww, hh = cv2.boundingRect(cnt)
        if ww < 15 or hh < h * 0.15:
            continue
        if hh / float(ww) < 1.3:
            continue
        if abs((y + hh) - baseline_y) > h * 0.12:
            continue
        bar_count += 1

    return bar_count >= 3


def _looks_like_line_chart(image: Image.Image) -> bool:
    try:
        import cv2
        import numpy as np
    except Exception:
        return False

    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    if h < _MIN_IMAGE_SIDE_PX or w < _MIN_IMAGE_SIDE_PX:
        return False

    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=max(50, int(min(h, w) * 0.1)),
        minLineLength=int(0.3 * w), maxLineGap=20,
    )
    if lines is None:
        return False

    horizontal = 0
    vertical = 0
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        if dx > 0 and dy / float(dx) < 0.1 and (y1 > 0.6 * h or y2 > 0.6 * h):
            horizontal += 1
        if dy > 0 and dx / float(dy) < 0.1 and (x1 < 0.4 * w or x2 < 0.4 * w):
            vertical += 1

    return horizontal >= 1 and vertical >= 1


def _looks_like_chart_image(image: Image.Image) -> bool:
    return _looks_like_bar_chart(image) or _looks_like_line_chart(image)


def _count_vertical_lines(image: Image.Image) -> int:
    try:
        import cv2
        import numpy as np
    except Exception:
        return 0

    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    if h < _MIN_IMAGE_SIDE_PX or w < _MIN_IMAGE_SIDE_PX:
        return 0

    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)
    if lines is None:
        return 0

    vertical = 0
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        if dy > dx * 2:
            vertical += 1
    return vertical


def _has_text(image: Image.Image) -> bool:
    try:
        import cv2
        import numpy as np
    except Exception:
        return False

    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    if h < _MIN_IMAGE_SIDE_PX or w < _MIN_IMAGE_SIDE_PX:
        return False

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    text_contours = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 10 or area > 1000:
            continue
        x, y, ww, hh = cv2.boundingRect(cnt)
        aspect = float(ww) / hh if hh > 0 else 0
        if 0.1 < aspect < 10 and ww > 5 and hh > 5:
            text_contours += 1
    return text_contours > 20


# Labels the TATR structure model uses that confirm tabular content.
# "table projected row header" and "table spanning cell" don't contain the simple
# substrings "table row"/"table column"/"table cell", hence the prefix matches.
_STRUCTURE_KEYWORDS = (
    "table row",
    "table column",
    "table cell",
    "table projected",  # table projected row header
    "table spanning",   # table spanning cell
    "table header",     # table column header (also caught by "table column", but explicit)
)


def classify_with_table_transformer(
    image_bytes: bytes,
    image_url: str = "",
    table_score_threshold: float = 0.9,
) -> Tuple[str, float, str]:

    processor, model, device, structure_processor, structure_model, load_error = _load_model()
    if load_error:
        return "normal", 0.0, load_error

    image = _decode_image(image_bytes)
    if image is None:
        return "normal", 0.0, "decode_failed"
    if min(image.width, image.height) < _MIN_IMAGE_SIDE_PX:
        return "normal", 0.0, "tatr_image_too_small"

    try:
        import torch
    except Exception:
        return "normal", 0.0, "missing_dependencies_torch"

    try:
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([[image.height, image.width]], device=device)
        results = processor.post_process_object_detection(
            outputs=outputs,
            threshold=0.0,
            target_sizes=target_sizes,
        )[0]

        id2label = getattr(model.config, "id2label", {}) or {}
        image_area = float(image.width * image.height) if image.width > 0 and image.height > 0 else 1.0

        # Collect ALL table detections, not just the single best-scoring one.
        # Pages with multiple independent tables would otherwise only get one shot at
        # validation — if the top candidate fails structure verification, the rest
        # never get checked. Also tracks the overall best for the final fallback reason string.
        best_table_score = 0.0
        best_label = ""
        table_candidates = []

        for score_tensor, label_tensor, box_tensor in zip(results["scores"], results["labels"], results["boxes"]):
            score = float(score_tensor.item())
            label_id = int(label_tensor.item())
            label_name = str(id2label.get(label_id, label_id)).lower()
            x1, y1, x2, y2 = [float(v.item()) for v in box_tensor]
            box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            area_ratio = box_area / image_area
            if "table" in label_name and score > 0.0:
                table_candidates.append((score, label_name, area_ratio, (x1, y1, x2, y2)))
            if "table" in label_name and score > best_table_score:
                best_table_score = score
                best_label = label_name

        # Sort by score desc, area desc as tiebreaker.
        table_candidates.sort(key=lambda c: (c[0], c[2]), reverse=True)

        # Minimum area ratio before we bother running structure recognition.
        # 0.15 * _MIN_TABLE_AREA_RATIO ≈ 0.045, small enough to catch tables that are
        # inset into full-page document scans (Image 1 pattern).
        _RELAXED_AREA_FLOOR = _MIN_TABLE_AREA_RATIO * 0.15

        for cand_score, cand_label, cand_area_ratio, cand_box in table_candidates:
            if cand_area_ratio < _RELAXED_AREA_FLOOR:
                continue

            cx1, cy1, cx2, cy2 = cand_box
            cropped = image.crop((cx1, cy1, cx2, cy2))
            if min(cropped.width, cropped.height) < _MIN_IMAGE_SIDE_PX:
                continue

            # --- Structure recognition ---
            structure_inputs = structure_processor(images=cropped, return_tensors="pt")
            structure_inputs = {k: v.to(device) for k, v in structure_inputs.items()}
            with torch.no_grad():
                structure_outputs = structure_model(**structure_inputs)

            structure_target_sizes = torch.tensor([[cropped.height, cropped.width]], device=device)
            structure_results = structure_processor.post_process_object_detection(
                outputs=structure_outputs,
                threshold=0.2,   # lowered from 0.35 — structure model is conservative on
                                 # striped/bordered tables; we prefer recall over precision here
                target_sizes=structure_target_sizes,
            )[0]

            structure_id2label = getattr(structure_model.config, "id2label", {}) or {}
            has_table_structure = any(
                any(kw in str(structure_id2label.get(int(t.item()), int(t.item()))).lower()
                    for kw in _STRUCTURE_KEYWORDS)
                for t in structure_results["labels"]
            )

            if has_table_structure:
                # Structure verification is the primary signal.
                # Do NOT re-gate on detector score — that's what was silently killing recall.
                score_reason = (
                    "tatr_structure_verified_high_detector_score"
                    if cand_score >= table_score_threshold
                    else "tatr_structure_verified_low_detector_score"
                )
                return "table", cand_score, f"{score_reason}_{cand_label}"

            # --- Heuristic fallback ---
            # Structure head missed. For rotated / low-quality / striped crops the structure
            # model can fail while the region is still clearly a table. Accept if area + text pass.
            # Area threshold is halved vs the global constant so inset tables aren't excluded.
            likely_table_by_heuristic = (
                cand_area_ratio >= _MIN_TABLE_AREA_RATIO * 0.5
                and _has_text(cropped)
                and (cand_score >= max(0.25, table_score_threshold * 0.25))
            )
            if likely_table_by_heuristic:
                score_reason = (
                    "tatr_structure_missing_but_heuristic_table"
                    if cand_score < table_score_threshold
                    else "tatr_structure_missing_but_heuristic_table_high_score"
                )
                return "table", cand_score, f"{score_reason}_{cand_label}"

            # Chart veto is per-candidate — one chart in a multi-table image shouldn't
            # block the other real tables from being validated.
            if _looks_like_chart_image(cropped):
                continue

        # No candidate passed validation.
        return "normal", best_table_score, f"tatr_no_candidate_validated_{best_label}"

    except Exception as exc:
        return "normal", 0.0, f"tatr_inference_failed_{type(exc).__name__.lower()}"