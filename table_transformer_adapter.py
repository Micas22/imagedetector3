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

    # Edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Detect lines
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=80,
        minLineLength=int(w * 0.4),
        maxLineGap=10,
    )

    if lines is None:
        return False

    baseline_y = None

    # Find strong horizontal baseline near bottom
    for line in lines:
        x1, y1, x2, y2 = line[0]

        dx = abs(x2 - x1)
        dy = abs(y2 - y1)

        if dx > w * 0.5 and dy < 5:
            if y1 > h * 0.6:
                baseline_y = y1
                break

    if baseline_y is None:
        return False

    # Threshold bars
    _, thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    bar_count = 0

    for cnt in contours:
        x, y, ww, hh = cv2.boundingRect(cnt)

        if ww < 15:
            continue

        if hh < h * 0.15:
            continue

        aspect = hh / float(ww)

        # vertical bar shape
        if aspect < 1.3:
            continue

        # anchored near baseline
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
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=max(50, int(min(h, w) * 0.1)), minLineLength=int(0.3 * w), maxLineGap=20)
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
        if dy > dx * 2:  # more vertical than horizontal
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
    return text_contours > 20  # threshold for likely text presence


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
        best_table_score = 0.0
        best_label = ""
        best_box = None

        image_area = float(image.width * image.height) if image.width > 0 and image.height > 0 else 1.0
        best_table_area_ratio = 0.0
        for score_tensor, label_tensor, box_tensor in zip(results["scores"], results["labels"], results["boxes"]):
            score = float(score_tensor.item())
            label_id = int(label_tensor.item())
            label_name = str(id2label.get(label_id, label_id)).lower()
            x1, y1, x2, y2 = [float(v.item()) for v in box_tensor]
            box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            area_ratio = box_area / image_area
            if "table" in label_name and score > best_table_score:
                best_table_score = score
                best_label = label_name
                best_table_area_ratio = area_ratio
                best_box = (x1, y1, x2, y2)

        # Historically this code hard-gated structure verification on detector confidence.
        # That causes real tables with moderate detector scores (~0.5) to be labeled "normal".
        # Instead, use the detector to propose a region, then let structure recognition decide.
        #
        # Still keep some coarse sanity checks to avoid validating tiny noise.
        # Require: we have a usable box, minimum crop size, and some minimum area ratio.
        relaxed_area_ratio_ok = best_table_area_ratio >= (_MIN_TABLE_AREA_RATIO * 0.30)

        if best_box is not None and relaxed_area_ratio_ok:
            x1, y1, x2, y2 = best_box
            cropped = image.crop((x1, y1, x2, y2))

            if min(cropped.width, cropped.height) >= _MIN_IMAGE_SIDE_PX:
                structure_inputs = structure_processor(images=cropped, return_tensors="pt")
                structure_inputs = {k: v.to(device) for k, v in structure_inputs.items()}
                with torch.no_grad():
                    structure_outputs = structure_model(**structure_inputs)

                structure_target_sizes = torch.tensor([[cropped.height, cropped.width]], device=device)
                structure_results = structure_processor.post_process_object_detection(
                    outputs=structure_outputs,
                    threshold=0.35,  # permissive for recall
                    target_sizes=structure_target_sizes,
                )[0]

                structure_id2label = getattr(structure_model.config, "id2label", {}) or {}
                has_table_structure = False
                for s_label_tensor in structure_results["labels"]:
                    s_label_id = int(s_label_tensor.item())
                    s_label_name = str(structure_id2label.get(s_label_id, s_label_id)).lower()
                    if (
                        "table row" in s_label_name
                        or "table column" in s_label_name
                        or "table cell" in s_label_name
                    ):
                        has_table_structure = True
                        break

                if has_table_structure:
                    # Key change: do NOT depend on detector score crossing the threshold.
                    # Structure verification is the primary signal for returning "table".
                    score_reason = (
                        "tatr_structure_verified_high_detector_score"
                        if best_table_score >= table_score_threshold
                        else "tatr_structure_verified_low_detector_score"
                    )
                    return "table", best_table_score, f"{score_reason}_{best_label}"

                # No structure recognized.
                # Many real-world tables (especially rotated/low-quality crops) may fail the strict
                # structure head but still look like tables: sizable region + lots of small text.
                likely_table_by_heuristic = (
                    best_table_area_ratio >= _MIN_TABLE_AREA_RATIO
                    and _has_text(cropped)
                    and (best_table_score >= max(0.25, table_score_threshold * 0.25))
                )
                if likely_table_by_heuristic:
                    score_reason = (
                        "tatr_structure_missing_but_heuristic_table"
                        if best_table_score < table_score_threshold
                        else "tatr_structure_missing_but_heuristic_table_high_score"
                    )
                    return "table", best_table_score, f"{score_reason}_{best_label}"

                # Secondary safeguard to avoid chart false positives.
                if _looks_like_chart_image(cropped):
                    return "normal", best_table_score, f"tatr_detected_{best_label}_chart_veto_no_structure"

                return "normal", best_table_score, f"tatr_detected_{best_label}_no_structure"

        # If we couldn't validate via structure, fall back safely.
        return "normal", best_table_score, f"tatr_detected_{best_label}_no_box_no_validation"


    except Exception as exc:
        return "normal", 0.0, f"tatr_inference_failed_{type(exc).__name__.lower()}"
