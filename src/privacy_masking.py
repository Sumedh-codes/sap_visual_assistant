
import re
from pathlib import Path

from PIL import Image, ImageDraw
from paddleocr import PaddleOCR


# Fields whose associated VALUES may identify a person.
# These labels are configurable for your SAP screenshots.
PERSON_FIELD_LABELS = {
    "vendor name",
    "contact name",
    "vendor contact",
    "employee name",
    "full name",
    "created by",
    "employee",
    "person name",
    "contact person",
    "user name",
}

PHONE_FIELD_LABELS = {
    "phone",
    "phone number",
    "mobile",
    "mobile number",
    "telephone",
    "contact number",
}

# Detect email addresses anywhere in recognized text.
EMAIL_PATTERN = re.compile(
    r"\b[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}\b",
    re.IGNORECASE,
)

# Detect explicitly formatted phone numbers, or 10-digit numbers
# beginning with 6-9 (commonly used for Indian mobile numbers).
PHONE_PATTERN = re.compile(
    r"(?<!\w)(?:\+?\d{1,3}[\s().\-]*)?"
    r"(?:\d[\s().\-]*){10,13}(?!\w)"
)

PLAIN_INDIAN_MOBILE_PATTERN = re.compile(r"(?<!\d)[6-9]\d{9}(?!\d)")


def _as_list(value):
    """Convert PaddleOCR array-like outputs into ordinary Python lists."""
    if value is None:
        return []
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value) if not isinstance(value, str) else [value]


def _extract_detections(engine, image_path):
    """
    Run PaddleOCR locally and return text with pixel bounding boxes.

    Supports the PaddleOCR 3.x predict() result format used by
    the project's OCR pipeline.
    """
    results = engine.predict(str(image_path))
    detections = []

    for result in results or []:
        payload = getattr(result, "json", None)
        if callable(payload):
            payload = payload()

        if not isinstance(payload, dict):
            payload = result if isinstance(result, dict) else {}

        if isinstance(payload.get("res"), dict):
            payload = payload["res"]

        texts = _as_list(payload.get("rec_texts", []))
        boxes = _as_list(payload.get("rec_boxes", []))

        # Fallback for OCR result formats exposing polygon boxes.
        if not boxes:
            boxes = _as_list(payload.get("dt_polys", []))

        for index, text in enumerate(texts):
            if text is None or not str(text).strip():
                continue

            if index >= len(boxes):
                continue

            box = _as_list(boxes[index])

            # rec_boxes normally contains [x1, y1, x2, y2].
            # Polygon boxes contain multiple [x, y] points.
            if len(box) >= 4 and not isinstance(box[0], (list, tuple)):
                coords = [float(x) for x in box[:4]]
                x1, y1, x2, y2 = coords
            elif box and isinstance(box[0], (list, tuple)):
                points = [
                    [float(point[0]), float(point[1])]
                    for point in box
                    if len(point) >= 2
                ]
                if not points:
                    continue

                x1 = min(point[0] for point in points)
                y1 = min(point[1] for point in points)
                x2 = max(point[0] for point in points)
                y2 = max(point[1] for point in points)
            else:
                continue

            detections.append({
                "text": str(text).strip(),
                "box": [x1, y1, x2, y2],
            })

    return detections


def _normalize(text):
    """Normalize a label for matching."""
    return re.sub(r"\s+", " ", text.casefold()).strip(" :\t")


def _is_phone_number(text):
    """
    Identify likely phone numbers without treating every numeric
    SAP identifier as a phone number.
    """
    text = text.strip()

    # Stronger evidence: international prefix or separators.
    if text.startswith("+") or any(c in text for c in "()- "):
        digits = re.sub(r"\D", "", text)
        return 10 <= len(digits) <= 13

    # Plain Indian mobile numbers.
    return bool(PLAIN_INDIAN_MOBILE_PATTERN.fullmatch(text))


def _find_sensitive_regions(detections):
    """
    Identify email addresses, phone numbers and values associated
    with personal-information labels.

    Returns boxes and category names only. Raw sensitive values
    are not written to a log or audit file.
    """
    regions = []

    def add_region(box, category):
        regions.append({
            "box": box,
            "category": category,
        })

    # First: directly recognizable emails and phone numbers.
    for detection in detections:
        text = detection["text"]
        box = detection["box"]

        if EMAIL_PATTERN.search(text):
            add_region(box, "email")
            continue

        if _is_phone_number(text) or PHONE_PATTERN.fullmatch(text):
            add_region(box, "phone")
            continue

    # Second: find values associated with sensitive field labels.
    for label_detection in detections:
        label = _normalize(label_detection["text"])
        label_box = label_detection["box"]

        is_person_label = label in PERSON_FIELD_LABELS
        is_phone_label = label in PHONE_FIELD_LABELS

        if not (is_person_label or is_phone_label):
            continue

        lx1, ly1, lx2, ly2 = label_box
        label_height = max(1, ly2 - ly1)

        candidates = []

        for value_detection in detections:
            if value_detection is label_detection:
                continue

            value_text = value_detection["text"].strip()
            vx1, vy1, vx2, vy2 = value_detection["box"]

            # Never mask another field label as the value.
            if _normalize(value_text) in (
                PERSON_FIELD_LABELS | PHONE_FIELD_LABELS
            ):
                continue

            # Only accept text to the right, aligned on the same row.
            same_row = (
                vx1 >= lx2 - 3
                and abs(
                    (vy1 + vy2) / 2 - (ly1 + ly2) / 2
                ) <= max(label_height, vy2 - vy1) * 0.8
            )

            if same_row:
                distance = max(0, vx1 - lx2)
                candidates.append((distance, value_detection))

        if candidates:
            candidates.sort(key=lambda item: item[0])
            value_detection = candidates[0][1]

            if is_phone_label:
                add_region(value_detection["box"], "phone")
            else:
                add_region(value_detection["box"], "person")

    # Remove duplicate regions.
    unique = []
    seen = set()

    for region in regions:
        key = (
            tuple(round(v) for v in region["box"]),
            region["category"],
        )
        if key not in seen:
            seen.add(key)
            unique.append(region)

    return unique


def mask_sensitive_regions(
    image_path,
    output_path,
    ocr_engine=None,
    padding=4,
):
    """
    Detect and irreversibly mask likely personal information.

    Args:
        image_path: Original screenshot path.
        output_path: Path for the sanitized PNG.
        ocr_engine: Optional existing PaddleOCR engine to reuse.
        padding: Extra pixels around detected text.

    Returns:
        A summary containing output path, mask count and categories.

    Important:
        If OCR or masking fails, this function raises an exception.
        The caller must not fall back to sending the original image.
    """
    image_path = Path(image_path).resolve()
    output_path = Path(output_path).resolve()

    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Use an existing engine if supplied; otherwise create one.
    if ocr_engine is None:
        ocr_engine = PaddleOCR(
            lang="en",
            enable_mkldnn=False,
        )

    # OCR and detection happen locally, before any external API call.
    detections = _extract_detections(ocr_engine, image_path)
    regions = _find_sensitive_regions(detections)

    # Always save a separate PNG copy, even when no regions are found.
    # Never modify the original screenshot.
    with Image.open(image_path) as source:
        image = source.convert("RGB")

    draw = ImageDraw.Draw(image)
    width, height = image.size
    masked_categories = []

    for region in regions:
        x1, y1, x2, y2 = region["box"]

        # Expand the region slightly to cover text edges.
        left = max(0, int(x1) - padding)
        top = max(0, int(y1) - padding)
        right = min(width - 1, int(x2) + padding)
        bottom = min(height - 1, int(y2) + padding)

        if right < left or bottom < top:
            continue

        # Solid opaque rectangle, not a reversible blur.
        draw.rectangle(
            [left, top, right, bottom],
            fill=(0, 0, 0),
        )
        masked_categories.append(region["category"])

    # Flatten to a new PNG without original metadata.
    # Do not copy the source image's metadata or layers.
    image.save(output_path, format="PNG", optimize=False)

    # Verify the saved file can be reopened.
    with Image.open(output_path) as check:
        check.verify()

    return {
        "output_path": str(output_path),
        "mask_count": len(masked_categories),
        "categories": sorted(set(masked_categories)),
        "ocr_detection_count": len(detections),
    }
