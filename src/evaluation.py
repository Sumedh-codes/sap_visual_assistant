
import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

from jsonschema import validate, ValidationError


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

REFERENCE_PATH = BASE_DIR / "data" / "reference" / "sap_manual_reference_output.json"

PIPELINES = {
    "vision_only": {
        "directory": BASE_DIR / "outputs",
        "schema": BASE_DIR / "configuration_files" / "schema.json",
    },
    "ocr_vision": {
        "directory": BASE_DIR / "ocr_outputs" / "json",
        "schema": BASE_DIR / "configuration_files" / "schema_ocr.json",
    },
}

REPORT_DIR = BASE_DIR / "evaluation_results"


# ---------------------------------------------------------
# GENERAL HELPERS
# ---------------------------------------------------------

def normalize(value):
    """Normalize text for case-insensitive, whitespace-insensitive matching."""
    if value is None:
        return ""

    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)

    text = str(value).strip().casefold()
    text = re.sub(r"\s+", " ", text)
    return text


def read_json(path):
    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def first_value(data, keys):
    """Return the first non-empty value among common field names."""
    if not isinstance(data, dict):
        return None

    normalized_keys = {normalize(key) for key in keys}

    for key, value in data.items():
        if normalize(key) in normalized_keys and value is not None:
            return value

    return None



def get_screens(data):
    # The reference file stores screenshot filenames as keys
    # inside the "screenshots" dictionary.
    if isinstance(data, dict) and isinstance(data.get("screenshots"), dict):
        return [
            {"image_file": filename, **screen_data}
            for filename, screen_data in data["screenshots"].items()
        ]

    # Keep support for other possible reference formats.
    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ("screens", "results", "annotations"):
            if isinstance(data.get(key), list):
                return data[key]

        return [data]

    raise ValueError("Unsupported reference JSON structure")



def screen_filename(screen):
    """Get the screenshot filename or ID from a reference record."""
    value = first_value(screen, [
        "image_file", "filename", "file_name", "screenshot",
        "image", "screen_id", "id"
    ])

    if value:
        return Path(str(value)).name

    return ""



def screen_id(screen, index):
    filename = screen.get("image_file")

    if filename:
        return Path(str(filename)).stem

    return f"reference_{index:02d}"



def find_result(directory, expected_id):
    """
    Prefer an exact filename match.
    Fall back to a prefixed duplicate ending in __<expected_id>.
    Never count multiple copies of the same screenshot.
    """
    if not directory.exists():
        return None

    exact = directory / f"{expected_id}.json"
    if exact.is_file():
        return exact

    candidates = sorted(directory.glob(f"*__{expected_id}.json"))

    if len(candidates) == 1:
        return candidates[0]

    if len(candidates) > 1:
        print(
            f"WARNING: Multiple prefixed outputs for {expected_id}; "
            f"using {candidates[0].name}"
        )
        return candidates[0]

    return None


# ---------------------------------------------------------
# SCREEN-LEVEL FIELDS
# ---------------------------------------------------------

def get_title(data):
    return first_value(data, [
        "screen_title", "title", "page_title", "window_title"
    ])


def get_transaction_code(data):
    return first_value(data, [
        "transaction_code", "tcode", "transaction", "transaction_id"
    ])


# ---------------------------------------------------------
# ELEMENT EXTRACTION
# ---------------------------------------------------------

def get_elements(data):
    """
    Find the screenshot's element list using common schema names.
    Returns an empty list if the expected list cannot be identified.
    """
    if isinstance(data, list):
        return data

    if not isinstance(data, dict):
        return []

    for key in ("elements", "fields", "items", "components", "controls"):
        value = first_value(data, [key])

        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]

    return []


def element_label(element):
    return first_value(element, [
        "label", "field_label", "name", "element_name", "caption"
    ])


def element_value(element):
    return first_value(element, [
        "value", "field_value", "text_value", "content"
    ])


def element_text(element):
    return first_value(element, [
        "text", "message", "description", "visible_text"
    ])


def element_type(element):
    return first_value(element, [
        "type", "element_type", "control_type", "category"
    ])


def element_uncertain(element):
    """Identify common uncertainty indicators."""
    for key in ("uncertain", "is_uncertain", "unreadable", "is_unreadable"):
        value = first_value(element, [key])

        if isinstance(value, bool):
            if value:
                return True

        elif normalize(value) in {"true", "yes", "uncertain", "unreadable"}:
            return True

    for key in ("uncertainty", "uncertain_fields", "confidence_status"):
        value = first_value(element, [key])

        if isinstance(value, str) and normalize(value) in {
            "uncertain", "unreadable", "unknown", "ambiguous"
        }:
            return True

        if isinstance(value, list) and value:
            return True

    status = normalize(first_value(element, ["status", "readability"]))
    if status in {"uncertain", "unreadable", "ambiguous", "illegible"}:
        return True

    return False


def element_key(element):
    """
    Define element identity using label first, then type/text/value.
    This prevents bounding boxes from influencing the scores.
    """
    label = normalize(element_label(element))
    kind = normalize(element_type(element))
    text = normalize(element_text(element))

    if label:
        return ("label", label)

    if text:
        return ("text", kind, text)

    value = normalize(element_value(element))
    if value:
        return ("value", kind, value)

    return None


def element_pair(element):
    label = normalize(element_label(element))
    value = normalize(element_value(element))

    if label and value:
        return (label, value)

    return None


def element_signature(element):
    """
    Signature for detecting exact duplicate elements.
    Bounding boxes are deliberately excluded.
    """
    return (
        normalize(element_type(element)),
        normalize(element_label(element)),
        normalize(element_value(element)),
        normalize(element_text(element)),
    )


# ---------------------------------------------------------
# METRIC CALCULATIONS
# ---------------------------------------------------------

def safe_accuracy(correct, total):
    return correct / total if total else None


def score_screen(reference, prediction):
    ref_title = normalize(get_title(reference))
    pred_title = normalize(get_title(prediction))

    ref_code_raw = get_transaction_code(reference)
    pred_code_raw = get_transaction_code(prediction)

    # An absent transaction code is represented by a blank value.
    ref_code = normalize(ref_code_raw)
    pred_code = normalize(pred_code_raw)

    title_applicable = bool(ref_title)
    code_applicable = True

    title_correct = (
        title_applicable and ref_title == pred_title
    )
    code_correct = ref_code == pred_code

    ref_elements = get_elements(reference)
    pred_elements = get_elements(prediction)

    ref_keys = [
        element_key(element)
        for element in ref_elements
        if element_key(element) is not None
    ]

    pred_keys = [
        element_key(element)
        for element in pred_elements
        if element_key(element) is not None
    ]

    ref_counter = Counter(ref_keys)
    pred_counter = Counter(pred_keys)

    matched_elements = sum(
        (ref_counter & pred_counter).values()
    )

    true_positive = matched_elements
    false_positive = sum(pred_counter.values()) - true_positive
    false_negative = sum(ref_counter.values()) - true_positive

    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive else None
    )

    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative else None
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None
        and recall is not None
        and precision + recall > 0
        else (
            0.0 if precision == 0 or recall == 0 else None
        )
    )

    # Match field-value pairs using normalized labels and values.
    ref_pairs = Counter(
        pair for element in ref_elements
        if (pair := element_pair(element)) is not None
    )

    pred_pairs = Counter(
        pair for element in pred_elements
        if (pair := element_pair(element)) is not None
    )

    matched_pairs = sum((ref_pairs & pred_pairs).values())
    expected_pairs = sum(ref_pairs.values())

    association_accuracy = (
        matched_pairs / expected_pairs if expected_pairs else None
    )

    # A predicted pair absent from the reference is a potential
    # unsupported pair, not automatically a proven hallucination.
    unsupported_pairs = sum(
        (pred_pairs - ref_pairs).values()
    )

    # Uncertainty handling: count reference elements explicitly marked
    # uncertain/unreadable, and check whether the prediction also marks
    # those same elements uncertain/unreadable.
    uncertain_reference = [
        element for element in ref_elements
        if element_uncertain(element)
    ]

    predicted_by_key = {}
    for element in pred_elements:
        key = element_key(element)
        if key is not None:
            predicted_by_key.setdefault(key, []).append(element)

    uncertainty_correct = 0
    for element in uncertain_reference:
        key = element_key(element)
        candidates = predicted_by_key.get(key, [])

        if any(element_uncertain(candidate) for candidate in candidates):
            uncertainty_correct += 1

    uncertainty_accuracy = (
        uncertainty_correct / len(uncertain_reference)
        if uncertain_reference else None
    )

    return {
        "title_correct": int(bool(title_correct)),
        "title_applicable": int(title_applicable),
        "transaction_code_correct": int(bool(code_correct)),
        "transaction_code_applicable": int(code_applicable),
        "reference_element_count": len(ref_keys),
        "predicted_element_count": len(pred_keys),
        "matched_elements": matched_elements,
        "element_precision": precision,
        "element_recall": recall,
        "element_f1": f1,
        "expected_field_value_pairs": expected_pairs,
        "matched_field_value_pairs": matched_pairs,
        "field_value_association_accuracy": association_accuracy,
        "potential_unsupported_pairs": unsupported_pairs,
        "uncertain_reference_elements": len(uncertain_reference),
        "correctly_flagged_uncertain_elements": uncertainty_correct,
        "uncertainty_handling_accuracy": uncertainty_accuracy,
    }


# ---------------------------------------------------------
# SCHEMA VALIDATION AND PIPELINE EVALUATION
# ---------------------------------------------------------

def load_schema(path):
    if not path.is_file():
        print(f"WARNING: Schema not found: {path}")
        return None

    try:
        return read_json(path)
    except Exception as exc:
        print(f"WARNING: Could not read schema {path}: {exc}")
        return None


def evaluate_pipeline(name, config, references):
    directory = config["directory"]
    schema = load_schema(config["schema"])

    rows = []
    parsed_outputs = 0
    schema_valid_outputs = 0

    for index, reference in enumerate(references, start=1):
        identifier = screen_id(reference, index)
        result_path = find_result(directory, identifier)
        print(f"Looking for: {identifier}.json")
        print(f"  Folder: {directory}")
        print(f"  Found: {result_path}")    

        row = {
            "pipeline": name,
            "screenshot": identifier,
            "output_file": str(result_path) if result_path else "",
            "processing_success": 0,
            "json_valid": 0,
            "schema_valid": 0,
            "error": "",
        }

        if result_path is None:
            row["error"] = "Output file missing"
            rows.append(row)
            continue

        try:
            prediction = read_json(result_path)
            parsed_outputs += 1
            row["json_valid"] = 1
            row["processing_success"] = 1

        except Exception as exc:
            row["error"] = f"JSON parsing failed: {exc}"
            rows.append(row)
            continue

        if schema is not None:
            try:
                validate(instance=prediction, schema=schema)
                schema_valid_outputs += 1
                row["schema_valid"] = 1
            except ValidationError as exc:
                row["error"] = f"Schema validation failed: {exc.message}"
            except Exception as exc:
                row["error"] = f"Schema validation error: {exc}"

        try:
            metrics = score_screen(reference, prediction)
            row.update(metrics)
        except Exception as exc:
            row["error"] = (
                (row["error"] + " | ") if row["error"] else ""
            ) + f"Metric calculation failed: {exc}"

        rows.append(row)

    total = len(references)

    summary = {
        "pipeline": name,
        "reference_screenshots": total,
        "output_files_found": sum(bool(row["output_file"]) for row in rows),
        "processing_successes": sum(row["processing_success"] for row in rows),
        "processing_success_rate": safe_accuracy(
            sum(row["processing_success"] for row in rows), total
        ),
        "json_parse_successes": parsed_outputs,
        "json_parse_success_rate": safe_accuracy(parsed_outputs, total),
        "schema_valid_outputs": schema_valid_outputs,
        "schema_valid_rate": (
            safe_accuracy(schema_valid_outputs, total)
            if schema is not None else None
        ),
        "screen_title_accuracy": safe_accuracy(
            sum(row.get("title_correct", 0) for row in rows),
            sum(row.get("title_applicable", 0) for row in rows),
        ),
        "transaction_code_accuracy": safe_accuracy(
            sum(row.get("transaction_code_correct", 0) for row in rows),
            sum(row.get("transaction_code_applicable", 0) for row in rows),
        ),
    }

    for metric in (
        "element_precision",
        "element_recall",
        "element_f1",
        "field_value_association_accuracy",
        "uncertainty_handling_accuracy",
    ):
        values = [
            row[metric] for row in rows
            if isinstance(row.get(metric), (int, float))
        ]
        summary[metric + "_macro_average"] = (
            sum(values) / len(values) if values else None
        )

    summary["matched_elements_total"] = sum(
        row.get("matched_elements", 0) for row in rows
    )
    summary["reference_elements_total"] = sum(
        row.get("reference_element_count", 0) for row in rows
    )
    summary["predicted_elements_total"] = sum(
        row.get("predicted_element_count", 0) for row in rows
    )
    summary["field_value_pairs_expected_total"] = sum(
        row.get("expected_field_value_pairs", 0) for row in rows
    )
    summary["field_value_pairs_matched_total"] = sum(
        row.get("matched_field_value_pairs", 0) for row in rows
    )
    summary["potential_unsupported_pairs_total"] = sum(
        row.get("potential_unsupported_pairs", 0) for row in rows
    )

    return rows, summary


# ---------------------------------------------------------
# REPORT GENERATION
# ---------------------------------------------------------

def format_metric(value):
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f} ({value * 100:.2f}%)"
    return str(value)


def write_csv(path, rows):
    if not rows:
        return

    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)

    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path, summaries, total_screens):
    lines = [
        "# SAP Screenshot Extraction Evaluation",
        "",
        f"Reference screenshots: **{total_screens}**",
        "",
        "Evaluation is performed locally using Python. "
        "Bounding-box IoU is excluded.",
        "",
        "## Aggregate comparison",
        "",
        "| Metric | Vision-only | OCR + vision |",
        "|---|---:|---:|",
    ]

    metrics = [
        ("Processing success rate", "processing_success_rate"),
        ("JSON parse success rate", "json_parse_success_rate"),
        ("Schema-valid JSON rate", "schema_valid_rate"),
        ("Screen title accuracy", "screen_title_accuracy"),
        ("Transaction code accuracy", "transaction_code_accuracy"),
        ("Element precision (macro)", "element_precision_macro_average"),
        ("Element recall (macro)", "element_recall_macro_average"),
        ("Element F1 (macro)", "element_f1_macro_average"),
        (
            "Field-value association accuracy (macro)",
            "field_value_association_accuracy_macro_average",
        ),
        (
            "Uncertainty handling accuracy (macro)",
            "uncertainty_handling_accuracy_macro_average",
        ),
    ]

    for label, key in metrics:
        vision = summaries.get("vision_only", {}).get(key)
        ocr = summaries.get("ocr_vision", {}).get(key)
        lines.append(
            f"| {label} | {format_metric(vision)} | {format_metric(ocr)} |"
        )

    lines.extend([
        "",
        "## Additional counts",
        "",
        "| Count | Vision-only | OCR + vision |",
        "|---|---:|---:|",
    ])

    counts = [
        ("Matched elements", "matched_elements_total"),
        ("Reference elements", "reference_elements_total"),
        ("Predicted elements", "predicted_elements_total"),
        ("Expected field-value pairs", "field_value_pairs_expected_total"),
        ("Matched field-value pairs", "field_value_pairs_matched_total"),
        (
            "Potential unsupported field-value pairs",
            "potential_unsupported_pairs_total",
        ),
    ]

    for label, key in counts:
        lines.append(
            f"| {label} | "
            f"{summaries.get('vision_only', {}).get(key, 'N/A')} | "
            f"{summaries.get('ocr_vision', {}).get(key, 'N/A')} |"
        )

    lines.extend([
        "",
        "## Interpretation notes",
        "",
        "- Element matching uses normalized labels, or fallback text/type "
        "when a label is absent.",
        "- Field-value association uses exact normalized label-value pairs.",
        "- Macro averages give each screenshot with a defined score equal weight.",
        "- A potential unsupported pair is a predicted pair absent from the "
        "reference; it is a candidate hallucination, not automatically a "
        "confirmed hallucination.",
        "- Uncertainty accuracy is calculated only for reference elements "
        "explicitly marked uncertain or unreadable.",
        "- A metric marked N/A means it could not be calculated from the "
        "available annotations or outputs.",
        "- Confirm that the reference annotations are complete and accurate "
        "before treating these scores as final.",
        "",
    ])

    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SAP screenshot extraction pipelines."
    )
    parser.add_argument(
        "--reference",
        default=str(REFERENCE_PATH),
        help="Path to manually annotated reference JSON.",
    )
    parser.add_argument(
        "--report-dir",
        default=str(REPORT_DIR),
        help="Directory for generated evaluation reports.",
    )
    args = parser.parse_args()

    reference_path = Path(args.reference)
    report_dir = Path(args.report_dir)

    if not reference_path.is_file():
        print(f"ERROR: Reference file not found: {reference_path}")
        sys.exit(1)

    try:
        references = get_screens(read_json(reference_path))
    except Exception as exc:
        print(f"ERROR: Could not load reference annotations: {exc}")
        sys.exit(1)

    if not references:
        print("ERROR: Reference file contains no screenshot records.")
        sys.exit(1)

    report_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    summaries = {}

    for name, config in PIPELINES.items():
        print(f"\nEvaluating pipeline: {name}")
        rows, summary = evaluate_pipeline(name, config, references)

        all_rows.extend(rows)
        summaries[name] = summary

        print(
            f"Processing success: "
            f"{summary['processing_successes']}/{summary['reference_screenshots']}"
        )
        print(
            f"Schema-valid JSON: "
            f"{summary['schema_valid_outputs']}/"
            f"{summary['reference_screenshots']}"
        )
        print(
            "Title accuracy:",
            format_metric(summary["screen_title_accuracy"]),
        )
        print(
            "Element F1 (macro):",
            format_metric(summary["element_f1_macro_average"]),
        )

    write_csv(report_dir / "evaluation_results.csv", all_rows)

    (report_dir / "evaluation_summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    write_report(
        report_dir / "evaluation_report.md",
        summaries,
        len(references),
    )

    print("\nEvaluation finished.")
    print(f"Detailed CSV: {report_dir / 'evaluation_results.csv'}")
    print(f"Summary JSON: {report_dir / 'evaluation_summary.json'}")
    print(f"Markdown report: {report_dir / 'evaluation_report.md'}")
    print(
        "\nNOTE: Review warnings and the reference/output structure. "
        "Do not interpret N/A or unsupported-pair counts as zero errors."
    )


if __name__ == "__main__":
    main()
