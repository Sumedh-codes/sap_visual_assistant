# SAP Screenshot Extraction Evaluation

Reference screenshots: **15**

Evaluation is performed locally using Python. Bounding-box IoU is excluded.

## Aggregate comparison

| Metric | Vision-only | OCR + vision |
|---|---:|---:|
| Processing success rate | 1.0000 (100.00%) | 1.0000 (100.00%) |
| JSON parse success rate | 1.0000 (100.00%) | 1.0000 (100.00%) |
| Schema-valid JSON rate | 1.0000 (100.00%) | 1.0000 (100.00%) |
| Screen title accuracy | 0.9333 (93.33%) | 0.9333 (93.33%) |
| Transaction code accuracy | 0.4667 (46.67%) | 0.8667 (86.67%) |
| Element precision (macro) | 0.2677 (26.77%) | 0.2726 (27.26%) |
| Element recall (macro) | 0.5446 (54.46%) | 0.4864 (48.64%) |
| Element F1 (macro) | 0.3420 (34.20%) | 0.3305 (33.05%) |
| Field-value association accuracy (macro) | 0.4344 (43.44%) | 0.4106 (41.06%) |
| Uncertainty handling accuracy (macro) | 0.5000 (50.00%) | 0.0000 (0.00%) |

## Additional counts

| Count | Vision-only | OCR + vision |
|---|---:|---:|
| Matched elements | 62 | 54 |
| Reference elements | 104 | 104 |
| Predicted elements | 400 | 401 |
| Expected field-value pairs | 76 | 76 |
| Matched field-value pairs | 35 | 33 |
| Potential unsupported field-value pairs | 260 | 239 |

## Interpretation notes

- Element matching uses normalized labels, or fallback text/type when a label is absent.
- Field-value association uses exact normalized label-value pairs.
- Macro averages give each screenshot with a defined score equal weight.
- A potential unsupported pair is a predicted pair absent from the reference; it is a candidate hallucination, not automatically a confirmed hallucination.
- Uncertainty accuracy is calculated only for reference elements explicitly marked uncertain or unreadable.
- A metric marked N/A means it could not be calculated from the available annotations or outputs.
- Confirm that the reference annotations are complete and accurate before treating these scores as final.
