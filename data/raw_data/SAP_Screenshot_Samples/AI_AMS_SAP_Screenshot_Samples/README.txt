AI-AMS SAP Screenshot Intelligence — Sample Dataset
=====================================================

This ZIP contains 15 SYNTHETIC SAP-STYLE SCREEN MOCKUPS for a 3-day AI-ML internship assignment.
They are generated examples, NOT genuine SAP captures and NOT validated representations of every SAP release.

Coverage:
- SAP GUI-style screens: ME31K, ME21N, ME51N, ME41, ME32K, ME2N, ME23N
- Contract types: MK, WK, ZMK, ZWK
- Purchasing organization P001, purchasing group P11, company code 1000
- SAP Fiori-style tiles, forms, and tables
- Error/warning messages, dense tables, custom Z transaction, obscured content, and redaction test

All names, values, IDs, dates, and messages are synthetic. Some transaction codes are real SAP transaction-code examples, but the screen content and message details are illustrative. In particular, the error message ID is illustrative.

Suggested assignment:
1. Run vision-only extraction.
2. Run OCR + vision extraction.
3. Compare critical text, SAP identifiers, field/value associations, bounding boxes, JSON validity, latency, and cost.
4. Report uncertainty rather than inventing unreadable fields.
5. Test redaction before any external vision API call.

Privacy note:
The images are synthetic and do not contain actual customer data. For any real SAP screenshots, obtain authorization and anonymize them before use. Redaction must be applied to the pixels, not just to the model prompt.

Coordinates:
Bounding boxes should use [x1, y1, x2, y2] in pixels relative to the original 1440x900 image, unless the intern documents a different convention.

Dataset manifest:
- 01_me31k_contract_header.png: ME31K contract header; contract types MK/WK/ZMK/ZWK; P001, P11, company code 1000.
- 02_me21n_purchase_order.png: ME21N purchase order with header fields and item table.
- 03_me51n_purchase_requisition.png: ME51N purchase requisition with item details.
- 04_me41_request_for_quotation.png: ME41 RFQ screen with purchasing context and vendor list.
- 05_me32k_contract_item_overview.png: ME32K contract item table for extraction and row-column association.
- 06_me2n_alv_purchase_order_list.png: ME2N-style ALV list with multiple rows and status values.
- 07_error_popup_missing_purch_org.png: Error popup with a missing purchasing organization; message ID is illustrative.
- 08_contract_warning_messages.png: Contract check messages with warning, information, and success states.
- 09_fiori_purchasing_launchpad.png: SAP Fiori-style purchasing launchpad with application tiles.
- 10_fiori_contract_form.png: Fiori-style purchase contract form with labels and input values.
- 11_fiori_requisition_items_table.png: Fiori-style requisition table for row/column extraction.
- 12_custom_zmm_contract_monitor.png: Synthetic custom Z transaction screen; tests generic parsing of Z objects.
- 13_dense_purchase_history.png: Dense SAP-style table with small text and many columns.
- 14_obscured_content_uncertainty.png: Screen with intentionally obscured content; parser should flag uncertainty.
- 15_sensitive_fields_redaction_test.png: Synthetic screenshot with sensitive-like fields visibly masked for redaction workflow testing.
