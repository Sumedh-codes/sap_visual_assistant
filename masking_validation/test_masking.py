
from pathlib import Path
from src.privacy_masking import mask_sensitive_regions

image = Path(r"C:\Sumedh\coding\sap_visual_assistant\masking_test_input.png")

# output = Path(
#     "masked_outputs/15_sensitive_fields_redaction_test_masked.png"
# )

output = Path(
    "masked_outputs/masking_test.png"
)
result = mask_sensitive_regions(image, output)
print(result)
