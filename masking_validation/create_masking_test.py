
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

output = Path("masking_validation/masking_test_input.png")
output.parent.mkdir(parents=True, exist_ok=True)

image = Image.new("RGB", (1100, 480), "white")
draw = ImageDraw.Draw(image)

try:
    font = ImageFont.truetype("arial.ttf", 25)
    title_font = ImageFont.truetype("arial.ttf", 32)
except OSError:
    font = ImageFont.load_default()
    title_font = font

draw.text((30, 20), "SAP Supplier Contact - PRIVACY TEST", fill="black", font=title_font)

rows = [
    "Vendor Name: Rahul Sharma",
    "Vendor Contact Email: rahul.sharma@example.com",
    "Phone Number: 9876543210",
    "Company Code: 1000",
    "Purchasing Organization: P001",
    "Transaction Code: ME31K",
]

y = 100
for row in rows:
    draw.text((40, y), row, fill="black", font=font)
    y += 55

image.save(output)
print(f"Created test image: {output.resolve()}")
