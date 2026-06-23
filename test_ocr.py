from PIL import Image, ImageDraw, ImageFont
import pytesseract
from pdf_reader import extract_text_from_pdf

# 1. Create an image with text
img = Image.new('RGB', (800, 600), color=(255, 255, 255))
draw = ImageDraw.Draw(img)

# Just using the default font, but making it bigger by drawing lines
text = "Glossary:\n\n1. Tesseract: an optical character recognition engine.\n2. Pipeline: a set of data processing elements connected in series."

try:
    # Try to load a generic font
    font = ImageFont.truetype("FreeMono.ttf", 30)
except IOError:
    # Fallback to default
    font = ImageFont.load_default()

draw.text((50, 50), text, fill=(0, 0, 0), font=font)

# 2. Save it as an image-only PDF
pdf_path = "scanned_test.pdf"
img.save(pdf_path, "PDF", resolution=300.0)
print(f"Created image-only PDF: {pdf_path}")

# 3. Test the extraction (which should trigger OCR)
print("Running pdf_reader extraction...")
text, error = extract_text_from_pdf(pdf_path)

if error:
    print(f"Error: {error}")
else:
    print("--- Extracted Text ---")
    print(text)
    print("----------------------")
    if "Tesseract" in text:
        print("✅ Success! OCR correctly extracted the text from the image-based PDF.")
    else:
        print("❌ OCR may have failed to read the text correctly.")
