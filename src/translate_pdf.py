import fitz  # PyMuPDF


def insert_text_with_fallback(page, pos, text, fontsize, fontname):
    try:
        page.insert_text(pos, text, fontsize=fontsize, fontname=fontname, color=(0,0,0))
    except (RuntimeError, Exception) as e:
        # Fall back font names to try in order
        fallback_fonts = ["helv", "times", "courier"]
        for fb_font in fallback_fonts:
            try:
                page.insert_text(pos, text, fontsize=fontsize * .7, fontname=fb_font, color=(0,0,0))
                print(f"Used fallback font '{fb_font}' for text: {text}")
                return
            except Exception:
                continue
        # If all fail, raise original error
        raise e


PDF_INPUT = "/media/mm/DEXP C100/User/Earn/Translate/Thai/Akulo/Doitech/perevodpravo.ru_2025-08-05_21-02-47/Doitech_Limited Company Certificate 2.pdf"
PDF_OUTPUT = "translated_output.pdf"

# Open input PDF and load first page
doc = fitz.open(PDF_INPUT)
page = doc.load_page(0)

# Extract all text spans with font info and bbox
spans = []
blocks = page.get_text("dict")["blocks"]
for b in blocks:
    if "lines" in b:
        for line in b["lines"]:
            for span in line["spans"]:
                text = span["text"].strip()
                if text:
                    spans.append({
                        "text": text,
                        "bbox": span["bbox"],  # (x0, y0, x1, y1)
                        "font": span["font"],
                        "size": span["size"]
                    })

# Create new PDF for translations
output_doc = fitz.open()
# Add blank page with original page size
new_page = output_doc.new_page(width=page.rect.width, height=page.rect.height)

for idx, info in enumerate(spans):
    # Prompt user with original text inside input prompt
    translation = input(
        f"\nText fragment [{idx+1}/{len(spans)}]: '{info['text']}'\nEnter translation (empty to finish): "
    ).strip()

    # Stop if empty input
    if not translation:
        break

    # Insert translated text at original bbox's (x0, y0) point with original font size and font
    x0, y0, x1, y1 = info["bbox"]
    insert_text_with_fallback(new_page, (x0, y0), translation, info["size"], info["font"])

if output_doc.page_count > 0:
    output_doc.save(PDF_OUTPUT)
    print(f"\nSaved translated PDF to '{PDF_OUTPUT}'.")
else:
    print("No translations were entered; no PDF saved.")

