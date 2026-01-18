import pymupdf as pmp  # PyMuPDF
import os

def extract_text_and_images(pdf_path):
    # Open the PDF file
    pdf_document = pmp.open(pdf_path)
    output_dir = pdf_path.replace('.pdf', '')
    # Create the main output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        page_dir = os.path.join(output_dir, f"page_{str(page_num + 1).zfill(3)}")
        os.makedirs(page_dir, exist_ok=True)
        
        # Extract text
        text = page.get_text()
        text_path = os.path.join(page_dir, f"page_{page_num + 1}.txt")
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(text)
        
        # Extract images
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            # Extract the image bytes
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            # Save image as PNG
            image_path = os.path.join(page_dir, f"image_{img_index + 1}.png")
            with open(image_path, "wb") as img_file:
                img_file.write(image_bytes)
        
        print(f"Processed page {page_num + 1}")
        
    print("Extraction complete.")

# Example usage:
pdf_path = "/media/mm/DEXP C100/User/Earn/Translate/Thai/Akulo/Doitech/2025.07.03 Company Affidavit.pdf"  # Path to your PDF
# output_dir = "cat_2025"  # Folder to save the results

extract_text_and_images(pdf_path)