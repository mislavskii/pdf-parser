import pymupdf as pmp
from PIL import Image
import pathlib
import multiprocessing
from pdf2image import convert_from_path
import tempfile

import pytesseract


class PdfParser:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.doc = pmp.open(pdf_path)

    def get_text(self, page_num):
        page = self.doc.load_page(page_num)
        text = page.get_text()
        return text
    
    def get_images(self, page_num):
        page = self.doc.load_page(page_num)
        image_list = page.get_images()
        images = []
        for image in image_list:
            xref = image[0]
            # Extract the image bytes
            base_image = self.doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            images.append(pix)
        return images


class PdfImageInserter:
    def __init__(self, key_string: str, small_image_path: str, padding: float = 5, processed_suffix: str = "_processed"):
        """
        Initialize the inserter with the key string to find,
        the small image path to insert, and padding below the text.
        processed_suffix is added to output filenames.
        """
        self.key_string = key_string
        self.small_image_path = small_image_path
        self.padding = padding
        self.processed_suffix = processed_suffix

        # Load small image once to get its size and aspect ratio
        self.small_img = Image.open(self.small_image_path)
        self.img_width_px, self.img_height_px = self.small_img.size
        self.aspect_ratio = self.img_width_px / self.img_height_px

    def calculate_image_position(self, page: pmp.Page) -> pmp.Rect:
        text_instances = page.search_for(self.key_string)

        if not text_instances:
            raise ValueError(f"'{self.key_string}' not found on page {page.number}")

        text_rect = text_instances[0]
        center_x = (text_rect.x0 + text_rect.x1) / 2
        text_height = text_rect.y1 - text_rect.y0

        img_height = text_height * 2
        img_width = img_height * self.aspect_ratio

        x0 = center_x - img_width / 2
        x1 = center_x + img_width / 2
        y0 = text_rect.y1 + self.padding
        y1 = y0 + img_height

        return pmp.Rect(x0, y0, x1, y1)

    def make_output_path(self, input_pdf_path: str, output_folder: str = None) -> pathlib.Path:
        """
        Constructs an output file path by:
        - Adding the processed_suffix before the file extension
        - Placing the file in output_folder if specified; otherwise same folder as input file
        """
        input_path = pathlib.Path(input_pdf_path)
        stem = input_path.stem
        suffix = input_path.suffix

        new_name = f"{stem}{self.processed_suffix}{suffix}"

        if output_folder:
            output_path = pathlib.Path(output_folder)
            output_path.mkdir(parents=True, exist_ok=True)  # Ensure folder exists
            return output_path / new_name
        else:
            return input_path.with_name(new_name)

    def insert_image_on_last_page(self, input_pdf_path: str, output_pdf_path: str = None):
        """
        Insert the small image on the last page below the key string.
        If output_pdf_path is not provided, it uses make_output_path() 
        to generate an output file path alongside the input.
        """
        if output_pdf_path is None:
            output_pdf_path = self.make_output_path(input_pdf_path)

        doc = pmp.open(input_pdf_path)
        last_page = doc[-1]

        img_rect = self.calculate_image_position(last_page)
        last_page.insert_image(img_rect, filename=self.small_image_path, keep_proportion=True, overlay=True)

        doc.save(str(output_pdf_path))
        doc.close()
        print(f"Processed and saved: {output_pdf_path}")

    def _process_file_worker(self, pdf_file: pathlib.Path, output_folder: pathlib.Path):
            """
            Worker function to process a single PDF file.
            Called by each worker process in the pool.
            """
            try:
                output_file_path = self.make_output_path(str(pdf_file), str(output_folder))
                self.insert_image_on_last_page(str(pdf_file), str(output_file_path))
                return f"Processed {pdf_file.name}"
            except Exception as e:
                return f"Failed to process {pdf_file.name}: {e}"

    def process_folder(self, input_folder: str, output_folder: str):
        """
        Wrapper to process all PDFs in input_folder in parallel using multiprocessing.
        The number of processes will not exceed the number of available CPU cores.
        """
        input_path = pathlib.Path(input_folder)
        output_path = pathlib.Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)

        pdf_files = list(input_path.glob("*.pdf"))

        # Number of CPU cores to limit parallelism
        num_cores = multiprocessing.cpu_count()

        # Create a pool of worker processes
        with multiprocessing.Pool(processes=num_cores) as pool:
            # Map the worker function with arguments for each file
            # Using starmap or map with helper wrapper function
            results = pool.starmap(
                self._process_file_worker,
                [(pdf_file, output_path) for pdf_file in pdf_files]
            )

        # Print individual results from worker
        # for result in results:
        #     print(result)


# import pytesseract
# from PIL import Image
# import pmp  # PyMuPDF
# import os

class PDFOrientationCorrector:
    def __init__(self, dpi=300):
        self.dpi = dpi
        
    def detect_rotation(self, image):
        # Get OSD data
        osd = pytesseract.image_to_osd(image, output_type=pytesseract.Output.DICT)
        return osd["rotate"], osd["orientation"], osd["orientation_conf"]  # rotate: degrees CW to correct

    def process_pdf(self, input_pdf: str, suffix='_corrected'):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Convert each PDF page to image
            images = convert_from_path(input_pdf, dpi=self.dpi, output_folder=tmpdir)
            corrected_images = []
            output_pdf = input_pdf.replace('.pdf', f'{suffix}.pdf')

            for i, img in enumerate(images):
                rotate, orientation, confidence = self.detect_rotation(img)
                print(f"Page {i+1}: Detected rotation={rotate}, orientation={orientation}, confidence={confidence}")
                if rotate != 0:  # and confidence > 1:
                    # Counter-clockwise rotation, so we use transpose for simplicity:
                    img = img.rotate(-rotate, expand=True)  # negative because PIL rotates CCW
                corrected_images.append(img)
            
            # Save corrected pages as a new PDF
            corrected_images[0].save(
                output_pdf,
                save_all=True,
                append_images=corrected_images[1:],
            )
        print(f"Saved: {output_pdf}")

# Example usage:
# corrector = PDFOrientationCorrector(tesseract_lang='eng')
# corrector.process_pdf('Flicka-20-tr.pdf', 'Flicka-20-tr_corrected.pdf')
