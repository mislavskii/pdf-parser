import sqlite3
from .entities import ParsedPage, ExtractedImage


def get_page(pdf_path: str, page_num: int) -> ParsedPage:
    """
    Retrieve a page from the database based on PDF path and page number.
    
    Args:
        pdf_path (str): Path to the PDF file
        page_num (int): Page number (1-based)
        
    Returns:
        ParsedPage: Page data with text content, OCR result, image, and extracted images
    """
    # Connect to the database
    conn = sqlite3.connect('db/database.db')
    cursor = conn.cursor()
    
    try:
        # Get document ID
        cursor.execute('SELECT id FROM documents WHERE file_path = ?', (pdf_path,))
        document_row = cursor.fetchone()
        if not document_row:
            raise ValueError(f"Document with path '{pdf_path}' not found in database")
        document_id = document_row[0]
        
        # Get page data
        cursor.execute('''
            SELECT id, text_content, ocr_result, as_image
            FROM pages
            WHERE document_id = ? AND page_number = ?
        ''', (document_id, page_num))
        
        page_row = cursor.fetchone()
        if not page_row:
            raise ValueError(f"Page {page_num} not found for document '{pdf_path}'")
            
        page_id, text_content, ocr_result, as_image = page_row
        
        # Get extracted images
        cursor.execute('''
            SELECT xref, extension, image_data
            FROM extracted_images
            WHERE page_id = ?
        ''', (page_id,))
        
        image_rows = cursor.fetchall()
        extracted_images = [
            ExtractedImage(image_data, page_num, xref, extension)
            for xref, extension, image_data in image_rows
        ]
        
        # Create and return ParsedPage object
        return ParsedPage(
            id=page_id,
            document_path=pdf_path,
            page_number=page_num,
            text_content=text_content or "",
            ocr_result=ocr_result or "",
            as_image=as_image,
            extracted_images=extracted_images
        )
        
    finally:
        conn.close()


def get_page_count(pdf_path: str) -> int:
    """
    Retrieve the number of pages in a PDF document.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        int: Number of pages in the document
    """
    # Connect to the database
    conn = sqlite3.connect('db/database.db')
    cursor = conn.cursor()
    
    try:
        # Get document ID
        cursor.execute('SELECT id FROM documents WHERE file_path = ?', (pdf_path,))
        document_row = cursor.fetchone()
        if not document_row:
            raise ValueError(f"Document with path '{pdf_path}' not found in database")
        document_id = document_row[0]
        
        # Get page count
        cursor.execute('SELECT COUNT(*) FROM pages WHERE document_id = ?', (document_id,))
        page_count = cursor.fetchone()[0]
        
        return page_count
        
    finally:
        conn.close()