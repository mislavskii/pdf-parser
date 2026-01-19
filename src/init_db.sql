-- SQL script to initialize SQLite database for PDF processing application

-- Table to store PDF documents
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT NOT NULL UNIQUE,
    file_name TEXT NOT NULL
);

-- Table to store PDF pages
CREATE TABLE IF NOT EXISTS pages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL,
    page_number INTEGER NOT NULL,
    text_content TEXT,
    ocr_result TEXT,
    as_image BLOB,
    FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE,
    UNIQUE(document_id, page_number)
);

-- Table to store extracted images
CREATE TABLE IF NOT EXISTS extracted_images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    page_id INTEGER NOT NULL,
    xref INTEGER NOT NULL,
    extension TEXT NOT NULL,
    image_data BLOB,
    FOREIGN KEY (page_id) REFERENCES pages (id) ON DELETE CASCADE
);

-- Table to store processed PDFs
CREATE TABLE IF NOT EXISTS processed_pdfs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    original_document_id INTEGER NOT NULL,
    processed_file_path TEXT NOT NULL UNIQUE,
    processing_type TEXT NOT NULL, -- e.g., 'image_insertion', 'orientation_correction', 'translation'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (original_document_id) REFERENCES documents (id) ON DELETE CASCADE
);

-- Table to store text translations
CREATE TABLE IF NOT EXISTS text_translations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    page_id INTEGER NOT NULL,
    original_text TEXT NOT NULL,
    translated_text TEXT,
    bbox TEXT, -- Bounding box coordinates
    font_name TEXT,
    font_size REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (page_id) REFERENCES pages (id) ON DELETE CASCADE
);

-- Indexes for better performance
CREATE INDEX IF NOT EXISTS idx_pages_document_id ON pages(document_id);
CREATE INDEX IF NOT EXISTS idx_images_page_id ON extracted_images(page_id);
CREATE INDEX IF NOT EXISTS idx_processed_pdfs_document_id ON processed_pdfs(original_document_id);
CREATE INDEX IF NOT EXISTS idx_translations_page_id ON text_translations(page_id);
