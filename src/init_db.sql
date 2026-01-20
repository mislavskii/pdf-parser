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

-- Indexes for better performance
CREATE INDEX IF NOT EXISTS idx_pages_document_id ON pages(document_id);
CREATE INDEX IF NOT EXISTS idx_images_page_id ON extracted_images(page_id);
