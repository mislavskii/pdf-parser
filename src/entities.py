from dataclasses import dataclass


@dataclass
class ExtractedImage:
    image: bytes
    page: int
    xref: int
    extension: str


@dataclass
class ParsedPage:
    id: int
    document_path: str
    page_number: int
    text_content: str
    ocr_result: str
    as_image: bytes
    extracted_images: list[ExtractedImage]