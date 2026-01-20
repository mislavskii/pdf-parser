from dataclasses import dataclass

@dataclass
class ExtractedImage:
    image: bytes
    page: int
    xref: int
    extension: str


@dataclass
class Page:
    pass