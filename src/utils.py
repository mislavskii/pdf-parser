class ExtractedImage:
    def __init__(self, bytes, page, xref, ext):
        self.image = bytes
        self.page = page
        self.xref = xref
        self.extension = ext