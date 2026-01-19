import pytest
from unittest.mock import Mock, patch, mock_open
import pymupdf as pmp
from src.handlers import PdfParser
from src.utils import ExtractedImage
import os


class TestPdfParser:
    """Test suite for the PdfParser class."""

    @pytest.fixture
    def mock_doc(self):
        """Create a mock document object."""
        doc = Mock(spec=pmp.Document)
        doc.__len__ = Mock(return_value=5)  # 5 pages
        return doc

    @pytest.fixture
    def mock_page(self):
        """Create a mock page object."""
        page = Mock(spec=pmp.Page)
        page.get_text.return_value = "Sample text content"
        page.get_images.return_value = [
            (1, 0, 0, 0, 0, 0, 0, 0, 0),
            (2, 0, 0, 0, 0, 0, 0, 0, 0)
        ]
        return page

    @pytest.fixture
    def mock_base_image(self):
        """Create a mock base image dictionary."""
        return {
            "image": b"fake_image_data",
            "ext": "png"
        }

    @patch('pymupdf.open')
    def test_init(self, mock_pmp_open, mock_doc):
        """Test PdfParser initialization."""
        mock_pmp_open.return_value = mock_doc
        parser = PdfParser("test.pdf")
        
        assert parser.pdf_path == "test.pdf"
        assert parser.doc == mock_doc
        mock_pmp_open.assert_called_once_with("test.pdf")

    @patch('pymupdf.open')
    def test_get_text(self, mock_pmp_open, mock_doc, mock_page):
        """Test get_text method."""
        mock_pmp_open.return_value = mock_doc
        mock_doc.load_page.return_value = mock_page
        
        parser = PdfParser("test.pdf")
        text = parser.get_text(0)
        
        assert text == "Sample text content"
        mock_doc.load_page.assert_called_once_with(0)
        mock_page.get_text.assert_called_once()

    @patch('pymupdf.open')
    def test_get_images(self, mock_pmp_open, mock_doc, mock_page, mock_base_image):
        """Test get_images method."""
        mock_pmp_open.return_value = mock_doc
        mock_doc.load_page.return_value = mock_page
        mock_doc.extract_image.return_value = mock_base_image
        
        parser = PdfParser("test.pdf")
        images = parser.get_images(0)
        
        assert len(images) == 2
        assert all(isinstance(img, ExtractedImage) for img in images)
        assert images[0].image == b"fake_image_data"
        assert images[0].page == 0
        assert images[0].xref == 1
        assert images[0].extension == "png"
        assert images[1].xref == 2
        mock_doc.load_page.assert_called_once_with(0)
        assert mock_doc.extract_image.call_count == 2
        mock_doc.extract_image.assert_any_call(1)
        mock_doc.extract_image.assert_any_call(2)

    @patch('pymupdf.open')
    @patch('os.makedirs')
    @patch('builtins.open', new_callable=mock_open)
    def test_save_to_files(self, mock_file_open, mock_makedirs, mock_pmp_open, 
                          mock_doc, mock_page, mock_base_image):
        """Test save_to_files method."""
        mock_pmp_open.return_value = mock_doc
        mock_doc.load_page.return_value = mock_page
        mock_doc.__len__ = Mock(return_value=2)  # 2 pages for this test
        mock_doc.extract_image.return_value = mock_base_image
        
        # Mock page.get_images to return different images per page
        mock_page.get_images.side_effect = [
            [(1, 0, 0, 0, 0, 0, 0, 0, 0)],  # Page 0: 1 image
            [(2, 0, 0, 0, 0, 0, 0, 0, 0)]   # Page 1: 1 image
        ]
        
        parser = PdfParser("test.pdf")
        parser.save_to_files()
        
        # Check that directories were created
        assert mock_makedirs.call_count == 3  # Main dir + 2 page dirs
        mock_makedirs.assert_any_call("test", exist_ok=True)
        mock_makedirs.assert_any_call("test/page_001", exist_ok=True)
        mock_makedirs.assert_any_call("test/page_002", exist_ok=True)
        
        # Check that files were written
        assert mock_file_open.call_count == 4  # 2 text files + 2 image files
        mock_file_open.assert_any_call("test/page_001/page_1.txt", "w", encoding="utf-8")
        mock_file_open.assert_any_call("test/page_002/page_2.txt", "w", encoding="utf-8")
        mock_file_open.assert_any_call("test/page_001/image_1.png", "wb")
        mock_file_open.assert_any_call("test/page_002/image_2.png", "wb")
        
        # Check that text was written
        text_handles = [call.return_value for call in mock_file_open.call_args_list if 'txt' in str(call)]
        for handle in text_handles:
            handle.write.assert_called_with("Sample text content")
        
        # Check that image data was written
        image_handles = [call.return_value for call in mock_file_open.call_args_list if 'image' in str(call)]
        for handle in image_handles:
            handle.write.assert_called_with(b"fake_image_data")