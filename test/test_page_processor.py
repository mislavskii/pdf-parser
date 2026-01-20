import pytest
import sqlite3
from unittest.mock import Mock, patch, mock_open
import tempfile
import os
from src.page_processor import get_page, get_page_count
from src.entities import ParsedPage, ExtractedImage


class TestGetPage:
    """Test suite for the get_page function."""

    @pytest.fixture
    def mock_db_connection(self):
        """Create a mock database connection."""
        conn = Mock(spec=sqlite3.Connection)
        cursor = Mock(spec=sqlite3.Cursor)
        conn.cursor.return_value = cursor
        return conn, cursor

    def test_get_page_success(self, mock_db_connection):
        """Test successful retrieval of a page from the database."""
        conn, cursor = mock_db_connection
        
        # Mock database responses
        cursor.fetchone.side_effect = [
            [1],  # document_id
            [10, "Sample text content", "OCR result", b"fake_image_data"]  # page data
        ]
        cursor.fetchall.return_value = [
            [100, "png", b"extracted_image_data"]
        ]
        
        with patch('sqlite3.connect', return_value=conn):
            result = get_page("test.pdf", 1)
        
        # Verify the result
        assert isinstance(result, ParsedPage)
        assert result.id == 10
        assert result.document_path == "test.pdf"
        assert result.page_number == 1
        assert result.text_content == "Sample text content"
        assert result.ocr_result == "OCR result"
        assert result.as_image == b"fake_image_data"
        assert len(result.extracted_images) == 1
        assert isinstance(result.extracted_images[0], ExtractedImage)
        assert result.extracted_images[0].image == b"extracted_image_data"
        assert result.extracted_images[0].page == 1
        assert result.extracted_images[0].xref == 100
        assert result.extracted_images[0].extension == "png"

    def test_get_page_document_not_found(self, mock_db_connection):
        """Test get_page when document is not found in database."""
        conn, cursor = mock_db_connection
        cursor.fetchone.return_value = None  # No document found
        
        with patch('sqlite3.connect', return_value=conn):
            with pytest.raises(ValueError, match="Document with path 'test.pdf' not found in database"):
                get_page("test.pdf", 1)

    def test_get_page_page_not_found(self, mock_db_connection):
        """Test get_page when page is not found in database."""
        conn, cursor = mock_db_connection
        
        # Mock database responses
        cursor.fetchone.side_effect = [
            [1],  # document_id found
            None  # page not found
        ]
        
        with patch('sqlite3.connect', return_value=conn):
            with pytest.raises(ValueError, match="Page 1 not found for document 'test.pdf'"):
                get_page("test.pdf", 1)

    def test_get_page_no_extracted_images(self, mock_db_connection):
        """Test get_page when there are no extracted images."""
        conn, cursor = mock_db_connection
        
        # Mock database responses
        cursor.fetchone.side_effect = [
            [1],  # document_id
            [10, "Sample text content", "OCR result", b"fake_image_data"]  # page data
        ]
        cursor.fetchall.return_value = []  # No extracted images
        
        with patch('sqlite3.connect', return_value=conn):
            result = get_page("test.pdf", 1)
        
        # Verify the result
        assert isinstance(result, ParsedPage)
        assert result.id == 10
        assert result.document_path == "test.pdf"
        assert result.page_number == 1
        assert result.text_content == "Sample text content"
        assert result.ocr_result == "OCR result"
        assert result.as_image == b"fake_image_data"
        assert len(result.extracted_images) == 0

    def test_get_page_empty_text_fields(self, mock_db_connection):
        """Test get_page when text fields are None in database."""
        conn, cursor = mock_db_connection
        
        # Mock database responses with None values for text fields
        cursor.fetchone.side_effect = [
            [1],  # document_id
            [10, None, None, b"fake_image_data"]  # page data with None text fields
        ]
        cursor.fetchall.return_value = []
        
        with patch('sqlite3.connect', return_value=conn):
            result = get_page("test.pdf", 1)
        
        # Verify the result - None values should be converted to empty strings
        assert isinstance(result, ParsedPage)
        assert result.text_content == ""
        assert result.ocr_result == ""

        
class TestGetPageCount:
    """Test suite for the get_page_count function."""

    @pytest.fixture
    def mock_db_connection(self):
        """Create a mock database connection."""
        conn = Mock(spec=sqlite3.Connection)
        cursor = Mock(spec=sqlite3.Cursor)
        conn.cursor.return_value = cursor
        return conn, cursor

    def test_get_page_count_success(self, mock_db_connection):
        """Test successful retrieval of page count from the database."""
        conn, cursor = mock_db_connection
        
        # Mock database responses
        cursor.fetchone.side_effect = [
            [1],  # document_id
            [5]   # page count
        ]
        
        with patch('sqlite3.connect', return_value=conn):
            result = get_page_count("test.pdf")
        
        # Verify the result
        assert result == 5
        
        # Verify database calls
        assert cursor.execute.call_count == 2
        cursor.execute.assert_any_call('SELECT id FROM documents WHERE file_path = ?', ("test.pdf",))
        cursor.execute.assert_any_call('SELECT COUNT(*) FROM pages WHERE document_id = ?', (1,))

    def test_get_page_count_document_not_found(self, mock_db_connection):
        """Test get_page_count when document is not found in database."""
        conn, cursor = mock_db_connection
        cursor.fetchone.return_value = None  # No document found
        
        with patch('sqlite3.connect', return_value=conn):
            with pytest.raises(ValueError, match="Document with path 'test.pdf' not found in database"):
                get_page_count("test.pdf")

    def test_get_page_count_db_connection_closed(self, mock_db_connection):
        """Test that database connection is properly closed."""
        conn, cursor = mock_db_connection
        
        # Mock database responses
        cursor.fetchone.side_effect = [
            [1],  # document_id
            [3]   # page count
        ]
        
        with patch('sqlite3.connect', return_value=conn):
            result = get_page_count("test.pdf")
        
        # Verify the result
        assert result == 3
        
        # Verify connection was closed
        conn.close.assert_called_once()

    def test_get_page_count_db_connection_closed_on_error(self, mock_db_connection):
        """Test that database connection is closed even when an error occurs."""
        conn, cursor = mock_db_connection
        cursor.fetchone.return_value = None  # No document found
        
        with patch('sqlite3.connect', return_value=conn):
            with pytest.raises(ValueError):
                get_page_count("test.pdf")
        
        # Verify connection was closed even though an exception was raised
        conn.close.assert_called_once()