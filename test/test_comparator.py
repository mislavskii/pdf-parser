import pytest
from PIL import Image
import numpy as np
import io
from src.comparator import compare_images


class TestComparator:
    """Test suite for the image comparator function."""

    def create_test_image(self, width=200, height=200, color=128):
        """Create a simple test image."""
        # Create a grayscale image with uniform color
        img_array = np.full((height, width), color, dtype=np.uint8)
        img = Image.fromarray(img_array, mode='L')
        return img

    def create_text_image(self, text="Test"):
        """Create an image with text."""
        # Create a simple grayscale image
        img_array = np.full((100, 200), 255, dtype=np.uint8)
        img = Image.fromarray(img_array, mode='L')
        return img

    def test_identical_images(self):
        """Test comparison of identical images."""
        img = self.create_test_image()
        similarity = compare_images(img, img)
        # Identical images should have high similarity
        assert similarity >= 0.80

    def test_similar_images(self):
        """Test comparison of similar images."""
        img1 = self.create_test_image(color=120)
        img2 = self.create_test_image(color=125)
        similarity = compare_images(img1, img2)
        # Similar images should have moderate to high similarity
        assert similarity >= 0.80

    def test_different_images(self):
        """Test comparison of different images."""
        img1 = self.create_test_image(color=0)      # Black image
        img2 = self.create_test_image(color=255)   # White image
        similarity = compare_images(img1, img2)
        # Very different images should have low similarity
        assert similarity <= 0.50

    def test_size_difference_handling(self):
        """Test comparison of images with different sizes."""
        img1 = self.create_test_image(100, 100)
        img2 = self.create_test_image(200, 200)
        # Should not raise an exception
        similarity = compare_images(img1, img2)
        # Should return a valid similarity score
        assert 0.0 <= similarity <= 1.0

    def test_edge_case_empty_images(self):
        """Test comparison with minimal images."""
        # Create 1x1 images
        img1 = self.create_test_image(1, 1, 0)
        img2 = self.create_test_image(1, 1, 255)
        similarity = compare_images(img1, img2)
        # Should return a valid similarity score
        assert 0.0 <= similarity <= 1.0