import pytest
from PIL import Image
import numpy as np
import io
from src.comparator import compare_images, PageComparator


class TestPageComparator:
    """Test suite for the PageComparator class."""

    def create_test_image(self, width=200, height=200, color=128):
        """Create a simple test image."""
        # Create a grayscale image with uniform color
        img_array = np.full((height, width), color, dtype=np.uint8)
        img = Image.fromarray(img_array, mode='L')
        return img

    def test_preprocess_same_size_images(self):
        """Test preprocessing with same size images."""
        img1 = self.create_test_image(100, 100)
        img2 = self.create_test_image(100, 100)
        
        comparator = PageComparator(img1, img2)
        
        # Images should not be resized
        assert comparator.subject.size == (100, 100)
        assert comparator.object.size == (100, 100)
        
        # Check that numpy arrays are created
        assert hasattr(comparator, 'subj_array')
        assert hasattr(comparator, 'obj_array')
        assert isinstance(comparator.subj_array, np.ndarray)
        assert isinstance(comparator.obj_array, np.ndarray)
    
    def test_preprocess_different_size_images(self):
        """Test preprocessing with different size images."""
        img1 = self.create_test_image(100, 150)
        img2 = self.create_test_image(200, 250)
        
        comparator = PageComparator(img1, img2)
        
        # Check that numpy arrays are created
        assert hasattr(comparator, 'subj_array')
        assert hasattr(comparator, 'obj_array')
        assert isinstance(comparator.subj_array, np.ndarray)
        assert isinstance(comparator.obj_array, np.ndarray)
        
        # Check that arrays have the same shape (resized to mean dimensions)
        assert comparator.subj_array.shape == comparator.obj_array.shape
    
    def test_preprocess_image_conversion(self):
        """Test that images are properly converted to numpy arrays."""
        img1 = self.create_test_image(50, 50, 100)
        img2 = self.create_test_image(50, 50, 200)
        
        comparator = PageComparator(img1, img2)
        
        # Check array shapes
        assert comparator.subj_array.shape == (50, 50)
        assert comparator.obj_array.shape == (50, 50)
        
        # Check array values (should match the color values)
        assert np.all(comparator.subj_array == 100)
        assert np.all(comparator.obj_array == 200)


# All below will be refactored to test class methods when they are ready
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