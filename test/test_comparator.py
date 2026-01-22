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

    def create_test_image_with_features(self, width=200, height=200):
        """Create a test image with detectable features."""
        # Create a grayscale image with some features
        img_array = np.random.randint(0, 255, (height, width), dtype=np.uint8)
        # Add a simple pattern - a square in the middle
        center_y, center_x = height // 2, width // 2
        img_array[center_y-20:center_y+20, center_x-20:center_x+20] = 255
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

    def test_calculate_ssim_similarity_identical_images(self):
        """Test SSIM similarity calculation for identical images."""
        img = self.create_test_image()
        comparator = PageComparator(img, img)
        similarity = comparator.calculate_ssim_similarity()
        # Identical images should have high SSIM similarity
        assert similarity >= 0.95

    def test_calculate_ssim_similarity_similar_images(self):
        """Test SSIM similarity calculation for similar images."""
        img1 = self.create_test_image(color=120)
        img2 = self.create_test_image(color=125)
        comparator = PageComparator(img1, img2)
        similarity = comparator.calculate_ssim_similarity()
        # Similar images should have moderate to high SSIM similarity
        assert similarity >= 0.90

    def test_calculate_ssim_similarity_different_images(self):
        """Test SSIM similarity calculation for different images."""
        img1 = self.create_test_image(color=0)      # Black image
        img2 = self.create_test_image(color=255)   # White image
        comparator = PageComparator(img1, img2)
        similarity = comparator.calculate_ssim_similarity()
        # Very different images should have low SSIM similarity
        assert similarity <= 0.50

    def test_calculate_ssim_similarity_edge_cases(self):
        """Test SSIM similarity calculation with edge cases."""
        # Create 1x1 images
        img1 = self.create_test_image(1, 1, 0)
        img2 = self.create_test_image(1, 1, 255)
        comparator = PageComparator(img1, img2)
        similarity = comparator.calculate_ssim_similarity()
        # Should return a valid similarity score
        assert 0.0 <= similarity <= 1.0

    def test_calculate_histogram_similarity_identical_images(self):
        """Test histogram similarity calculation for identical images."""
        img = self.create_test_image()
        comparator = PageComparator(img, img)
        similarity = comparator.calculate_histogram_similarity()
        # Identical images should have perfect histogram similarity
        assert similarity == 1.0

    def test_calculate_histogram_similarity_similar_images(self):
        """Test histogram similarity calculation for similar images."""
        # Create images with similar but not identical color distributions
        img_array1 = np.random.randint(120, 130, (100, 100), dtype=np.uint8)
        img_array2 = np.random.randint(122, 128, (100, 100), dtype=np.uint8)
        img1 = Image.fromarray(img_array1, mode='L')
        img2 = Image.fromarray(img_array2, mode='L')
        comparator = PageComparator(img1, img2)
        similarity = comparator.calculate_histogram_similarity()
        # Similar images should have reasonable histogram similarity
        assert similarity >= 0.50

    def test_calculate_histogram_similarity_different_images(self):
        """Test histogram similarity calculation for different images."""
        # Create images with very different color distributions
        img_array1 = np.random.randint(0, 10, (100, 100), dtype=np.uint8)
        img_array2 = np.random.randint(245, 255, (100, 100), dtype=np.uint8)
        img1 = Image.fromarray(img_array1, mode='L')
        img2 = Image.fromarray(img_array2, mode='L')
        comparator = PageComparator(img1, img2)
        similarity = comparator.calculate_histogram_similarity()
        # Very different images should have lower histogram similarity
        assert similarity <= 0.70

    def test_calculate_histogram_similarity_edge_cases(self):
        """Test histogram similarity calculation with edge cases."""
        # Create 1x1 images
        img1 = self.create_test_image(1, 1, 0)
        img2 = self.create_test_image(1, 1, 255)
        comparator = PageComparator(img1, img2)
        similarity = comparator.calculate_histogram_similarity()
        # Should return a valid similarity score
        assert 0.0 <= similarity <= 1.0
        
    def test_calculate_orb_similarity_identical_images(self):
        """Test ORB similarity calculation for identical images."""
        img = self.create_test_image_with_features()
        comparator = PageComparator(img, img)
        similarity = comparator.calculate_orb_similarity()
        # Identical images with features should have high ORB similarity
        assert similarity >= 0.8, f"Expected high similarity for identical images with features, got {similarity}"

    def test_calculate_orb_similarity_similar_images(self):
        """Test ORB similarity calculation for similar images."""
        # Create two similar images with features
        img1 = self.create_test_image_with_features()
        img2 = self.create_test_image_with_features()
        comparator = PageComparator(img1, img2)
        similarity = comparator.calculate_orb_similarity()
        # Similar images should have reasonable ORB similarity
        assert 0.0 <= similarity <= 1.0, f"Similarity should be between 0 and 1, got {similarity}"

    def test_calculate_orb_similarity_different_images(self):
        """Test ORB similarity calculation for different images."""
        # Create images with very different content
        img1 = self.create_test_image_with_features()
        # Create a completely different image (random noise)
        img_array = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
        img2 = Image.fromarray(img_array, mode='L')
        comparator = PageComparator(img1, img2)
        similarity = comparator.calculate_orb_similarity()
        # Different images should have lower ORB similarity
        assert 0.0 <= similarity <= 0.9, f"Similarity should be between 0 and 0.9 for different images, got {similarity}"

    def test_calculate_orb_similarity_edge_cases(self):
        """Test ORB similarity calculation with edge cases."""
        # Create 1x1 images
        img1 = self.create_test_image(1, 1, 0)
        img2 = self.create_test_image(1, 1, 255)
        comparator = PageComparator(img1, img2)
        similarity = comparator.calculate_orb_similarity()
        # Should return a valid similarity score
        assert 0.0 <= similarity <= 1.0, f"Similarity should be between 0 and 1, got {similarity}"

    def test_calculate_orb_similarity_no_features(self):
        """Test ORB similarity when no features are detected."""
        # Create uniform images that likely won't have detectable features
        img_array = np.full((50, 50), 128, dtype=np.uint8)
        img = Image.fromarray(img_array, mode='L')
        comparator = PageComparator(img, img)
        similarity = comparator.calculate_orb_similarity()
        # Should return 0.0 when no features are detected
        assert similarity == 0.0, f"Expected 0.0 for images with no features, got {similarity}"
