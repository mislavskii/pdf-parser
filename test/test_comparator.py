import pytest
from PIL import Image, ImageDraw, ImageFont
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

    def create_test_image_with_text(self, text_lines=["Test Line 1", "Test Line 2", "Test Line 3"], width=300, height=200):
        """Create a test image with multiple lines of text content."""
        # Create a grayscale image
        img = Image.new('L', (width, height), color=255)  # White background
        draw = ImageDraw.Draw(img)
        
        # Use the default font
        font = ImageFont.load_default()
        
        # Calculate total text height
        line_height = 20  # Approximate line height
        total_text_height = len(text_lines) * line_height
        
        # Starting y position (centered vertically)
        start_y = (height - total_text_height) // 2
        
        # Draw each line of text
        for i, line in enumerate(text_lines):
            # Calculate text position (centered horizontally)
            text_bbox = draw.textbbox((0, 0), line, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            x = (width - text_width) // 2
            y = start_y + i * line_height
            
            # Draw text
            draw.text((x, y), line, fill=0, font=font)  # Black text
        
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


    def test_calculate_pixel_similarity_identical_images(self):
        """Test pixel similarity calculation for identical images."""
        img = self.create_test_image()
        comparator = PageComparator(img, img)
        similarity = comparator.calculate_pixel_similarity()
        # Identical images should have perfect pixel similarity
        assert similarity == 1.0

    def test_calculate_pixel_similarity_similar_images(self):
        """Test pixel similarity calculation for similar images."""
        # Create images with similar but not identical pixel values
        img_array1 = np.full((100, 100), 120, dtype=np.uint8)
        img_array2 = np.full((100, 100), 125, dtype=np.uint8)
        img1 = Image.fromarray(img_array1, mode='L')
        img2 = Image.fromarray(img_array2, mode='L')
        comparator = PageComparator(img1, img2)
        similarity = comparator.calculate_pixel_similarity()
        # Similar images should have high pixel similarity
        assert similarity >= 0.95

    def test_calculate_pixel_similarity_different_images(self):
        """Test pixel similarity calculation for different images."""
        # Create images with very different pixel values
        img_array1 = np.full((100, 100), 0, dtype=np.uint8)    # Black image
        img_array2 = np.full((100, 100), 255, dtype=np.uint8)  # White image
        img1 = Image.fromarray(img_array1, mode='L')
        img2 = Image.fromarray(img_array2, mode='L')
        comparator = PageComparator(img1, img2)
        similarity = comparator.calculate_pixel_similarity()
        # Very different images should have low pixel similarity
        assert similarity <= 0.50

    def test_calculate_pixel_similarity_edge_cases(self):
        """Test pixel similarity calculation with edge cases."""
        # Create 1x1 images
        img1 = self.create_test_image(1, 1, 0)
        img2 = self.create_test_image(1, 1, 255)
        comparator = PageComparator(img1, img2)
        similarity = comparator.calculate_pixel_similarity()
        # Should return a valid similarity score
        assert 0.0 <= similarity <= 1.0

    def test_calculate_pixel_similarity_different_sizes(self):
        """Test pixel similarity calculation with different sized images."""
        img1 = self.create_test_image(100, 150)
        img2 = self.create_test_image(200, 250)
        comparator = PageComparator(img1, img2)
        similarity = comparator.calculate_pixel_similarity()
        # Should handle different sizes by resizing
        assert 0.0 <= similarity <= 1.0

    def test_calculate_text_similarity_identical_text(self):
        """Test text similarity calculation for images with identical text."""
        # Create two images with identical text
        text_lines = ["Hello World", "This is a test", "Page comparison"]
        img1 = self.create_test_image_with_text(text_lines)
        img2 = self.create_test_image_with_text(text_lines)
        
        comparator = PageComparator(img1, img2)
        similarity = comparator.calculate_text_similarity()
        
        # Identical text should result in perfect similarity
        assert similarity == 1.0, f"Expected 1.0 for identical text, got {similarity}"

    def test_calculate_text_similarity_similar_text(self):
        """Test text similarity calculation for images with similar text."""
        # Create two images with similar but not identical text
        text_lines1 = ["Hello World", "This is a test", "Page comparison"]
        text_lines2 = ["Hello World", "This is a test", "Document comparison"]
        img1 = self.create_test_image_with_text(text_lines1)
        img2 = self.create_test_image_with_text(text_lines2)
        
        comparator = PageComparator(img1, img2)
        similarity = comparator.calculate_text_similarity()
        
        # Similar text should result in high similarity (at least 0.5)
        assert similarity >= 0.5, f"Expected >= 0.5 for similar text, got {similarity}"
        assert 0.0 <= similarity <= 1.0, f"Similarity should be between 0 and 1, got {similarity}"

    def test_calculate_text_similarity_different_text(self):
        """Test text similarity calculation for images with different text."""
        # Create two images with completely different text
        text_lines1 = ["Hello World", "This is a test", "Page comparison"]
        text_lines2 = ["Goodbye Universe", "This is not a test", "Document analysis"]
        img1 = self.create_test_image_with_text(text_lines1)
        img2 = self.create_test_image_with_text(text_lines2)
        
        comparator = PageComparator(img1, img2)
        similarity = comparator.calculate_text_similarity()
        
        # Different text should result in lower similarity
        assert similarity <= 0.8, f"Expected <= 0.8 for different text, got {similarity}"
        assert 0.0 <= similarity <= 1.0, f"Similarity should be between 0 and 1, got {similarity}"

    def test_calculate_text_similarity_empty_text(self):
        """Test text similarity calculation for images with no text."""
        # Create images without text (just blank images)
        img1 = self.create_test_image(200, 200, 255)  # White image
        img2 = self.create_test_image(200, 200, 255)  # White image
        
        comparator = PageComparator(img1, img2)
        similarity = comparator.calculate_text_similarity()
        
        # Empty text on both images should result in perfect similarity
        assert similarity == 1.0, f"Expected 1.0 for empty text on both images, got {similarity}"

    def test_calculate_text_similarity_one_empty_text(self):
        """Test text similarity calculation when one image has no text."""
        # Create one image with text and one without text
        text_lines = ["Hello World", "This is a test"]
        img1 = self.create_test_image_with_text(text_lines)
        img2 = self.create_test_image(200, 200, 255)  # White image (no text)
        
        comparator = PageComparator(img1, img2)
        similarity = comparator.calculate_text_similarity()
        
        # One image with text, one without should result in zero similarity
        assert similarity == 0.0, f"Expected 0.0 for one empty text, got {similarity}"

    def test_calculate_text_similarity_edge_cases(self):
        """Test text similarity calculation with edge cases."""
        # Test with 1x1 images
        img1 = self.create_test_image(1, 1, 0)
        img2 = self.create_test_image(1, 1, 255)
        comparator = PageComparator(img1, img2)
        similarity = comparator.calculate_text_similarity()
        # Should return a valid similarity score
        assert 0.0 <= similarity <= 1.0, f"Similarity should be between 0 and 1, got {similarity}"
