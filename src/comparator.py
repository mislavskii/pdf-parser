import cv2
import numpy as np
from PIL import Image
import pytesseract
from skimage.metrics import structural_similarity as ssim
import hashlib


class PageComparator:
    def __init__(self, subject: Image.Image, object: Image.Image):
        self.subject = subject.convert('L') if subject.mode != 'L' else subject
        self.object = object.convert('L') if object.mode != 'L' else object
        self.preprocess_images_for_comparison()

    def preprocess_images_for_comparison(self):
        subject, object = self.subject, self.object
        if subject.size != object.size:
            mean_dimensions = tuple(int(np.mean([subj_side, obj_side])) for subj_side, obj_side in zip(subject.size, object.size))
            subject, object = map(lambda x: x.resize(mean_dimensions), [subject, object])

        self.subj_array, self.obj_array = map(lambda x: np.array(x), [subject, object])

    def calculate_ssim_similarity(self) -> float:
        """
        Calculate Structural Similarity Index between two images.
        Returns:
            float: SSIM similarity score (0-1)
        """   
        try:
            similarity_score = ssim(self.subj_array, self.obj_array)
            # Handle different return value types
            if isinstance(similarity_score, (tuple, list)):
                if len(similarity_score) > 0:
                    similarity_score = similarity_score[0]
                else:
                    similarity_score = 0.0
            # Convert to float
            similarity_score = float(similarity_score)
            return max(0.0, similarity_score)  # Ensure non-negative
        except Exception:
            # If SSIM fails, return 0
            return 0.0


# all below will be refactored as class methods!

def calculate_histogram_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Calculate histogram correlation similarity between two images.
    
    Args:
        img1: First preprocessed image
        img2: Second preprocessed image
        
    Returns:
        float: Histogram similarity score (0-1)
    """
    # Calculate histograms
    hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
    
    # Normalize histograms
    hist1 = hist1 / hist1.sum()
    hist2 = hist2 / hist2.sum()
    
    # Calculate correlation
    correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    
    # Convert to similarity score (0-1)
    return max(0.0, min(1.0, (correlation + 1) / 2))


def calculate_orb_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Calculate ORB feature matching similarity between two images.
    
    Args:
        img1: First preprocessed image
        img2: Second preprocessed image
        
    Returns:
        float: ORB similarity score (0-1)
    """
    # Initialize AKAZE detector
    akaze = cv2.AKAZE_create()
    
    # Find keypoints and descriptors
    kp1, des1 = akaze.detectAndCompute(img1, None)
    kp2, des2 = akaze.detectAndCompute(img2, None)
    
    # Handle case where no keypoints are found
    if des1 is None or des2 is None:
        return 0.0
    
    # Create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # Match descriptors
    matches = bf.match(des1, des2)
    
    # Calculate similarity based on matches
    if len(matches) == 0:
        return 0.0
    
    # Calculate similarity as ratio of good matches to total possible matches
    max_possible_matches = min(len(kp1), len(kp2))
    if max_possible_matches == 0:
        return 0.0
    
    similarity = len(matches) / max_possible_matches
    return float(min(1.0, similarity))


def calculate_text_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Calculate text similarity using OCR between two images.
    
    Args:
        img1: First preprocessed image
        img2: Second preprocessed image
        
    Returns:
        float: Text similarity score (0-1)
    """
    try:
        # Perform OCR on both images
        # Using PSM 1 (Automatic page segmentation with OSD) for full-page document analysis
        text1 = pytesseract.image_to_string(img1, config='--psm 1')
        text2 = pytesseract.image_to_string(img2, config='--psm 1')
        
        # Handle empty text cases
        if not text1 and not text2:
            return 1.0  # Both empty, considered identical
        if not text1 or not text2:
            return 0.0  # One empty, one not, considered different
        
        # Normalize text (remove extra whitespace, convert to lowercase)
        text1 = ' '.join(text1.split()).lower()
        text2 = ' '.join(text2.split()).lower()
        
        # Handle empty normalized text
        if not text1 and not text2:
            return 1.0
        if not text1 or not text2:
            return 0.0
        
        # Calculate similarity using simple hashing approach
        hash1 = hashlib.md5(text1.encode()).hexdigest()
        hash2 = hashlib.md5(text2.encode()).hexdigest()
        
        # If hashes match, texts are identical
        if hash1 == hash2:
            return 1.0
        
        # Calculate character-level similarity
        # Using a simple approach: common characters / total unique characters
        set1 = set(text1)
        set2 = set(text2)
        
        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0
            
        common_chars = len(set1.intersection(set2))
        total_chars = len(set1.union(set2))
        
        if total_chars == 0:
            return 1.0
            
        return common_chars / total_chars
        
    except Exception:
        # If OCR fails, return 0
        return 0.0


def calculate_pixel_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Calculate pixel-level similarity between two images.
    
    Args:
        img1: First preprocessed image
        img2: Second preprocessed image
        
    Returns:
        float: Pixel similarity score (0-1)
    """
    # Resize images to the same dimensions
    if img1.shape != img2.shape:
        min_height = min(img1.shape[0], img2.shape[0])
        min_width = min(img1.shape[1], img2.shape[1])
        img1 = cv2.resize(img1, (min_width, min_height))
        img2 = cv2.resize(img2, (min_width, min_height))
    
    # Calculate absolute difference
    diff = cv2.absdiff(img1, img2)
    
    # Calculate similarity as inverse of average difference
    avg_diff = np.mean(diff)
    similarity = 1.0 - (avg_diff / 255.0)  # Normalize to 0-1 range
    
    return float(max(0.0, similarity))


def compare_images(subject: Image.Image, object: Image.Image) -> float:
    """
    Compare two PIL images and return a similarity score.
    This function uses multiple comparison methods for comprehensive analysis
    of grayscale photocopies of pages.
    
    Args:
        subject: First PIL Image (grayscale photocopy)
        object: Second PIL Image (grayscale photocopy)
        
    Returns:
        float: Similarity score between 0.0 (completely different) and 1.0 (identical)
    """
    try:
        # Preprocess images for comparison (less aggressive for SSIM/pixel)
        processed_img1 = preprocess_image_for_comparison(subject)
        processed_img2 = preprocess_image_for_comparison(object)
        
        # Calculate multiple similarity metrics
        ssim_score = calculate_ssim_similarity(processed_img1, processed_img2)
        hist_score = calculate_histogram_similarity(processed_img1, processed_img2)
        orb_score = calculate_orb_similarity(processed_img1, processed_img2)
        text_score = calculate_text_similarity(processed_img1, processed_img2)
        pixel_score = calculate_pixel_similarity(processed_img1, processed_img2)
        
        # Weighted combination of all metrics
        # SSIM and histogram are most important for photocopies
        # ORB is useful for structural elements
        # Text similarity helps with content matching
        # Pixel similarity provides baseline comparison
        weighted_score = (
            ssim_score * 0.35 +      # Structural similarity
            hist_score * 0.25 +      # Histogram correlation
            orb_score * 0.15 +       # Feature matching
            text_score * 0.15 +      # Text content similarity
            pixel_score * 0.10       # Pixel-level similarity
        )
        
        # Ensure result is in valid range
        return max(0.0, min(1.0, weighted_score))
        
    except Exception as e:
        # In case of any error, return 0.0 (no similarity)
        print(f"Error in image comparison: {str(e)}")
        return 0.0