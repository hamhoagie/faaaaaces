"""
Unified Face and Mask Detection Service

This service combines face detection and mask detection into a single pipeline
to handle cases where traditional face detection fails on masked faces.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import torch
import torchvision.transforms as transforms
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class UnifiedFaceMaskDetector:
    """
    Unified detector that can simultaneously detect faces (masked and unmasked)
    and determine mask status in a single pass.
    """
    
    def __init__(self, 
                 use_gpu: bool = True,
                 confidence_threshold: float = 0.3,
                 nms_threshold: float = 0.4):
        """
        Initialize the unified detector.
        
        Args:
            use_gpu: Whether to use GPU acceleration if available
            confidence_threshold: Minimum confidence for detections
            nms_threshold: Non-maximum suppression threshold
        """
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize detection models
        self.face_cascade = None
        self.profile_cascade = None
        self.yolo_model = None
        
        self._load_models()
    
    def _load_models(self):
        """Load all detection models."""
        try:
            # Load OpenCV cascades for fallback
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            self.profile_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_profileface.xml'
            )
            
            # Try to load YOLO model if available
            self._try_load_yolo()
            
            logger.info("Models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            # Continue with basic OpenCV only
    
    def _try_load_yolo(self):
        """Try to load YOLO model for better detection."""
        try:
            # Try to load a pre-trained YOLO model
            # This would ideally be a model trained on faces and masks
            import ultralytics
            self.yolo_model = ultralytics.YOLO('yolov8n.pt')  # nano model for speed
            logger.info("YOLO model loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load YOLO model: {e}")
            self.yolo_model = None
    
    def detect_faces_and_masks(self, image: np.ndarray) -> List[Dict]:
        """
        Detect both faces and masks in a single pass.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of detection dictionaries with face bbox and mask status
        """
        detections = []
        
        if self.yolo_model is not None:
            detections.extend(self._detect_with_yolo(image))
        
        # Always run OpenCV detection as backup/supplement
        cv_detections = self._detect_with_opencv(image)
        detections.extend(cv_detections)
        
        # Remove duplicate detections
        detections = self._remove_duplicates(detections)
        
        # For any faces without mask status, determine mask status
        for detection in detections:
            if detection.get('mask_status') is None:
                detection.update(self._analyze_mask_status(image, detection))
        
        return detections
    
    def _detect_with_yolo(self, image: np.ndarray) -> List[Dict]:
        """Detect using YOLO model."""
        detections = []
        
        try:
            results = self.yolo_model(image, conf=self.confidence_threshold)
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Extract box coordinates and confidence
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Map class to face/person detection
                        if class_id == 0:  # person class in COCO
                            # For person detections, focus on head region
                            height = y2 - y1
                            head_height = height * 0.3  # Approximate head region
                            y2 = y1 + head_height
                            
                            detections.append({
                                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                                'confidence': float(confidence),
                                'method': 'yolo',
                                'mask_status': None  # To be determined
                            })
                            
        except Exception as e:
            logger.error(f"YOLO detection error: {e}")
        
        return detections
    
    def _detect_with_opencv(self, image: np.ndarray) -> List[Dict]:
        """Detect using OpenCV cascades with enhanced parameters for masked faces."""
        detections = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Enhanced parameters for masked face detection
        scale_factors = [1.05, 1.1, 1.2]
        min_neighbors = [3, 4, 5]
        
        for scale_factor in scale_factors:
            for min_neighbor in min_neighbors:
                # Frontal face detection
                faces = self.face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=scale_factor, 
                    minNeighbors=min_neighbor,
                    minSize=(30, 30),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                
                for (x, y, w, h) in faces:
                    detections.append({
                        'bbox': (x, y, x + w, y + h),
                        'confidence': 0.7,  # OpenCV doesn't provide confidence
                        'method': 'opencv_frontal',
                        'mask_status': None
                    })
                
                # Profile face detection for better coverage
                profiles = self.profile_cascade.detectMultiScale(
                    gray,
                    scaleFactor=scale_factor,
                    minNeighbors=min_neighbor,
                    minSize=(30, 30)
                )
                
                for (x, y, w, h) in profiles:
                    detections.append({
                        'bbox': (x, y, x + w, y + h),
                        'confidence': 0.6,
                        'method': 'opencv_profile',
                        'mask_status': None
                    })
        
        return detections
    
    def _analyze_mask_status(self, image: np.ndarray, detection: Dict) -> Dict:
        """
        Analyze whether a detected face is wearing a mask.
        
        Args:
            image: Full image
            detection: Detection dictionary with bbox
            
        Returns:
            Dictionary with mask analysis results
        """
        x1, y1, x2, y2 = detection['bbox']
        face_region = image[y1:y2, x1:x2]
        
        if face_region.size == 0:
            return {
                'mask_status': 'unknown',
                'mask_confidence': 0.0,
                'mask_type': None
            }
        
        # Store original color face for color analysis
        face_bgr = face_region.copy()
        
        # Convert to grayscale for analysis
        gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        
        # Enhanced mask detection logic with both color and grayscale
        is_masked, confidence, mask_type = self._enhanced_mask_detection_with_color(gray_face, face_bgr)
        
        return {
            'mask_status': 'masked' if is_masked else 'unmasked',
            'mask_confidence': confidence,
            'mask_type': mask_type
        }
    
    def _enhanced_mask_detection(self, face_gray: np.ndarray) -> Tuple[bool, float, Optional[str]]:
        """
        Enhanced mask detection using multiple heuristics.
        
        Returns:
            Tuple of (is_masked, confidence, mask_type)
        """
        h, w = face_gray.shape
        
        # Analyze different regions of the face
        upper_face = face_gray[:h//2, :]  # Eyes, forehead
        lower_face = face_gray[h//2:, :]  # Nose, mouth, chin
        center_face = face_gray[h//3:2*h//3, :]  # Central region
        
        # Feature detection with relaxed parameters
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        
        eyes = eye_cascade.detectMultiScale(upper_face, 1.05, 2)  # More sensitive
        mouths = mouth_cascade.detectMultiScale(lower_face, 1.05, 2)  # More sensitive
        
        has_eyes = len(eyes) > 0
        has_mouth = len(mouths) > 0
        
        # Edge analysis for mask detection
        edges = cv2.Canny(lower_face, 20, 80)  # Even more sensitive
        
        # Look for horizontal and vertical lines (mask edges)
        horizontal_lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=5)
        has_horizontal_lines = horizontal_lines is not None and len(horizontal_lines) >= 1
        
        # Texture analysis - masks often have uniform texture
        lower_std = np.std(lower_face)
        center_std = np.std(center_face)
        is_uniform_texture = lower_std < 30 or center_std < 25
        
        # Darkness analysis - black masks are very dark
        lower_mean = np.mean(lower_face)
        is_very_dark = lower_mean < 70  # Increased threshold based on data
        
        # Contrast analysis between upper and lower face
        upper_mean = np.mean(upper_face)
        lower_mean = np.mean(lower_face)
        contrast_diff = abs(upper_mean - lower_mean)
        has_sharp_contrast = contrast_diff > 20  # Lowered threshold based on data
        
        # Scoring system with enhanced sensitivity
        mask_score = 0.0
        
        # Very strong indicators for black masks/face coverings
        if is_very_dark:
            mask_score += 0.4  # Dark lower face region
        
        if has_sharp_contrast:
            mask_score += 0.3  # Sharp contrast between upper/lower face
        
        # Strong indicators
        if has_eyes and not has_mouth:
            mask_score += 0.3
        
        if has_horizontal_lines:
            mask_score += 0.2
        
        if is_uniform_texture:
            mask_score += 0.2
        
        # Additional checks for face coverings
        if lower_std < 20:  # Very uniform texture
            mask_score += 0.2
        
        if lower_mean < 65 and is_uniform_texture:  # Dark + uniform (adjusted)
            mask_score += 0.3
        
        # Check for high percentage of very dark pixels (key indicator)
        total_lower_pixels = lower_face.shape[0] * lower_face.shape[1]
        very_dark_pixel_ratio = np.sum(lower_face < 50) / total_lower_pixels
        
        if very_dark_pixel_ratio > 0.4:  # 40%+ very dark pixels suggests mask
            mask_score += 0.4
        elif very_dark_pixel_ratio > 0.25:  # 25%+ suggests possible mask
            mask_score += 0.2
        
        # Determine mask type based on characteristics
        mask_type = None
        if mask_score > 0.4:  # Lower threshold
            if is_very_dark and is_uniform_texture:
                mask_type = "black_face_covering"
            elif lower_std < 15:
                mask_type = "cloth_mask"
            elif has_horizontal_lines:
                mask_type = "surgical_mask"
            else:
                mask_type = "face_covering"
        
        is_masked = mask_score > 0.4  # Lower threshold for detection
        confidence = min(mask_score, 1.0)
        
        return is_masked, confidence, mask_type
    
    def _enhanced_mask_detection_with_color(self, face_gray: np.ndarray, face_bgr: np.ndarray) -> Tuple[bool, float, Optional[str]]:
        """
        Enhanced mask detection using both grayscale and color information.
        
        Returns:
            Tuple of (is_masked, confidence, mask_type)
        """
        # Get base detection from grayscale
        is_masked_gray, confidence_gray, mask_type_gray = self._enhanced_mask_detection(face_gray)
        
        # Add color analysis
        h, w = face_gray.shape
        lower_face_bgr = face_bgr[h//2:, :]
        
        # Enhanced color analysis for black masks/face coverings
        color_score = self._analyze_mask_colors_enhanced(lower_face_bgr)
        
        # Combine scores
        combined_confidence = confidence_gray + (color_score * 0.3)
        combined_confidence = min(combined_confidence, 1.0)
        
        # Enhanced detection logic
        is_masked = is_masked_gray or (color_score > 0.6)
        
        # Enhanced mask type detection
        mask_type = mask_type_gray
        if is_masked and color_score > 0.8:
            mask_type = "black_tactical_mask"
        
        return is_masked, combined_confidence, mask_type
    
    def _analyze_mask_colors_enhanced(self, face_region_bgr: np.ndarray) -> float:
        """Enhanced color analysis specifically for tactical/black masks."""
        if face_region_bgr.size == 0:
            return 0.0
        
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(face_region_bgr, cv2.COLOR_BGR2HSV)
        
        # Black/dark mask detection (very broad range)
        black_lower = np.array([0, 0, 0])
        black_upper = np.array([180, 255, 60])  # Increased upper value
        
        # Very dark gray (common for tactical masks)
        dark_gray_lower = np.array([0, 0, 0])
        dark_gray_upper = np.array([180, 50, 80])
        
        # Calculate coverage
        black_mask = cv2.inRange(hsv, black_lower, black_upper)
        dark_gray_mask = cv2.inRange(hsv, dark_gray_lower, dark_gray_upper)
        
        total_pixels = face_region_bgr.shape[0] * face_region_bgr.shape[1]
        
        black_ratio = np.sum(black_mask > 0) / total_pixels
        dark_gray_ratio = np.sum(dark_gray_mask > 0) / total_pixels
        
        # Also check RGB values directly for very dark regions
        gray_face = cv2.cvtColor(face_region_bgr, cv2.COLOR_BGR2GRAY)
        very_dark_pixels = np.sum(gray_face < 50) / total_pixels
        
        # Maximum score from different dark color analyses
        max_dark_ratio = max(black_ratio, dark_gray_ratio, very_dark_pixels)
        
        return max_dark_ratio
    
    def _analyze_mask_colors(self, face_region_bgr: np.ndarray) -> float:
        """Analyze color patterns typical of masks."""
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(face_region_bgr, cv2.COLOR_BGR2HSV)
        
        # Define mask color ranges
        # Black masks
        black_lower = np.array([0, 0, 0])
        black_upper = np.array([180, 255, 50])
        
        # Blue masks (surgical)
        blue_lower = np.array([100, 50, 50])
        blue_upper = np.array([130, 255, 255])
        
        # White/light masks
        white_lower = np.array([0, 0, 200])
        white_upper = np.array([180, 30, 255])
        
        # Calculate color mask coverage
        black_mask = cv2.inRange(hsv, black_lower, black_upper)
        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
        white_mask = cv2.inRange(hsv, white_lower, white_upper)
        
        total_pixels = face_region_bgr.shape[0] * face_region_bgr.shape[1]
        
        black_ratio = np.sum(black_mask > 0) / total_pixels
        blue_ratio = np.sum(blue_mask > 0) / total_pixels
        white_ratio = np.sum(white_mask > 0) / total_pixels
        
        max_color_ratio = max(black_ratio, blue_ratio, white_ratio)
        
        return max_color_ratio
    
    def _remove_duplicates(self, detections: List[Dict]) -> List[Dict]:
        """Remove duplicate detections using IoU threshold."""
        if len(detections) <= 1:
            return detections
        
        # Sort by confidence
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        filtered = []
        for detection in detections:
            bbox1 = detection['bbox']
            
            is_duplicate = False
            for existing in filtered:
                bbox2 = existing['bbox']
                iou = self._calculate_iou(bbox1, bbox2)
                
                if iou > self.nms_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered.append(detection)
        
        return filtered
    
    def _calculate_iou(self, bbox1: Tuple[int, int, int, int], 
                       bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union of two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def extract_face_region(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Extract face region from image given bounding box."""
        x1, y1, x2, y2 = bbox
        return image[y1:y2, x1:x2]
    
    def process_frame(self, frame: np.ndarray) -> List[Dict]:
        """
        Process a single frame and return all face detections with mask status.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            List of detections with enhanced information
        """
        detections = self.detect_faces_and_masks(frame)
        
        # Add additional metadata
        for i, detection in enumerate(detections):
            detection['detection_id'] = i
            detection['frame_shape'] = frame.shape
            
            # Extract face image
            face_region = self.extract_face_region(frame, detection['bbox'])
            detection['face_image'] = face_region
        
        return detections