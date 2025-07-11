"""
Mask Detection Service
Detects whether faces are wearing masks or not using computer vision
"""
import cv2
import numpy as np
from typing import List, Tuple, Dict
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

class MaskDetector:
    def __init__(self, model_path: str = None, confidence_threshold: float = 0.5):
        """
        Initialize mask detector
        
        Args:
            model_path: Path to pre-trained mask detection model
            confidence_threshold: Minimum confidence for mask detection
        """
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.model_path = model_path or "models/mask_detector.h5"
        
        # Try to load pre-trained model
        self._load_model()
        
        # If no model available, use simple CV approach
        self.use_cv_fallback = self.model is None
        
    def _load_model(self):
        """Load pre-trained mask detection model"""
        try:
            if os.path.exists(self.model_path):
                self.model = load_model(self.model_path)
                print(f"✅ Loaded mask detection model from {self.model_path}")
            else:
                print(f"⚠️  Mask detection model not found at {self.model_path}")
                print("   Using computer vision fallback method")
        except Exception as e:
            print(f"⚠️  Error loading mask detection model: {e}")
            print("   Using computer vision fallback method")
    
    def detect_mask_cv(self, face_image: np.ndarray) -> Tuple[bool, float]:
        """
        Fallback mask detection using computer vision techniques
        
        Args:
            face_image: Face image as numpy array
            
        Returns:
            Tuple of (is_masked, confidence)
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            
            # Detect facial features
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
            
            # Detect eyes and mouth
            eyes = eye_cascade.detectMultiScale(gray, 1.1, 3)
            mouths = mouth_cascade.detectMultiScale(gray, 1.1, 3)
            
            # Heuristic: if eyes detected but no mouth, likely wearing mask
            has_eyes = len(eyes) > 0
            has_mouth = len(mouths) > 0
            
            # Additional checks for mask-like features
            # Look for horizontal lines in lower face region (mask edges)
            h, w = gray.shape
            lower_face = gray[h//2:, :]
            
            # Edge detection with lower threshold for mask detection
            edges = cv2.Canny(lower_face, 30, 100)
            horizontal_lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=15)
            
            has_horizontal_lines = horizontal_lines is not None and len(horizontal_lines) > 2
            
            # Mask detection logic - more sensitive
            if has_eyes and not has_mouth and has_horizontal_lines:
                return True, 0.9  # Very likely masked
            elif has_eyes and not has_mouth:
                return True, 0.8  # Likely masked
            elif has_horizontal_lines:  # Any horizontal lines in lower face
                return True, 0.7  # Possibly masked  
            elif has_eyes and has_mouth:
                return False, 0.8  # Likely unmasked
            else:
                return False, 0.3  # Uncertain
                
        except Exception as e:
            print(f"Error in CV mask detection: {e}")
            return False, 0.0
    
    def detect_mask_ml(self, face_image: np.ndarray) -> Tuple[bool, float]:
        """
        ML-based mask detection using pre-trained model
        
        Args:
            face_image: Face image as numpy array
            
        Returns:
            Tuple of (is_masked, confidence)
        """
        try:
            # Preprocess image for model
            image = cv2.resize(face_image, (224, 224))
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)
            image = preprocess_input(image)
            
            # Predict
            prediction = self.model.predict(image, verbose=0)[0]
            
            # Assuming model outputs [no_mask_prob, mask_prob]
            mask_prob = prediction[1] if len(prediction) > 1 else prediction[0]
            is_masked = mask_prob > self.confidence_threshold
            
            return is_masked, float(mask_prob)
            
        except Exception as e:
            print(f"Error in ML mask detection: {e}")
            return self.detect_mask_cv(face_image)
    
    def detect_mask(self, face_image: np.ndarray) -> Dict:
        """
        Detect if face is wearing a mask
        
        Args:
            face_image: Face image as numpy array
            
        Returns:
            Dictionary with mask detection results
        """
        if self.use_cv_fallback:
            is_masked, confidence = self.detect_mask_cv(face_image)
            method = "computer_vision"
        else:
            is_masked, confidence = self.detect_mask_ml(face_image)
            method = "machine_learning"
        
        return {
            'is_masked': is_masked,
            'confidence': confidence,
            'method': method,
            'mask_type': self._classify_mask_type(face_image) if is_masked else None
        }
    
    def _classify_mask_type(self, face_image: np.ndarray) -> str:
        """
        Classify type of mask (surgical, N95, cloth, etc.)
        
        Args:
            face_image: Face image as numpy array
            
        Returns:
            String describing mask type
        """
        # Simplified mask type classification
        # In practice, this would use another ML model
        
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        
        # Analyze texture and patterns
        # Surgical masks tend to be more uniform
        # Cloth masks have more texture variation
        # N95 masks have distinctive patterns
        
        # Calculate texture measures
        texture_variance = np.var(gray)
        
        if texture_variance < 100:
            return "surgical"
        elif texture_variance > 200:
            return "cloth"
        else:
            return "unknown"
    
    def process_face_batch(self, face_images: List[np.ndarray]) -> List[Dict]:
        """
        Process multiple face images for mask detection
        
        Args:
            face_images: List of face images as numpy arrays
            
        Returns:
            List of mask detection results
        """
        results = []
        for face_image in face_images:
            result = self.detect_mask(face_image)
            results.append(result)
        
        return results
    
    def get_masked_faces(self, face_data: List[Dict]) -> List[Dict]:
        """
        Filter face data to return only masked faces
        
        Args:
            face_data: List of face detection results
            
        Returns:
            List of faces detected as wearing masks
        """
        masked_faces = []
        
        for face in face_data:
            if 'face_image_path' in face:
                # Load face image
                face_image = cv2.imread(face['face_image_path'])
                if face_image is not None:
                    mask_result = self.detect_mask(face_image)
                    
                    if mask_result['is_masked']:
                        face.update(mask_result)
                        masked_faces.append(face)
        
        return masked_faces

def download_mask_detection_model():
    """
    Download pre-trained mask detection model
    This would download from a model repository
    """
    # For now, we'll use the CV fallback
    # In production, download from HuggingFace, TensorFlow Hub, etc.
    print("📦 Mask detection model download not implemented yet")
    print("   Using computer vision fallback for now")
    return False