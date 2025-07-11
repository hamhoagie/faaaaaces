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
        Improved fallback mask detection using computer vision techniques
        
        Args:
            face_image: Face image as numpy array
            
        Returns:
            Tuple of (is_masked, confidence)
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            
            # Detect facial features with more conservative parameters
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
            
            # Detect eyes and mouth with stricter parameters
            eyes = eye_cascade.detectMultiScale(gray, 1.1, 5)  # More strict detection
            mouths = mouth_cascade.detectMultiScale(gray, 1.1, 5)
            
            # Check for strong eye detection (at least 2 eyes)
            has_strong_eyes = len(eyes) >= 2
            has_mouth = len(mouths) > 0
            
            # Calculate confidence score based on multiple factors
            confidence_score = 0.0
            
            # Factor 1: Eye-mouth ratio (primary indicator)
            if has_strong_eyes and not has_mouth:
                confidence_score += 0.6  # Strong indicator
            elif has_strong_eyes and has_mouth:
                confidence_score -= 0.4  # Strong counter-indicator
            
            # Factor 2: Improved edge detection for mask boundaries
            h, w = gray.shape
            lower_face = gray[h//2:, :]
            
            # More conservative edge detection
            edges = cv2.Canny(lower_face, 80, 150)  # Higher thresholds
            horizontal_lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=30)  # Higher threshold
            
            # Check for strong horizontal line patterns (mask edges)
            strong_horizontal_lines = False
            if horizontal_lines is not None and len(horizontal_lines) > 5:  # More lines required
                # Filter for truly horizontal lines (within 15 degrees)
                horizontal_count = 0
                for line in horizontal_lines:
                    rho, theta = line[0]
                    angle = np.degrees(theta)
                    if 75 <= angle <= 105:  # Near horizontal
                        horizontal_count += 1
                
                strong_horizontal_lines = horizontal_count >= 3
            
            if strong_horizontal_lines:
                confidence_score += 0.3
            
            # Factor 3: Color uniformity in lower face (masks tend to be uniform)
            lower_face_color = cv2.cvtColor(face_image[h//2:, :], cv2.COLOR_BGR2HSV)
            color_variance = np.var(lower_face_color[:, :, 1])  # Saturation variance
            
            if color_variance < 200:  # Very uniform coloring
                confidence_score += 0.2
            
            # Factor 4: Brightness analysis (masks often darken lower face)
            upper_face = gray[:h//2, :]
            lower_face_gray = gray[h//2:, :]
            
            upper_brightness = np.mean(upper_face)
            lower_brightness = np.mean(lower_face_gray)
            brightness_diff = upper_brightness - lower_brightness
            
            # Only consider significant brightness differences
            if brightness_diff > 25:  # Significant darkening
                confidence_score += 0.2
            
            # Apply stricter thresholds
            if confidence_score > 0.7:
                return True, min(confidence_score, 0.95)  # High confidence
            elif confidence_score > 0.5:
                return True, confidence_score  # Medium confidence
            else:
                return False, max(0.1, 1.0 - confidence_score)  # Not masked
                
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
    
    def detect_mask_lightweight_ml(self, face_image: np.ndarray) -> Tuple[bool, float]:
        """
        Lightweight ML-based mask detection using feature extraction
        
        Args:
            face_image: Face image as numpy array
            
        Returns:
            Tuple of (is_masked, confidence)
        """
        try:
            # Extract advanced features for classification
            features = self._extract_mask_features(face_image)
            
            # Simple decision tree based on extracted features
            confidence = self._classify_features(features)
            is_masked = confidence > 0.6
            
            return is_masked, confidence
            
        except Exception as e:
            print(f"Error in lightweight ML mask detection: {e}")
            return self.detect_mask_cv(face_image)
    
    def _extract_mask_features(self, face_image: np.ndarray) -> Dict:
        """Extract features for mask classification"""
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Divide face into regions
        upper_face = gray[:h//3, :]
        middle_face = gray[h//3:2*h//3, :]
        lower_face = gray[2*h//3:, :]
        
        features = {}
        
        # Brightness features
        features['upper_brightness'] = np.mean(upper_face)
        features['middle_brightness'] = np.mean(middle_face)
        features['lower_brightness'] = np.mean(lower_face)
        features['brightness_ratio'] = features['upper_brightness'] / (features['lower_brightness'] + 1)
        
        # Texture features
        features['upper_texture'] = np.std(upper_face)
        features['middle_texture'] = np.std(middle_face)
        features['lower_texture'] = np.std(lower_face)
        
        # Edge features
        edges = cv2.Canny(gray, 100, 200)
        features['total_edges'] = np.sum(edges > 0)
        
        # Lower face edge concentration
        lower_edges = cv2.Canny(lower_face, 100, 200)
        features['lower_edge_density'] = np.sum(lower_edges > 0) / (lower_face.shape[0] * lower_face.shape[1])
        
        # Color features
        hsv = cv2.cvtColor(face_image, cv2.COLOR_BGR2HSV)
        lower_face_hsv = hsv[2*h//3:, :, :]
        features['lower_saturation'] = np.mean(lower_face_hsv[:, :, 1])
        features['lower_value'] = np.mean(lower_face_hsv[:, :, 2])
        
        # Histogram features
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        features['histogram_peak'] = np.argmax(hist)
        features['histogram_spread'] = np.std(hist)
        
        return features
    
    def _classify_features(self, features: Dict) -> float:
        """Classify extracted features to determine mask probability"""
        score = 0.0
        
        # Rule 1: Brightness ratio (masked faces often have darker lower regions)
        if features['brightness_ratio'] > 1.3:
            score += 0.25
        elif features['brightness_ratio'] > 1.1:
            score += 0.15
        
        # Rule 2: Lower face edge density (masks create distinct edges)
        if features['lower_edge_density'] > 0.05:
            score += 0.20
        elif features['lower_edge_density'] > 0.03:
            score += 0.10
        
        # Rule 3: Texture uniformity (masks tend to have more uniform texture)
        texture_ratio = features['upper_texture'] / (features['lower_texture'] + 1)
        if texture_ratio > 1.5:
            score += 0.20
        elif texture_ratio > 1.2:
            score += 0.10
        
        # Rule 4: Color saturation (masks often have lower saturation)
        if features['lower_saturation'] < 50:
            score += 0.15
        elif features['lower_saturation'] < 80:
            score += 0.08
        
        # Rule 5: Histogram analysis (masks create more uniform color distribution)
        if features['histogram_spread'] < 1000:
            score += 0.10
        
        # Rule 6: Overall darkness of lower face
        if features['lower_brightness'] < 80:
            score += 0.10
        
        return min(score, 0.95)  # Cap at 95% confidence
    
    def detect_mask(self, face_image: np.ndarray, min_confidence: float = 0.6) -> Dict:
        """
        Detect if face is wearing a mask using improved algorithms
        
        Args:
            face_image: Face image as numpy array
            min_confidence: Minimum confidence threshold for positive detection
            
        Returns:
            Dictionary with mask detection results
        """
        results = []
        
        # Try multiple detection methods and ensemble them
        if self.use_cv_fallback:
            # Use both CV methods
            cv_result = self.detect_mask_cv(face_image)
            lightweight_result = self.detect_mask_lightweight_ml(face_image)
            
            # Ensemble the results
            avg_confidence = (cv_result[1] + lightweight_result[1]) / 2
            is_masked = avg_confidence > min_confidence
            method = "ensemble_cv_lightweight"
            
            results.append({
                'method': 'computer_vision',
                'confidence': cv_result[1],
                'is_masked': cv_result[0]
            })
            results.append({
                'method': 'lightweight_ml',
                'confidence': lightweight_result[1],
                'is_masked': lightweight_result[0]
            })
            
        else:
            # Use ML model
            is_masked, avg_confidence = self.detect_mask_ml(face_image)
            method = "machine_learning"
            
            results.append({
                'method': 'machine_learning',
                'confidence': avg_confidence,
                'is_masked': is_masked
            })
        
        # Apply confidence filtering
        filtered_is_masked = is_masked and avg_confidence >= min_confidence
        
        return {
            'is_masked': filtered_is_masked,
            'confidence': avg_confidence,
            'method': method,
            'min_confidence_threshold': min_confidence,
            'mask_type': self._classify_mask_type(face_image) if filtered_is_masked else None,
            'detection_details': results
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
    
    def get_masked_faces(self, face_data: List[Dict], min_confidence: float = 0.7) -> List[Dict]:
        """
        Filter face data to return only masked faces with improved detection
        
        Args:
            face_data: List of face detection results
            min_confidence: Minimum confidence threshold for including faces
            
        Returns:
            List of faces detected as wearing masks with high confidence
        """
        masked_faces = []
        
        for face in face_data:
            if 'face_image_path' in face:
                # Load face image
                face_image = cv2.imread(face['face_image_path'])
                if face_image is not None:
                    mask_result = self.detect_mask(face_image, min_confidence)
                    
                    if mask_result['is_masked']:
                        face.update(mask_result)
                        masked_faces.append(face)
        
        return masked_faces
    
    def reprocess_all_faces(self, face_data: List[Dict], min_confidence: float = 0.7) -> Dict:
        """
        Reprocess all faces with improved detection algorithm
        
        Args:
            face_data: List of face detection results
            min_confidence: Minimum confidence threshold
            
        Returns:
            Dictionary with reprocessing results
        """
        results = {
            'total_faces': len(face_data),
            'masked_faces': [],
            'unmasked_faces': [],
            'processing_errors': [],
            'confidence_distribution': {'high': 0, 'medium': 0, 'low': 0}
        }
        
        for i, face in enumerate(face_data):
            try:
                if 'face_image_path' in face:
                    # Load face image
                    face_image = cv2.imread(face['face_image_path'])
                    if face_image is not None:
                        mask_result = self.detect_mask(face_image, min_confidence)
                        
                        # Update face with new detection results
                        face.update(mask_result)
                        
                        # Categorize by confidence
                        if mask_result['confidence'] > 0.8:
                            results['confidence_distribution']['high'] += 1
                        elif mask_result['confidence'] > 0.6:
                            results['confidence_distribution']['medium'] += 1
                        else:
                            results['confidence_distribution']['low'] += 1
                        
                        # Sort into masked/unmasked
                        if mask_result['is_masked']:
                            results['masked_faces'].append(face)
                        else:
                            results['unmasked_faces'].append(face)
                    else:
                        results['processing_errors'].append(f"Could not load image: {face.get('face_image_path', 'unknown')}")
                else:
                    results['processing_errors'].append(f"Face {i} missing image path")
                    
            except Exception as e:
                results['processing_errors'].append(f"Error processing face {i}: {str(e)}")
        
        return results

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