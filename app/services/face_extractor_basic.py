"""
Basic face extraction service using OpenCV (fallback when DeepFace is not available)
"""
import cv2
import os
import numpy as np
from typing import List, Dict, Tuple, Optional
from PIL import Image

class BasicFaceExtractor:
    def __init__(self, confidence_threshold: float = 0.7, face_size: int = 224):
        """
        Initialize basic face extractor using OpenCV
        """
        self.confidence_threshold = confidence_threshold
        self.face_size = face_size
        
        # Load OpenCV's pre-trained face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
    def extract_faces_from_image(self, image_path: str) -> List[Dict]:
        """
        Extract faces from a single image using OpenCV
        """
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                return []
            
            # Convert to grayscale for detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            face_results = []
            
            for i, (x, y, w, h) in enumerate(faces):
                try:
                    # Extract face region
                    face_img = img[y:y+h, x:x+w]
                    
                    if face_img.size > 0:
                        # Resize face
                        face_img_resized = cv2.resize(face_img, (self.face_size, self.face_size))
                        
                        # Create a simple embedding (this is just a placeholder)
                        # In a real implementation, you'd use a proper face recognition model
                        embedding = self._create_simple_embedding(face_img_resized)
                        
                        face_results.append({
                            'face_image': face_img_resized,
                            'embedding': embedding,
                            'confidence': 0.8,  # OpenCV doesn't provide confidence, use default
                            'bbox': (x, y, w, h)
                        })
                
                except Exception as e:
                    print(f"Error processing face {i}: {e}")
                    continue
            
            return face_results
            
        except Exception as e:
            print(f"Error extracting faces from {image_path}: {e}")
            return []
    
    def _create_simple_embedding(self, face_img: np.ndarray) -> List[float]:
        """
        Create a simple embedding from face image (placeholder for DeepFace)
        This is just for demonstration - not suitable for real face recognition
        """
        # Convert to grayscale and compute histogram as a simple feature
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        
        # Normalize and convert to list
        embedding = hist.flatten().astype(float).tolist()
        
        # Reduce dimensionality to make it more manageable
        step = len(embedding) // 128  # Reduce to 128 dimensions
        if step > 1:
            embedding = embedding[::step]
        
        # Ensure we have exactly 128 dimensions
        if len(embedding) > 128:
            embedding = embedding[:128]
        elif len(embedding) < 128:
            embedding.extend([0.0] * (128 - len(embedding)))
            
        return embedding
    
    def save_face_image(self, face_img: np.ndarray, output_path: str) -> bool:
        """Save face image to file"""
        try:
            # Ensure the face image is in the right format
            if face_img.dtype != np.uint8:
                face_img = (face_img * 255).astype(np.uint8)
            
            # Convert BGR to RGB if needed (OpenCV uses BGR, PIL uses RGB)
            if len(face_img.shape) == 3 and face_img.shape[2] == 3:
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            
            # Save using PIL for better quality control
            pil_img = Image.fromarray(face_img)
            pil_img.save(output_path, 'JPEG', quality=95)
            return True
            
        except Exception as e:
            print(f"Error saving face image to {output_path}: {e}")
            return False
    
    def process_video_frames(self, frame_paths: List[str], output_dir: str, video_id: int) -> List[Dict]:
        """
        Process multiple video frames to extract faces
        """
        os.makedirs(output_dir, exist_ok=True)
        all_faces = []
        
        for frame_path in frame_paths:
            try:
                # Extract timestamp from filename (assumes format: frame_X.XXs.jpg)
                filename = os.path.basename(frame_path)
                timestamp = float(filename.split('_')[1].replace('s.jpg', ''))
                
                # Extract faces from this frame
                faces = self.extract_faces_from_image(frame_path)
                
                for i, face_data in enumerate(faces):
                    # Generate unique filename for face
                    face_filename = f"video_{video_id}_frame_{timestamp:.2f}s_face_{i}.jpg"
                    face_path = os.path.join(output_dir, face_filename)
                    
                    # Save face image
                    if self.save_face_image(face_data['face_image'], face_path):
                        all_faces.append({
                            'video_id': video_id,
                            'frame_timestamp': timestamp,
                            'face_image_path': face_path,
                            'bbox': face_data['bbox'],
                            'confidence': face_data['confidence'],
                            'embedding': face_data['embedding']
                        })
                
            except Exception as e:
                print(f"Error processing frame {frame_path}: {e}")
                continue
        
        return all_faces
    
    def get_available_models(self) -> Dict[str, List[str]]:
        """Get available models and backends"""
        return {
            'models': ['OpenCV Haar Cascades (Basic)'],
            'detectors': ['opencv']
        }
    
    def get_face_analysis(self, face_path: str) -> Dict:
        """Basic face analysis (placeholder)"""
        return {
            'detector': 'opencv',
            'model': 'haar_cascade',
            'note': 'Install DeepFace for advanced face recognition and analysis'
        }