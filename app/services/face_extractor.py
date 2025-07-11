"""
Face extraction service using DeepFace with mask detection and reconstruction
"""
import cv2
import os
import numpy as np
from deepface import DeepFace
from typing import List, Dict, Tuple, Optional
import json
from PIL import Image
from .mask_detector import MaskDetector
from .face_reconstructor import FaceReconstructor

class FaceExtractor:
    def __init__(self, 
                 detector_backend: str = 'opencv',
                 model_name: str = 'VGG-Face',
                 confidence_threshold: float = 0.7,
                 face_size: int = 224,
                 enable_mask_detection: bool = True,
                 enable_face_reconstruction: bool = True):
        """
        Initialize face extractor
        
        Args:
            detector_backend: Face detection backend ('opencv', 'mtcnn', 'retinaface', 'ssd')
            model_name: Face recognition model ('VGG-Face', 'Facenet', 'OpenFace', 'DeepID')
            confidence_threshold: Minimum confidence for face detection
            face_size: Size to resize extracted faces
            enable_mask_detection: Enable mask detection on faces
            enable_face_reconstruction: Enable face reconstruction for masked faces
        """
        self.detector_backend = detector_backend
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.face_size = face_size
        self.enable_mask_detection = enable_mask_detection
        self.enable_face_reconstruction = enable_face_reconstruction
        
        # Initialize mask detection and reconstruction services
        if self.enable_mask_detection:
            self.mask_detector = MaskDetector()
            print("✅ Mask detection enabled")
        
        if self.enable_face_reconstruction:
            # Use classical reconstruction as fallback since GFPGAN may not be available
            self.face_reconstructor = FaceReconstructor(model_type="classical")
            print("✅ Face reconstruction enabled")
        
        # Preload models to avoid repeated loading
        try:
            # Test DeepFace with a dummy image
            dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
            DeepFace.represent(dummy_img, model_name=model_name, detector_backend=detector_backend)
        except Exception as e:
            print(f"Warning: Could not preload models: {e}")
    
    def extract_faces_from_image(self, image_path: str) -> List[Dict]:
        """
        Extract faces from a single image
        
        Returns:
            List of face dictionaries with bbox, confidence, and embedding
        """
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                return []
            
            # Use DeepFace to detect and analyze faces
            faces = DeepFace.extract_faces(
                img_path=image_path,
                detector_backend=self.detector_backend,
                enforce_detection=False  # Don't fail if no faces found
            )
            
            face_results = []
            
            for i, face_img in enumerate(faces):
                try:
                    # Convert face image back to proper format
                    if face_img.max() <= 1.0:  # Normalized to 0-1
                        face_img = (face_img * 255).astype(np.uint8)
                    
                    # Get face embedding
                    embedding = DeepFace.represent(
                        img_path=face_img,
                        model_name=self.model_name,
                        detector_backend=self.detector_backend,
                        enforce_detection=False
                    )
                    
                    if embedding:
                        # For now, use a placeholder bbox since DeepFace.extract_faces doesn't return it
                        # We'll improve this with direct detection later
                        face_results.append({
                            'face_image': face_img,
                            'embedding': embedding[0]['embedding'],
                            'confidence': 0.8,  # Placeholder
                            'bbox': (0, 0, face_img.shape[1], face_img.shape[0])  # Placeholder
                        })
                        
                except Exception as e:
                    print(f"Error processing face {i}: {e}")
                    continue
            
            return face_results
            
        except Exception as e:
            print(f"Error extracting faces from {image_path}: {e}")
            return []
    
    def extract_faces_with_detection(self, image_path: str) -> List[Dict]:
        """
        Extract faces with proper bounding box detection
        """
        try:
            # Use DeepFace.analyze to get both detection and embedding
            results = DeepFace.analyze(
                img_path=image_path,
                actions=['emotion'],  # We just need detection, emotion is lightweight
                detector_backend=self.detector_backend,
                enforce_detection=False
            )
            
            # Handle both single face and multiple faces results
            if not isinstance(results, list):
                results = [results]
            
            face_results = []
            img = cv2.imread(image_path)
            
            for result in results:
                try:
                    # Get bounding box
                    region = result.get('region', {})
                    x = region.get('x', 0)
                    y = region.get('y', 0) 
                    w = region.get('w', 0)
                    h = region.get('h', 0)
                    
                    if w > 0 and h > 0:
                        # Extract face region
                        face_img = img[y:y+h, x:x+w]
                        
                        if face_img.size > 0:
                            # Resize face
                            face_img_resized = cv2.resize(face_img, (self.face_size, self.face_size))
                            
                            # Get embedding for the face
                            embedding = DeepFace.represent(
                                img_path=face_img_resized,
                                model_name=self.model_name,
                                detector_backend='skip',  # Skip detection since we already have the face
                                enforce_detection=False
                            )
                            
                            if embedding:
                                face_results.append({
                                    'face_image': face_img_resized,
                                    'embedding': embedding[0]['embedding'],
                                    'confidence': 0.9,  # DeepFace doesn't provide confidence, use default
                                    'bbox': (x, y, w, h)
                                })
                
                except Exception as e:
                    print(f"Error processing detected face: {e}")
                    continue
            
            return face_results
            
        except Exception as e:
            print(f"Error in face detection for {image_path}: {e}")
            return []
    
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
        
        Returns:
            List of face data dictionaries ready for database storage
        """
        os.makedirs(output_dir, exist_ok=True)
        all_faces = []
        
        for frame_path in frame_paths:
            try:
                # Extract timestamp from filename (assumes format: frame_X.XXs.jpg)
                filename = os.path.basename(frame_path)
                timestamp = float(filename.split('_')[1].replace('s.jpg', ''))
                
                # Extract faces from this frame
                faces = self.extract_faces_with_detection(frame_path)
                
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
            'models': ['VGG-Face', 'Facenet', 'Facenet512', 'OpenFace', 'DeepFace', 'DeepID', 'Dlib', 'ArcFace'],
            'detectors': ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe']
        }
    
    def verify_faces(self, face1_path: str, face2_path: str) -> Dict:
        """Verify if two face images are of the same person"""
        try:
            result = DeepFace.verify(
                img1_path=face1_path,
                img2_path=face2_path,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=False
            )
            return result
        except Exception as e:
            return {'verified': False, 'distance': float('inf'), 'error': str(e)}
    
    def get_face_analysis(self, face_path: str) -> Dict:
        """Get detailed face analysis (age, gender, emotion, etc.)"""
        try:
            result = DeepFace.analyze(
                img_path=face_path,
                actions=['age', 'gender', 'race', 'emotion'],
                detector_backend=self.detector_backend,
                enforce_detection=False
            )
            return result[0] if isinstance(result, list) else result
        except Exception as e:
            return {'error': str(e)}
    
    def process_masked_faces(self, face_data: List[Dict]) -> Dict:
        """
        Process faces to detect masks and reconstruct masked faces
        
        Args:
            face_data: List of face detection results
            
        Returns:
            Dictionary with masked faces and reconstruction results
        """
        if not self.enable_mask_detection:
            return {'masked_faces': [], 'reconstructed_faces': []}
        
        print("🔍 Detecting masked faces...")
        
        # Detect masked faces
        masked_faces = []
        for face in face_data:
            if 'face_image_path' in face:
                face_image = cv2.imread(face['face_image_path'])
                if face_image is not None:
                    mask_result = self.mask_detector.detect_mask(face_image)
                    
                    if mask_result['is_masked']:
                        face.update(mask_result)
                        masked_faces.append(face)
                        print(f"😷 Found masked face: {face['face_image_path']} (confidence: {mask_result['confidence']:.2f})")
        
        print(f"Found {len(masked_faces)} masked faces out of {len(face_data)} total faces")
        
        # Reconstruct masked faces if enabled
        reconstructed_faces = []
        if self.enable_face_reconstruction and masked_faces:
            print("🔧 Reconstructing masked faces...")
            reconstructed_faces = self.face_reconstructor.batch_reconstruct_faces(masked_faces)
            
            # Save reconstruction results
            self._save_reconstruction_results(reconstructed_faces)
        
        return {
            'masked_faces': masked_faces,
            'reconstructed_faces': reconstructed_faces,
            'total_faces': len(face_data),
            'masked_count': len(masked_faces),
            'reconstructed_count': len(reconstructed_faces)
        }
    
    def _save_reconstruction_results(self, reconstructed_faces: List[Dict]):
        """Save reconstruction results to faces directory"""
        if not reconstructed_faces:
            return
        
        # Create reconstructed faces directory
        base_dir = os.path.dirname(reconstructed_faces[0]['face_image_path'])
        recon_dir = os.path.join(base_dir, 'reconstructed')
        os.makedirs(recon_dir, exist_ok=True)
        
        for i, result in enumerate(reconstructed_faces):
            if result['success']:
                # Generate filename for reconstructed face
                original_filename = os.path.basename(result['face_image_path'])
                name, ext = os.path.splitext(original_filename)
                recon_filename = f"{name}_reconstructed{ext}"
                recon_path = os.path.join(recon_dir, recon_filename)
                
                # Save reconstructed image
                cv2.imwrite(recon_path, result['reconstructed_image'])
                
                # Update result with saved path
                result['reconstructed_image_path'] = recon_path
                
                print(f"💾 Saved reconstructed face: {recon_filename}")
    
    def get_masked_face_statistics(self, face_data: List[Dict]) -> Dict:
        """
        Get statistics about masked vs unmasked faces
        
        Args:
            face_data: List of face detection results
            
        Returns:
            Statistics dictionary
        """
        if not self.enable_mask_detection:
            return {'error': 'Mask detection not enabled'}
        
        masked_count = 0
        unmasked_count = 0
        mask_types = {}
        
        for face in face_data:
            if 'face_image_path' in face:
                face_image = cv2.imread(face['face_image_path'])
                if face_image is not None:
                    mask_result = self.mask_detector.detect_mask(face_image)
                    
                    if mask_result['is_masked']:
                        masked_count += 1
                        mask_type = mask_result.get('mask_type', 'unknown')
                        mask_types[mask_type] = mask_types.get(mask_type, 0) + 1
                    else:
                        unmasked_count += 1
        
        total_faces = masked_count + unmasked_count
        
        return {
            'total_faces': total_faces,
            'masked_faces': masked_count,
            'unmasked_faces': unmasked_count,
            'mask_percentage': (masked_count / total_faces * 100) if total_faces > 0 else 0,
            'mask_types': mask_types
        }
    
    def enable_advanced_features(self):
        """Enable mask detection and face reconstruction if not already enabled"""
        if not self.enable_mask_detection:
            self.enable_mask_detection = True
            self.mask_detector = MaskDetector()
            print("✅ Mask detection enabled")
        
        if not self.enable_face_reconstruction:
            self.enable_face_reconstruction = True
            self.face_reconstructor = FaceReconstructor(model_type="classical")
            print("✅ Face reconstruction enabled")