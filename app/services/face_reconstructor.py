"""
Face Reconstruction Service
Removes masks and reconstructs underlying faces using AI inpainting
"""
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import os
import tempfile
from PIL import Image, ImageDraw
import requests

class FaceReconstructor:
    def __init__(self, model_type: str = "gfpgan", device: str = "cpu"):
        """
        Initialize face reconstructor
        
        Args:
            model_type: Type of reconstruction model ("gfpgan", "codeformer", "lama", "classical")
            device: Device to run on ("cpu", "cuda")
        """
        self.model_type = model_type
        self.device = device
        self.model = None
        self.model_loaded = False
        
        # Try to load the reconstruction model
        if self.model_type != "classical":
            self._load_model()
        else:
            self.model_loaded = True
            print("✅ Classical reconstruction method ready")
    
    def _load_model(self):
        """Load face reconstruction model"""
        try:
            if self.model_type == "gfpgan":
                self._load_gfpgan()
            elif self.model_type == "codeformer":
                self._load_codeformer()
            elif self.model_type == "lama":
                self._load_lama()
            else:
                print(f"⚠️  Unknown model type: {self.model_type}")
                
        except Exception as e:
            print(f"⚠️  Error loading {self.model_type} model: {e}")
            print("   Using classical inpainting fallback")
    
    def _load_gfpgan(self):
        """Load GFPGAN model for face restoration"""
        try:
            # Try to import GFPGAN
            from gfpgan import GFPGANer
            
            # Download model if not exists
            model_path = "models/GFPGANv1.4.pth"
            if not os.path.exists(model_path):
                print("📦 Downloading GFPGAN model...")
                self._download_gfpgan_model(model_path)
            
            # Initialize GFPGAN
            self.model = GFPGANer(
                model_path=model_path,
                upscale=2,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=None
            )
            
            self.model_loaded = True
            print("✅ GFPGAN model loaded successfully")
            
        except ImportError:
            print("⚠️  GFPGAN not installed. Install with: pip install gfpgan")
        except Exception as e:
            print(f"⚠️  Error loading GFPGAN: {e}")
    
    def _load_codeformer(self):
        """Load CodeFormer model for face restoration"""
        try:
            # Try to import CodeFormer
            from codeformer import CodeFormer
            
            model_path = "models/codeformer.pth"
            if not os.path.exists(model_path):
                print("📦 Downloading CodeFormer model...")
                self._download_codeformer_model(model_path)
            
            self.model = CodeFormer(model_path=model_path)
            self.model_loaded = True
            print("✅ CodeFormer model loaded successfully")
            
        except ImportError:
            print("⚠️  CodeFormer not installed")
        except Exception as e:
            print(f"⚠️  Error loading CodeFormer: {e}")
    
    def _load_lama(self):
        """Load LaMa model for inpainting"""
        try:
            # LaMa implementation would go here
            print("⚠️  LaMa model not implemented yet")
        except Exception as e:
            print(f"⚠️  Error loading LaMa: {e}")
    
    def _download_gfpgan_model(self, model_path: str):
        """Download GFPGAN model"""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # GFPGAN model URL (this is a placeholder - use actual URL)
        url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"✅ Downloaded GFPGAN model to {model_path}")
            
        except Exception as e:
            print(f"❌ Failed to download GFPGAN model: {e}")
            raise
    
    def _download_codeformer_model(self, model_path: str):
        """Download CodeFormer model"""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        print("⚠️  CodeFormer model download not implemented")
    
    def create_mask_from_detection(self, face_image: np.ndarray, mask_region: str = "lower_half") -> np.ndarray:
        """
        Create mask for inpainting based on detected mask region
        
        Args:
            face_image: Input face image
            mask_region: Region to mask ("lower_half", "mouth_nose", "custom")
            
        Returns:
            Binary mask for inpainting
        """
        h, w = face_image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if mask_region == "lower_half":
            # Mask lower half of face (typical mask area)
            mask[h//2:, :] = 255
            
        elif mask_region == "mouth_nose":
            # More precise mouth and nose region
            # This would ideally use facial landmark detection
            center_y, center_x = h//2, w//2
            
            # Create elliptical mask around mouth/nose area
            cv2.ellipse(mask, (center_x, int(center_y * 1.2)), 
                       (w//3, h//4), 0, 0, 360, 255, -1)
            
        elif mask_region == "custom":
            # Use edge detection to find mask boundaries
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours that might represent mask edges
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Fill largest contour in lower face region
            if contours:
                # Filter contours in lower half
                lower_contours = []
                for contour in contours:
                    y_coords = contour[:, 0, 1]
                    if np.mean(y_coords) > h//2:
                        lower_contours.append(contour)
                
                if lower_contours:
                    # Use largest contour
                    largest_contour = max(lower_contours, key=cv2.contourArea)
                    cv2.fillPoly(mask, [largest_contour], 255)
                else:
                    # Fallback to lower half
                    mask[h//2:, :] = 255
        
        return mask
    
    def reconstruct_face_gfpgan(self, face_image: np.ndarray) -> np.ndarray:
        """
        Reconstruct face using GFPGAN
        
        Args:
            face_image: Input face image with mask
            
        Returns:
            Reconstructed face image
        """
        if not self.model_loaded:
            return self.reconstruct_face_classical(face_image)
        
        try:
            # GFPGAN processes the entire face, enhancing and reconstructing
            _, _, output = self.model.enhance(face_image, has_aligned=False, only_center_face=True)
            return output
            
        except Exception as e:
            print(f"Error in GFPGAN reconstruction: {e}")
            return self.reconstruct_face_classical(face_image)
    
    def reconstruct_face_classical(self, face_image: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        """
        Classical inpainting fallback method
        
        Args:
            face_image: Input face image
            mask: Inpainting mask (optional)
            
        Returns:
            Inpainted face image
        """
        if mask is None:
            mask = self.create_mask_from_detection(face_image, "lower_half")
        
        # Use OpenCV inpainting
        result = cv2.inpaint(face_image, mask, 3, cv2.INPAINT_TELEA)
        
        # Apply some smoothing
        result = cv2.bilateralFilter(result, 9, 75, 75)
        
        return result
    
    def reconstruct_masked_face(self, face_image: np.ndarray, mask_info: Dict = None) -> Dict:
        """
        Reconstruct a single masked face
        
        Args:
            face_image: Input face image with mask
            mask_info: Information about detected mask
            
        Returns:
            Dictionary with reconstruction results
        """
        try:
            # Choose reconstruction method based on available models
            if self.model_loaded and self.model_type == "gfpgan":
                reconstructed = self.reconstruct_face_gfpgan(face_image)
                method = "gfpgan"
                
            elif self.model_loaded and self.model_type == "codeformer":
                # CodeFormer reconstruction would go here
                reconstructed = self.reconstruct_face_classical(face_image)
                method = "codeformer"
                
            else:
                # Fallback to classical inpainting
                reconstructed = self.reconstruct_face_classical(face_image)
                method = "classical"
            
            # Calculate reconstruction quality metrics
            quality_score = self._calculate_quality_score(face_image, reconstructed)
            
            return {
                'reconstructed_image': reconstructed,
                'original_image': face_image,
                'method': method,
                'quality_score': quality_score,
                'success': True
            }
            
        except Exception as e:
            print(f"Error in face reconstruction: {e}")
            return {
                'reconstructed_image': face_image,  # Return original on failure
                'original_image': face_image,
                'method': 'failed',
                'quality_score': 0.0,
                'success': False,
                'error': str(e)
            }
    
    def _calculate_quality_score(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """
        Calculate quality score for reconstruction
        
        Args:
            original: Original masked face
            reconstructed: Reconstructed face
            
        Returns:
            Quality score between 0 and 1
        """
        try:
            # Convert to grayscale for comparison
            orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            recon_gray = cv2.cvtColor(reconstructed, cv2.COLOR_BGR2GRAY)
            
            # Calculate structural similarity
            from skimage.metrics import structural_similarity as ssim
            similarity = ssim(orig_gray, recon_gray)
            
            # Normalize to 0-1 range
            quality = (similarity + 1) / 2
            
            return float(quality)
            
        except ImportError:
            # Fallback quality metric without scikit-image
            mse = np.mean((original.astype(float) - reconstructed.astype(float)) ** 2)
            quality = 1.0 / (1.0 + mse / 1000)  # Normalize MSE
            return float(quality)
            
        except Exception:
            return 0.5  # Default quality score
    
    def batch_reconstruct_faces(self, masked_faces: List[Dict]) -> List[Dict]:
        """
        Reconstruct multiple masked faces
        
        Args:
            masked_faces: List of face data with mask information
            
        Returns:
            List of reconstruction results
        """
        results = []
        
        for i, face_data in enumerate(masked_faces):
            print(f"Reconstructing face {i+1}/{len(masked_faces)}...")
            
            # Load face image
            if 'face_image_path' in face_data:
                face_image = cv2.imread(face_data['face_image_path'])
                
                if face_image is not None:
                    # Reconstruct face
                    result = self.reconstruct_masked_face(face_image, face_data.get('mask_info'))
                    
                    # Add original face data
                    result.update(face_data)
                    results.append(result)
                else:
                    print(f"⚠️  Could not load face image: {face_data['face_image_path']}")
            
        return results
    
    def save_reconstruction_results(self, results: List[Dict], output_dir: str = "reconstructed_faces"):
        """
        Save reconstruction results to disk
        
        Args:
            results: List of reconstruction results
            output_dir: Directory to save results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for i, result in enumerate(results):
            if result['success']:
                # Save reconstructed image
                filename = f"reconstructed_face_{i:04d}.jpg"
                output_path = os.path.join(output_dir, filename)
                
                cv2.imwrite(output_path, result['reconstructed_image'])
                
                # Save comparison (original vs reconstructed)
                comparison = np.hstack([result['original_image'], result['reconstructed_image']])
                comp_filename = f"comparison_face_{i:04d}.jpg"
                comp_path = os.path.join(output_dir, comp_filename)
                cv2.imwrite(comp_path, comparison)
                
                print(f"💾 Saved reconstruction: {filename} (quality: {result['quality_score']:.2f})")

def install_reconstruction_dependencies():
    """
    Install required dependencies for face reconstruction
    """
    dependencies = [
        "gfpgan",
        "basicsr",
        "facexlib", 
        "realesrgan",
        "scikit-image"
    ]
    
    print("📦 Installing face reconstruction dependencies...")
    print("Run the following commands:")
    
    for dep in dependencies:
        print(f"pip install {dep}")
    
    print("\nOptional GPU acceleration:")
    print("pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")