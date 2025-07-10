"""
Video processing service for frame extraction and metadata
"""
import cv2
import os
from typing import List, Tuple, Dict, Generator
import tempfile

# Optional moviepy import
try:
    from moviepy.editor import VideoFileClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    print("⚠️  MoviePy not available, some video features may be limited")

class VideoProcessor:
    def __init__(self, frame_interval: int = 30):
        """
        Initialize video processor
        
        Args:
            frame_interval: Extract frame every N seconds
        """
        self.frame_interval = frame_interval
    
    def get_video_info(self, video_path: str) -> Dict:
        """Extract video metadata"""
        try:
            # Use OpenCV for basic info
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError("Cannot open video file")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            cap.release()
            
            # Get file size
            file_size = os.path.getsize(video_path)
            
            return {
                'fps': fps,
                'frame_count': frame_count,
                'width': width,
                'height': height,
                'duration_seconds': duration,
                'file_size_bytes': file_size
            }
            
        except Exception as e:
            raise ValueError(f"Error reading video metadata: {str(e)}")
    
    def extract_frames(self, video_path: str, output_dir: str = None) -> Generator[Tuple[float, str], None, None]:
        """
        Extract frames from video at specified intervals
        
        Yields:
            (timestamp, frame_image_path) tuples
        """
        if output_dir is None:
            output_dir = tempfile.mkdtemp()
        
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError("Cannot open video file")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                raise ValueError("Invalid FPS")
            
            frame_interval_frames = int(fps * self.frame_interval)
            frame_number = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Extract frame at intervals
                if frame_number % frame_interval_frames == 0:
                    timestamp = frame_number / fps
                    frame_filename = f"frame_{timestamp:.2f}s.jpg"
                    frame_path = os.path.join(output_dir, frame_filename)
                    
                    # Save frame
                    cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    
                    yield timestamp, frame_path
                
                frame_number += 1
            
            cap.release()
            
        except Exception as e:
            if cap:
                cap.release()
            raise ValueError(f"Error extracting frames: {str(e)}")
    
    def extract_frame_at_timestamp(self, video_path: str, timestamp: float, output_path: str) -> bool:
        """Extract a single frame at specific timestamp"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return False
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_number = int(timestamp * fps)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            
            if ret:
                cv2.imwrite(output_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                cap.release()
                return True
            else:
                cap.release()
                return False
                
        except Exception:
            if cap:
                cap.release()
            return False
    
    def create_video_thumbnail(self, video_path: str, output_path: str, timestamp: float = 5.0) -> bool:
        """Create a thumbnail for the video"""
        return self.extract_frame_at_timestamp(video_path, timestamp, output_path)
    
    def validate_video_file(self, video_path: str) -> Tuple[bool, str]:
        """Validate if file is a proper video"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return False, "Cannot open video file"
            
            # Try to read first frame
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                return False, "Cannot read video frames"
            
            return True, "Valid video file"
            
        except Exception as e:
            return False, f"Video validation error: {str(e)}"
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported video formats"""
        return [
            '.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', 
            '.webm', '.m4v', '.3gp', '.ogv', '.ts', '.mts'
        ]
    
    def cleanup_frames(self, frame_paths: List[str]):
        """Clean up extracted frame files"""
        for frame_path in frame_paths:
            try:
                if os.path.exists(frame_path):
                    os.remove(frame_path)
            except Exception:
                pass  # Ignore cleanup errors