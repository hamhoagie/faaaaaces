"""
Enhanced Video Processor with Unified Face+Mask Detection

This processor uses the unified detection pipeline to capture both
masked and unmasked faces in a single pass.
"""

import cv2
import os
import numpy as np
from typing import List, Tuple, Dict, Generator, Optional
import tempfile
import logging
from pathlib import Path

from .unified_face_mask_detector import UnifiedFaceMaskDetector

logger = logging.getLogger(__name__)

# Optional moviepy import
try:
    from moviepy.editor import VideoFileClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    logger.warning("MoviePy not available, some video features may be limited")

class EnhancedVideoProcessor:
    """
    Enhanced video processor that uses unified face+mask detection
    to capture faces that traditional detection might miss.
    """
    
    def __init__(self, 
                 frame_interval: int = 30,
                 use_gpu: bool = True,
                 detection_threshold: float = 0.3):
        """
        Initialize enhanced video processor.
        
        Args:
            frame_interval: Extract frame every N seconds
            use_gpu: Whether to use GPU acceleration
            detection_threshold: Minimum confidence threshold for detections
        """
        self.frame_interval = frame_interval
        self.use_gpu = use_gpu
        self.detection_threshold = detection_threshold
        
        # Initialize unified detector
        self.unified_detector = UnifiedFaceMaskDetector(
            use_gpu=use_gpu,
            confidence_threshold=detection_threshold
        )
        
        logger.info(f"Enhanced video processor initialized with GPU: {use_gpu}")
    
    def get_video_info(self, video_path: str) -> Dict:
        """Extract video metadata using multiple methods."""
        try:
            # Primary method: OpenCV
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            duration = frame_count / fps if fps > 0 else 0
            
            cap.release()
            
            # Fallback to moviepy if available and OpenCV fails
            if duration == 0 and MOVIEPY_AVAILABLE:
                try:
                    with VideoFileClip(video_path) as clip:
                        duration = clip.duration
                        fps = clip.fps
                        width, height = clip.size
                except Exception as e:
                    logger.warning(f"MoviePy fallback failed: {e}")
            
            # Get file size
            file_size = os.path.getsize(video_path) if os.path.exists(video_path) else 0
            
            return {
                'duration_seconds': duration,
                'fps': fps,
                'width': width,
                'height': height,
                'frame_count': frame_count,
                'file_size_bytes': file_size
            }
            
        except Exception as e:
            logger.error(f"Error getting video info: {e}")
            return {
                'duration_seconds': 0,
                'fps': 0,
                'width': 0,
                'height': 0,
                'frame_count': 0,
                'file_size_bytes': 0
            }
    
    def extract_frames_with_enhanced_detection(self, 
                                             video_path: str,
                                             output_dir: str,
                                             video_id: int) -> List[Dict]:
        """
        Extract frames and detect faces using enhanced pipeline.
        
        Args:
            video_path: Path to video file
            output_dir: Directory to save face images
            video_id: ID of the video in database
            
        Returns:
            List of face detection results
        """
        os.makedirs(output_dir, exist_ok=True)
        all_detections = []
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Processing video: {frame_count} frames at {fps} FPS")
        
        processed_frames = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                current_time = processed_frames / fps
                
                # Extract frames at specified intervals
                if processed_frames % (fps * self.frame_interval) == 0 or processed_frames == 0:
                    logger.info(f"Processing frame at {current_time:.2f}s")
                    
                    # Use unified detection
                    detections = self.unified_detector.process_frame(frame)
                    
                    # Process each detection
                    for i, detection in enumerate(detections):
                        face_data = self._save_face_detection(
                            detection, frame, output_dir, video_id, 
                            current_time, i
                        )
                        if face_data:
                            all_detections.append(face_data)
                
                processed_frames += 1
                
                # Progress logging
                if processed_frames % (fps * 10) == 0:  # Every 10 seconds
                    progress = (processed_frames / frame_count) * 100
                    logger.info(f"Progress: {progress:.1f}%")
        
        except Exception as e:
            logger.error(f"Error processing video: {e}")
        
        finally:
            cap.release()
        
        logger.info(f"Completed processing. Found {len(all_detections)} face detections")
        return all_detections
    
    def _save_face_detection(self, 
                           detection: Dict,
                           frame: np.ndarray,
                           output_dir: str,
                           video_id: int,
                           timestamp: float,
                           face_index: int) -> Optional[Dict]:
        """
        Save face detection and return metadata.
        
        Args:
            detection: Detection result from unified detector
            frame: Original frame
            output_dir: Output directory
            video_id: Video ID
            timestamp: Frame timestamp
            face_index: Face index in frame
            
        Returns:
            Face detection metadata or None if failed
        """
        try:
            # Extract face region
            x1, y1, x2, y2 = detection['bbox']
            face_region = frame[y1:y2, x1:x2]
            
            if face_region.size == 0:
                logger.warning("Empty face region detected")
                return None
            
            # Create filename
            filename = f"video_{video_id}_frame_{timestamp:.2f}s_face_{face_index}.jpg"
            face_path = os.path.join(output_dir, filename)
            
            # Save face image
            success = cv2.imwrite(face_path, face_region)
            if not success:
                logger.warning(f"Failed to save face image: {face_path}")
                return None
            
            # Calculate face area and quality metrics
            face_area = (x2 - x1) * (y2 - y1)
            face_quality = self._assess_face_quality(face_region)
            
            # Prepare face metadata
            face_data = {
                'video_id': video_id,
                'frame_timestamp': timestamp,
                'face_image_path': filename,
                'bbox': (x1, y1, x2, y2),
                'confidence': detection['confidence'],
                'detection_method': detection['method'],
                'face_area': face_area,
                'face_quality': face_quality,
                
                # Mask detection results
                'mask_status': detection.get('mask_status', 'unknown'),
                'mask_confidence': detection.get('mask_confidence', 0.0),
                'mask_type': detection.get('mask_type'),
                
                # Enhanced metadata
                'is_masked': detection.get('mask_status') == 'masked',
                'detection_source': 'unified_pipeline'
            }
            
            return face_data
            
        except Exception as e:
            logger.error(f"Error saving face detection: {e}")
            return None
    
    def _assess_face_quality(self, face_region: np.ndarray) -> float:
        """
        Assess the quality of a detected face.
        
        Args:
            face_region: Face image region
            
        Returns:
            Quality score between 0 and 1
        """
        try:
            if face_region.size == 0:
                return 0.0
            
            # Convert to grayscale
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            
            # Calculate sharpness using Laplacian variance
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(laplacian_var / 1000.0, 1.0)  # Normalize
            
            # Calculate contrast
            contrast = gray.std() / 255.0
            
            # Calculate brightness (avoid too dark or too bright)
            brightness = gray.mean() / 255.0
            brightness_score = 1.0 - abs(brightness - 0.5) * 2
            
            # Size score (larger faces are generally better)
            h, w = gray.shape
            size_score = min((h * w) / (100 * 100), 1.0)  # Normalize to 100x100
            
            # Combine scores
            quality = (sharpness_score * 0.4 + 
                      contrast * 0.3 + 
                      brightness_score * 0.2 + 
                      size_score * 0.1)
            
            return min(quality, 1.0)
            
        except Exception as e:
            logger.error(f"Error assessing face quality: {e}")
            return 0.5  # Default middle quality
    
    def extract_specific_timestamps(self,
                                  video_path: str,
                                  timestamps: List[float],
                                  output_dir: str,
                                  video_id: int) -> List[Dict]:
        """
        Extract frames at specific timestamps for targeted analysis.
        
        Args:
            video_path: Path to video file
            timestamps: List of timestamps in seconds
            output_dir: Output directory
            video_id: Video ID
            
        Returns:
            List of face detections from specified timestamps
        """
        os.makedirs(output_dir, exist_ok=True)
        all_detections = []
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        try:
            for timestamp in timestamps:
                # Seek to specific timestamp
                frame_number = int(timestamp * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"Could not read frame at {timestamp}s")
                    continue
                
                logger.info(f"Analyzing frame at {timestamp}s")
                
                # Use unified detection
                detections = self.unified_detector.process_frame(frame)
                
                # Process each detection
                for i, detection in enumerate(detections):
                    face_data = self._save_face_detection(
                        detection, frame, output_dir, video_id, 
                        timestamp, i
                    )
                    if face_data:
                        all_detections.append(face_data)
                        
        except Exception as e:
            logger.error(f"Error extracting specific timestamps: {e}")
            
        finally:
            cap.release()
        
        return all_detections
    
    def analyze_video_for_masks(self, video_path: str, sample_interval: int = 5) -> Dict:
        """
        Analyze entire video for mask statistics without saving all faces.
        
        Args:
            video_path: Path to video file
            sample_interval: Analyze every N seconds
            
        Returns:
            Summary statistics of mask detection in video
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return {}
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        stats = {
            'total_frames_analyzed': 0,
            'total_faces_detected': 0,
            'masked_faces': 0,
            'unmasked_faces': 0,
            'mask_detection_confidence': [],
            'timestamps_with_masks': [],
            'mask_types_detected': {}
        }
        
        try:
            processed_frames = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                current_time = processed_frames / fps
                
                # Sample frames at specified intervals
                if processed_frames % (fps * sample_interval) == 0:
                    detections = self.unified_detector.process_frame(frame)
                    
                    stats['total_frames_analyzed'] += 1
                    stats['total_faces_detected'] += len(detections)
                    
                    frame_has_masks = False
                    
                    for detection in detections:
                        if detection.get('mask_status') == 'masked':
                            stats['masked_faces'] += 1
                            stats['mask_detection_confidence'].append(
                                detection.get('mask_confidence', 0.0)
                            )
                            frame_has_masks = True
                            
                            # Track mask types
                            mask_type = detection.get('mask_type', 'unknown')
                            stats['mask_types_detected'][mask_type] = (
                                stats['mask_types_detected'].get(mask_type, 0) + 1
                            )
                        else:
                            stats['unmasked_faces'] += 1
                    
                    if frame_has_masks:
                        stats['timestamps_with_masks'].append(current_time)
                
                processed_frames += 1
                
        except Exception as e:
            logger.error(f"Error analyzing video for masks: {e}")
            
        finally:
            cap.release()
        
        # Calculate summary statistics
        if stats['mask_detection_confidence']:
            stats['average_mask_confidence'] = np.mean(stats['mask_detection_confidence'])
        else:
            stats['average_mask_confidence'] = 0.0
        
        stats['mask_detection_rate'] = (
            stats['masked_faces'] / max(stats['total_faces_detected'], 1)
        )
        
        logger.info(f"Video analysis complete: {stats['masked_faces']} masked faces found")
        return stats