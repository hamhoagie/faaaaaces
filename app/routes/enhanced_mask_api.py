"""
Enhanced Mask Detection API using the unified face+mask detector.
"""

from flask import Blueprint, jsonify, request
import logging
from pathlib import Path
import tempfile
import os

from ..services.enhanced_video_processor import EnhancedVideoProcessor
from ..models.database import FaceModel

logger = logging.getLogger(__name__)

enhanced_mask_api_bp = Blueprint('enhanced_mask_api', __name__)

@enhanced_mask_api_bp.route('/enhanced_detect_masks/<int:video_id>', methods=['POST'])
def enhanced_detect_masks(video_id):
    """
    Reprocess a video using the enhanced unified face+mask detection pipeline.
    
    Args:
        video_id: ID of the video to reprocess
    """
    try:
        # For now, we'll reprocess the downloaded video
        # In a production system, you'd store the video file path in the database
        
        if video_id == 4:
            # Use the downloaded MELT Act video
            video_path = "temp/MELT_Act_video.mp4"
        else:
            return jsonify({
                'error': f'Enhanced detection currently only supports video 4',
                'video_id': video_id
            }), 400
        
        if not os.path.exists(video_path):
            return jsonify({
                'error': f'Video file not found: {video_path}',
                'video_id': video_id
            }), 404
        
        # Initialize enhanced processor
        processor = EnhancedVideoProcessor(
            frame_interval=5,  # Every 5 seconds for more thorough analysis
            use_gpu=False,     # Start with CPU, can enable GPU later
            detection_threshold=0.3
        )
        
        # Create output directory
        output_dir = f"faces/{video_id}_enhanced"
        os.makedirs(output_dir, exist_ok=True)
        
        # Process video with enhanced detection
        logger.info(f"Starting enhanced processing of video {video_id}")
        detections = processor.extract_frames_with_enhanced_detection(
            video_path, output_dir, video_id
        )
        
        # Prepare results
        results = []
        masked_count = 0
        unmasked_count = 0
        
        for detection in detections:
            is_masked = detection.get('is_masked', False)
            if is_masked:
                masked_count += 1
            else:
                unmasked_count += 1
            
            # Convert numpy types to Python native types for JSON serialization
            bbox = detection['bbox']
            if hasattr(bbox[0], 'item'):  # numpy type
                bbox = tuple(int(coord.item()) if hasattr(coord, 'item') else int(coord) for coord in bbox)
            
            results.append({
                'timestamp': float(detection['frame_timestamp']),
                'bbox': bbox,
                'confidence': float(detection['confidence']),
                'detection_method': detection['detection_method'],
                'mask_status': detection['mask_status'],
                'mask_confidence': float(detection['mask_confidence']),
                'mask_type': detection.get('mask_type'),
                'face_image_path': detection['face_image_path'],
                'face_quality': float(detection.get('face_quality', 0.0))
            })
        
        # Log results
        logger.info(f"Enhanced detection complete: {len(detections)} faces, "
                   f"{masked_count} masked, {unmasked_count} unmasked")
        
        return jsonify({
            'video_id': video_id,
            'total_faces': len(detections),
            'masked_faces': masked_count,
            'unmasked_faces': unmasked_count,
            'results': results,
            'detection_method': 'enhanced_unified_pipeline'
        })
        
    except Exception as e:
        logger.error(f"Error in enhanced mask detection: {e}")
        return jsonify({
            'error': str(e),
            'video_id': video_id
        }), 500

@enhanced_mask_api_bp.route('/enhanced_analyze_video/<int:video_id>', methods=['POST'])
def enhanced_analyze_video(video_id):
    """
    Analyze a video for mask statistics without saving all faces.
    """
    try:
        if video_id == 4:
            video_path = "temp/MELT_Act_video.mp4"
        else:
            return jsonify({
                'error': f'Enhanced analysis currently only supports video 4',
                'video_id': video_id
            }), 400
        
        if not os.path.exists(video_path):
            return jsonify({
                'error': f'Video file not found: {video_path}',
                'video_id': video_id
            }), 404
        
        # Initialize enhanced processor
        processor = EnhancedVideoProcessor(
            frame_interval=30,  # Standard interval for analysis
            use_gpu=False,
            detection_threshold=0.3
        )
        
        # Analyze video for masks
        stats = processor.analyze_video_for_masks(video_path, sample_interval=2)
        
        stats['video_id'] = video_id
        
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Error in enhanced video analysis: {e}")
        return jsonify({
            'error': str(e),
            'video_id': video_id
        }), 500

@enhanced_mask_api_bp.route('/enhanced_extract_timestamps/<int:video_id>', methods=['POST'])
def enhanced_extract_timestamps(video_id):
    """
    Extract faces at specific timestamps using enhanced detection.
    """
    try:
        data = request.get_json() or {}
        timestamps = data.get('timestamps', [])
        
        if not timestamps:
            return jsonify({'error': 'No timestamps provided'}), 400
        
        if video_id == 4:
            video_path = "temp/MELT_Act_video.mp4"
        else:
            return jsonify({
                'error': f'Enhanced extraction currently only supports video 4',
                'video_id': video_id
            }), 400
        
        if not os.path.exists(video_path):
            return jsonify({
                'error': f'Video file not found: {video_path}',
                'video_id': video_id
            }), 404
        
        # Initialize enhanced processor
        processor = EnhancedVideoProcessor(
            use_gpu=False,
            detection_threshold=0.2  # Lower threshold for targeted extraction
        )
        
        # Create output directory
        output_dir = f"faces/{video_id}_targeted"
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract at specific timestamps
        detections = processor.extract_specific_timestamps(
            video_path, timestamps, output_dir, video_id
        )
        
        # Prepare results
        results = []
        for detection in detections:
            # Convert numpy types to Python native types for JSON serialization
            bbox = detection['bbox']
            if hasattr(bbox[0], 'item'):  # numpy type
                bbox = tuple(int(coord.item()) if hasattr(coord, 'item') else int(coord) for coord in bbox)
            
            results.append({
                'timestamp': float(detection['frame_timestamp']),
                'bbox': bbox,
                'confidence': float(detection['confidence']),
                'mask_status': detection['mask_status'],
                'mask_confidence': float(detection['mask_confidence']),
                'mask_type': detection.get('mask_type'),
                'face_image_path': detection['face_image_path']
            })
        
        masked_count = sum(1 for r in results if r['mask_status'] == 'masked')
        
        return jsonify({
            'video_id': video_id,
            'timestamps': timestamps,
            'total_faces': len(results),
            'masked_faces': masked_count,
            'results': results
        })
        
    except Exception as e:
        logger.error(f"Error in enhanced timestamp extraction: {e}")
        return jsonify({
            'error': str(e),
            'video_id': video_id
        }), 500