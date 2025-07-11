"""
Masked Faces Gallery API
Dedicated endpoints for displaying and managing detected masked faces.
"""

from flask import Blueprint, jsonify, request
import os
import glob
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

masked_faces_api_bp = Blueprint('masked_faces_api', __name__)

@masked_faces_api_bp.route('/all_masked_faces', methods=['GET'])
def get_all_masked_faces():
    """
    Get all detected masked faces from the enhanced detection results.
    This reads from the faces directories to find all saved masked faces.
    """
    try:
        masked_faces = []
        base_faces_dir = "faces"
        
        # Look for enhanced detection directories (with "_enhanced" or "_targeted" suffix)
        enhanced_dirs = []
        if os.path.exists(base_faces_dir):
            for item in os.listdir(base_faces_dir):
                item_path = os.path.join(base_faces_dir, item)
                if os.path.isdir(item_path) and ("_enhanced" in item or "_targeted" in item):
                    enhanced_dirs.append(item)
        
        # Also check the regular video directories for any faces
        regular_dirs = [d for d in os.listdir(base_faces_dir) 
                       if os.path.isdir(os.path.join(base_faces_dir, d)) 
                       and d.isdigit()]
        
        all_dirs = enhanced_dirs + regular_dirs
        
        for dir_name in all_dirs:
            dir_path = os.path.join(base_faces_dir, dir_name)
            
            # Extract video ID from directory name
            if "_" in dir_name:
                video_id = dir_name.split("_")[0]
            else:
                video_id = dir_name
            
            # Find all face images in this directory
            face_files = glob.glob(os.path.join(dir_path, "*.jpg"))
            
            for face_file in face_files:
                filename = os.path.basename(face_file)
                
                # Parse filename to extract timestamp and face info
                # Format: video_X_frame_Y.YYs_face_Z.jpg
                try:
                    parts = filename.replace('.jpg', '').split('_')
                    if len(parts) >= 5:
                        timestamp_part = parts[3]  # e.g., "15.00s"
                        timestamp = float(timestamp_part.replace('s', ''))
                        face_index = int(parts[5])
                        
                        # For now, we'll assume faces in enhanced directories are potentially masked
                        # In a real implementation, we'd store this metadata in the database
                        face_data = {
                            'video_id': int(video_id),
                            'face_id': f"{video_id}_{timestamp}_{face_index}",
                            'filename': filename,
                            'image_path': f"/faces/{dir_name}/{filename}",
                            'timestamp': timestamp,
                            'face_index': face_index,
                            'directory': dir_name,
                            'is_enhanced_detection': "_enhanced" in dir_name or "_targeted" in dir_name,
                            'file_size': os.path.getsize(face_file),
                            'last_modified': os.path.getmtime(face_file)
                        }
                        
                        masked_faces.append(face_data)
                        
                except (ValueError, IndexError) as e:
                    logger.warning(f"Could not parse filename {filename}: {e}")
                    continue
        
        # Sort by video ID and timestamp
        masked_faces.sort(key=lambda x: (x['video_id'], x['timestamp']))
        
        # Group by video for easier display
        videos_with_faces = {}
        for face in masked_faces:
            video_id = face['video_id']
            if video_id not in videos_with_faces:
                videos_with_faces[video_id] = {
                    'video_id': video_id,
                    'faces': []
                }
            videos_with_faces[video_id]['faces'].append(face)
        
        return jsonify({
            'total_faces': len(masked_faces),
            'videos_count': len(videos_with_faces),
            'videos': list(videos_with_faces.values()),
            'all_faces': masked_faces
        })
        
    except Exception as e:
        logger.error(f"Error getting masked faces: {e}")
        return jsonify({'error': str(e)}), 500

@masked_faces_api_bp.route('/reconstruct_single_face', methods=['POST'])
def reconstruct_single_face():
    """
    Reconstruct a single masked face given its face_id and method.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        face_id = data.get('face_id')
        method = data.get('method', 'classical')
        face_path = data.get('face_path')
        
        if not face_id or not face_path:
            return jsonify({'error': 'face_id and face_path are required'}), 400
        
        # Import reconstruction service
        from ..services.face_reconstructor import FaceReconstructor
        
        # Initialize reconstructor
        reconstructor = FaceReconstructor(model_type=method)
        
        # Load the face image
        import cv2
        import numpy as np
        
        # Convert relative path to absolute path
        if face_path.startswith('/faces/'):
            actual_path = face_path[1:]  # Remove leading slash
        else:
            actual_path = face_path
        
        if not os.path.exists(actual_path):
            return jsonify({'error': f'Face image not found: {actual_path}'}), 404
        
        face_image = cv2.imread(actual_path)
        if face_image is None:
            return jsonify({'error': f'Could not load face image: {actual_path}'}), 400
        
        # Perform reconstruction
        reconstruction_result = reconstructor.reconstruct_masked_face(face_image)
        
        if not reconstruction_result.get('success', False):
            return jsonify({
                'error': 'Reconstruction failed',
                'details': reconstruction_result.get('error', 'Unknown error')
            }), 500
        
        # Save reconstructed image
        reconstructed_image = reconstruction_result['reconstructed_image']
        
        # Create output directory
        output_dir = f"faces/reconstructed/{face_id.replace('/', '_')}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save reconstructed image
        output_filename = f"reconstructed_{method}_{face_id.replace('/', '_')}.jpg"
        output_path = os.path.join(output_dir, output_filename)
        
        success = cv2.imwrite(output_path, reconstructed_image)
        if not success:
            return jsonify({'error': 'Failed to save reconstructed image'}), 500
        
        return jsonify({
            'success': True,
            'face_id': face_id,
            'method': method,
            'original_path': face_path,
            'reconstructed_path': f"/faces/reconstructed/{face_id.replace('/', '_')}/{output_filename}",
            'quality_score': reconstruction_result.get('quality', 0.0),
            'reconstruction_time': reconstruction_result.get('processing_time', 0.0)
        })
        
    except Exception as e:
        logger.error(f"Error reconstructing face: {e}")
        return jsonify({'error': str(e)}), 500

@masked_faces_api_bp.route('/batch_reconstruct', methods=['POST'])
def batch_reconstruct_faces():
    """
    Reconstruct multiple masked faces in batch.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        face_ids = data.get('face_ids', [])
        method = data.get('method', 'classical')
        
        if not face_ids:
            return jsonify({'error': 'No face_ids provided'}), 400
        
        results = []
        errors = []
        
        for face_id in face_ids:
            try:
                # Call single face reconstruction for each face
                single_result = reconstruct_single_face_internal(face_id, method)
                if single_result.get('success'):
                    results.append(single_result)
                else:
                    errors.append({
                        'face_id': face_id,
                        'error': single_result.get('error', 'Unknown error')
                    })
            except Exception as e:
                errors.append({
                    'face_id': face_id,
                    'error': str(e)
                })
        
        return jsonify({
            'success': True,
            'total_requested': len(face_ids),
            'successful_reconstructions': len(results),
            'failed_reconstructions': len(errors),
            'results': results,
            'errors': errors,
            'method': method
        })
        
    except Exception as e:
        logger.error(f"Error in batch reconstruction: {e}")
        return jsonify({'error': str(e)}), 500

def reconstruct_single_face_internal(face_id, method):
    """
    Internal function for single face reconstruction (used by batch processing).
    """
    try:
        # First, find the face data by face_id
        face_data = None
        
        # Get all masked faces to find the one with matching face_id
        masked_faces = []
        base_faces_dir = "faces"
        
        # Look for enhanced detection directories and regular directories
        enhanced_dirs = []
        if os.path.exists(base_faces_dir):
            for item in os.listdir(base_faces_dir):
                item_path = os.path.join(base_faces_dir, item)
                if os.path.isdir(item_path) and ("_enhanced" in item or "_targeted" in item):
                    enhanced_dirs.append(item)
        
        regular_dirs = [d for d in os.listdir(base_faces_dir) 
                       if os.path.isdir(os.path.join(base_faces_dir, d)) 
                       and d.isdigit()]
        
        all_dirs = enhanced_dirs + regular_dirs
        
        # Find the face with matching face_id
        for dir_name in all_dirs:
            dir_path = os.path.join(base_faces_dir, dir_name)
            
            # Extract video ID from directory name
            if "_" in dir_name:
                video_id = dir_name.split("_")[0]
            else:
                video_id = dir_name
            
            # Find all face images in this directory
            face_files = glob.glob(os.path.join(dir_path, "*.jpg"))
            
            for face_file in face_files:
                filename = os.path.basename(face_file)
                
                # Parse filename to extract timestamp and face info
                try:
                    parts = filename.replace('.jpg', '').split('_')
                    if len(parts) >= 5:
                        timestamp_part = parts[3]  # e.g., "15.00s"
                        timestamp = float(timestamp_part.replace('s', ''))
                        face_index = int(parts[5])
                        
                        # Generate face_id for comparison
                        current_face_id = f"{video_id}_{timestamp}_{face_index}"
                        
                        if current_face_id == face_id:
                            face_data = {
                                'face_id': face_id,
                                'image_path': face_file,  # Use full path for cv2.imread
                                'relative_path': f"/faces/{dir_name}/{filename}",
                                'timestamp': timestamp,
                                'face_index': face_index,
                                'directory': dir_name,
                                'video_id': int(video_id)
                            }
                            break
                except (ValueError, IndexError):
                    continue
            
            if face_data:
                break
        
        if not face_data:
            return {
                'success': False,
                'face_id': face_id,
                'method': method,
                'error': f'Face with ID {face_id} not found'
            }
        
        # Import reconstruction service
        from ..services.face_reconstructor import FaceReconstructor
        
        # Initialize reconstructor
        reconstructor = FaceReconstructor(model_type=method)
        
        # Load the face image
        import cv2
        
        face_image = cv2.imread(face_data['image_path'])
        if face_image is None:
            return {
                'success': False,
                'face_id': face_id,
                'method': method,
                'error': f'Could not load face image: {face_data["image_path"]}'
            }
        
        # Perform reconstruction
        reconstruction_result = reconstructor.reconstruct_masked_face(face_image)
        
        if not reconstruction_result.get('success', False):
            return {
                'success': False,
                'face_id': face_id,
                'method': method,
                'error': reconstruction_result.get('error', 'Reconstruction failed')
            }
        
        # Save reconstructed image
        reconstructed_image = reconstruction_result['reconstructed_image']
        
        # Create output directory
        output_dir = f"faces/reconstructed/{face_id.replace('/', '_')}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save reconstructed image
        output_filename = f"reconstructed_{method}_{face_id.replace('/', '_')}.jpg"
        output_path = os.path.join(output_dir, output_filename)
        
        success = cv2.imwrite(output_path, reconstructed_image)
        if not success:
            return {
                'success': False,
                'face_id': face_id,
                'method': method,
                'error': 'Failed to save reconstructed image'
            }
        
        return {
            'success': True,
            'face_id': face_id,
            'method': method,
            'original_path': face_data['relative_path'],
            'reconstructed_path': f"/faces/reconstructed/{face_id.replace('/', '_')}/{output_filename}",
            'quality_score': reconstruction_result.get('quality_score', 0.0),
            'processing_time': reconstruction_result.get('processing_time', 0.0)
        }
        
    except Exception as e:
        logger.error(f"Error in reconstruct_single_face_internal: {e}")
        return {
            'success': False,
            'face_id': face_id,
            'method': method,
            'error': str(e)
        }