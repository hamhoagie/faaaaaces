"""
API routes for mask detection and face reconstruction
"""
from flask import Blueprint, request, jsonify, current_app
from app.models.database import FaceModel, VideoModel
from app.services.mask_detector import MaskDetector
from app.services.face_reconstructor import FaceReconstructor
import os

mask_api_bp = Blueprint('mask_api', __name__)

# Initialize services
mask_detector = MaskDetector()
face_reconstructor = FaceReconstructor()

@mask_api_bp.route('/mask_statistics')
def get_mask_statistics():
    """Get overall mask detection statistics"""
    try:
        stats = FaceModel.get_mask_statistics()
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@mask_api_bp.route('/masked_faces')
def get_masked_faces():
    """Get all masked faces"""
    try:
        video_id = request.args.get('video_id', type=int)
        masked_faces = FaceModel.get_masked_faces(video_id)
        
        return jsonify({
            'masked_faces': masked_faces,
            'count': len(masked_faces)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@mask_api_bp.route('/reconstructed_faces')
def get_reconstructed_faces():
    """Get all reconstructed faces"""
    try:
        video_id = request.args.get('video_id', type=int)
        reconstructed_faces = FaceModel.get_reconstructed_faces(video_id)
        
        return jsonify({
            'reconstructed_faces': reconstructed_faces,
            'count': len(reconstructed_faces)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@mask_api_bp.route('/detect_masks/<int:video_id>', methods=['POST'])
def detect_masks_in_video(video_id):
    """Detect masks in all faces of a specific video"""
    try:
        video = VideoModel.get_by_id(video_id)
        if not video:
            return jsonify({'error': 'Video not found'}), 404
        
        # Get all faces for the video
        faces = FaceModel.get_by_video(video_id)
        if not faces:
            return jsonify({'error': 'No faces found for this video'}), 404
        
        # Detect masks in all faces
        results = []
        for face in faces:
            face_image_path = face['face_image_path']
            if os.path.exists(face_image_path):
                import cv2
                face_image = cv2.imread(face_image_path)
                if face_image is not None:
                    mask_result = mask_detector.detect_mask(face_image)
                    results.append({
                        'face_id': face['id'],
                        'face_image_path': face_image_path,
                        **mask_result
                    })
        
        # Count masked faces
        masked_count = sum(1 for r in results if r['is_masked'])
        
        return jsonify({
            'video_id': video_id,
            'total_faces': len(results),
            'masked_faces': masked_count,
            'unmasked_faces': len(results) - masked_count,
            'results': results
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@mask_api_bp.route('/detect_masks_improved/<int:video_id>', methods=['POST'])
def detect_masks_improved(video_id):
    """Detect masks using improved algorithm with confidence filtering"""
    try:
        min_confidence = request.json.get('min_confidence', 0.7) if request.json else 0.7
        
        video = VideoModel.get_by_id(video_id)
        if not video:
            return jsonify({'error': 'Video not found'}), 404
        
        # Get all faces for the video
        faces = FaceModel.get_by_video(video_id)
        if not faces:
            return jsonify({'error': 'No faces found for this video'}), 404
        
        # Reprocess with improved detection
        processing_results = mask_detector.reprocess_all_faces(faces, min_confidence)
        
        return jsonify({
            'video_id': video_id,
            'min_confidence_threshold': min_confidence,
            'total_faces': processing_results['total_faces'],
            'masked_faces': len(processing_results['masked_faces']),
            'unmasked_faces': len(processing_results['unmasked_faces']),
            'processing_errors': len(processing_results['processing_errors']),
            'confidence_distribution': processing_results['confidence_distribution'],
            'masked_faces_data': processing_results['masked_faces'],
            'errors': processing_results['processing_errors']
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@mask_api_bp.route('/compare_detection_methods/<int:video_id>', methods=['POST'])
def compare_detection_methods(video_id):
    """Compare old vs new mask detection methods"""
    try:
        min_confidence = request.json.get('min_confidence', 0.7) if request.json else 0.7
        
        video = VideoModel.get_by_id(video_id)
        if not video:
            return jsonify({'error': 'Video not found'}), 404
        
        faces = FaceModel.get_by_video(video_id)
        if not faces:
            return jsonify({'error': 'No faces found for this video'}), 404
        
        comparison_results = []
        
        for face in faces[:10]:  # Limit to first 10 faces for testing
            face_image_path = face['face_image_path']
            if os.path.exists(face_image_path):
                import cv2
                face_image = cv2.imread(face_image_path)
                if face_image is not None:
                    # Old method (basic CV)
                    old_result = mask_detector.detect_mask_cv(face_image)
                    
                    # New method (improved with confidence filtering)
                    new_result = mask_detector.detect_mask(face_image, min_confidence)
                    
                    comparison_results.append({
                        'face_id': face['id'],
                        'face_image_path': face_image_path,
                        'old_method': {
                            'is_masked': old_result[0],
                            'confidence': old_result[1],
                            'method': 'basic_cv'
                        },
                        'new_method': {
                            'is_masked': new_result['is_masked'],
                            'confidence': new_result['confidence'],
                            'method': new_result['method'],
                            'detection_details': new_result['detection_details']
                        },
                        'changed': old_result[0] != new_result['is_masked']
                    })
        
        # Calculate summary statistics
        changed_count = sum(1 for r in comparison_results if r['changed'])
        old_masked_count = sum(1 for r in comparison_results if r['old_method']['is_masked'])
        new_masked_count = sum(1 for r in comparison_results if r['new_method']['is_masked'])
        
        return jsonify({
            'video_id': video_id,
            'faces_compared': len(comparison_results),
            'min_confidence_threshold': min_confidence,
            'summary': {
                'faces_with_changed_detection': changed_count,
                'old_method_masked_count': old_masked_count,
                'new_method_masked_count': new_masked_count,
                'reduction_in_false_positives': max(0, old_masked_count - new_masked_count)
            },
            'detailed_results': comparison_results
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@mask_api_bp.route('/reconstruct_faces/<int:video_id>', methods=['POST'])
def reconstruct_faces_in_video(video_id):
    """Reconstruct masked faces in a specific video"""
    try:
        video = VideoModel.get_by_id(video_id)
        if not video:
            return jsonify({'error': 'Video not found'}), 404
        
        # Get masked faces for the video
        masked_faces = FaceModel.get_masked_faces(video_id)
        if not masked_faces:
            return jsonify({'error': 'No masked faces found for this video'}), 404
        
        # Reconstruct faces
        reconstructed_results = []
        for face in masked_faces:
            face_image_path = face['face_image_path']
            if os.path.exists(face_image_path):
                import cv2
                face_image = cv2.imread(face_image_path)
                if face_image is not None:
                    reconstruction_result = face_reconstructor.reconstruct_masked_face(face_image)
                    
                    # Save reconstructed image if successful
                    if reconstruction_result['success']:
                        # Generate path for reconstructed image
                        base_dir = os.path.dirname(face_image_path)
                        recon_dir = os.path.join(base_dir, 'reconstructed')
                        os.makedirs(recon_dir, exist_ok=True)
                        
                        original_filename = os.path.basename(face_image_path)
                        name, ext = os.path.splitext(original_filename)
                        recon_filename = f"{name}_reconstructed{ext}"
                        recon_path = os.path.join(recon_dir, recon_filename)
                        
                        cv2.imwrite(recon_path, reconstruction_result['reconstructed_image'])
                        reconstruction_result['reconstructed_image_path'] = recon_path
                    
                    reconstructed_results.append({
                        'face_id': face['id'],
                        'original_path': face_image_path,
                        **reconstruction_result
                    })
        
        # Count successful reconstructions
        success_count = sum(1 for r in reconstructed_results if r['success'])
        
        return jsonify({
            'video_id': video_id,
            'total_masked_faces': len(masked_faces),
            'successful_reconstructions': success_count,
            'failed_reconstructions': len(reconstructed_results) - success_count,
            'results': reconstructed_results
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@mask_api_bp.route('/reprocess_video_with_masks/<int:video_id>', methods=['POST'])
def reprocess_video_with_masks(video_id):
    """Reprocess a video with mask detection and reconstruction enabled"""
    try:
        video = VideoModel.get_by_id(video_id)
        if not video:
            return jsonify({'error': 'Video not found'}), 404
        
        # This would trigger a full reprocessing with mask detection
        # For now, return a placeholder response
        return jsonify({
            'message': 'Video reprocessing with mask detection started',
            'video_id': video_id,
            'status': 'pending'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@mask_api_bp.route('/mask_detection_health')
def mask_detection_health():
    """Check health of mask detection and reconstruction services"""
    try:
        # Test mask detector
        mask_detector_status = {
            'available': hasattr(mask_detector, 'model'),
            'method': 'machine_learning' if hasattr(mask_detector, 'model') and mask_detector.model else 'computer_vision'
        }
        
        # Test face reconstructor
        reconstructor_status = {
            'available': hasattr(face_reconstructor, 'model'),
            'model_type': face_reconstructor.model_type,
            'model_loaded': face_reconstructor.model_loaded
        }
        
        return jsonify({
            'mask_detector': mask_detector_status,
            'face_reconstructor': reconstructor_status,
            'services_ready': True
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'services_ready': False
        }), 500

@mask_api_bp.route('/install_reconstruction_models', methods=['POST'])
def install_reconstruction_models():
    """Install or download required models for face reconstruction"""
    try:
        # This would trigger model downloads
        # For now, return installation instructions
        
        dependencies = [
            "gfpgan",
            "basicsr", 
            "facexlib",
            "realesrgan",
            "scikit-image"
        ]
        
        return jsonify({
            'message': 'Model installation instructions',
            'dependencies': dependencies,
            'install_commands': [f"pip install {dep}" for dep in dependencies],
            'note': 'Manual installation required for now'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500