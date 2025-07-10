"""
API routes for video processing and face recognition
"""
import os
import tempfile
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
from app.models.database import VideoModel, FaceModel, ProcessingJobModel
from app.services.video_downloader import VideoDownloader
from app.services.video_processor import VideoProcessor
# Try to import DeepFace version, fallback to basic version
try:
    from app.services.face_extractor import FaceExtractor
    DEEPFACE_AVAILABLE = True
except ImportError:
    from app.services.face_extractor_basic import BasicFaceExtractor as FaceExtractor
    DEEPFACE_AVAILABLE = False
    print("⚠️  DeepFace not available, using basic OpenCV face detection")
from app.services.face_clustering import FaceClusterer

api_bp = Blueprint('api', __name__)

# Initialize services
video_downloader = VideoDownloader()
video_processor = VideoProcessor()
face_extractor = FaceExtractor()
face_clusterer = FaceClusterer()

@api_bp.route('/upload_video', methods=['POST'])
def upload_video():
    """Handle video file upload"""
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Validate file extension
        allowed_extensions = video_processor.get_supported_formats()
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in allowed_extensions:
            return jsonify({'error': f'Unsupported file format: {file_ext}'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Validate video file
        is_valid, message = video_processor.validate_video_file(file_path)
        if not is_valid:
            os.remove(file_path)
            return jsonify({'error': f'Invalid video file: {message}'}), 400
        
        # Get video metadata
        video_info = video_processor.get_video_info(file_path)
        
        # Create video record in database
        video_id = VideoModel.create(
            filename=filename,
            source_type='upload',
            **video_info
        )
        
        # Create processing job
        job_id = ProcessingJobModel.create('video_processing', video_id)
        
        # Start processing (in a real app, this would be queued)
        process_video_async(video_id, file_path)
        
        return jsonify({
            'video_id': video_id,
            'job_id': job_id,
            'message': 'Video uploaded successfully and processing started'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/process_url', methods=['POST'])
def process_url():
    """Process video from URL"""
    try:
        data = request.get_json()
        if not data or 'url' not in data:
            return jsonify({'error': 'URL is required'}), 400
        
        url = data['url']
        
        # Validate URL format
        is_valid, platform, validation_error = video_downloader.validate_url_format(url)
        if not is_valid:
            return jsonify({'error': validation_error}), 400
        
        # Download video
        success, file_path, video_info, error_msg = video_downloader.download_from_url(url)
        
        if not success:
            # Try direct download as fallback
            success, file_path, video_info, error_msg = video_downloader.download_direct_video(url)
        
        if not success:
            return jsonify({'error': f'Failed to download video: {error_msg}'}), 400
        
        # Get additional video metadata
        try:
            additional_info = video_processor.get_video_info(file_path)
            video_info.update(additional_info)
        except Exception as e:
            print(f"Warning: Could not get additional video info: {e}")
        
        # Create video record
        video_id = VideoModel.create(
            filename=os.path.basename(file_path),
            source_type='url',
            **video_info
        )
        
        # Create processing job
        job_id = ProcessingJobModel.create('video_processing', video_id)
        
        # Start processing
        process_video_async(video_id, file_path)
        
        return jsonify({
            'video_id': video_id,
            'job_id': job_id,
            'message': 'Video downloaded successfully and processing started'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/process_urls', methods=['POST'])
def process_multiple_urls():
    """Process multiple videos from URLs"""
    try:
        data = request.get_json()
        if not data or 'urls' not in data:
            return jsonify({'error': 'URLs list is required'}), 400
        
        urls = data['urls']
        if not isinstance(urls, list):
            return jsonify({'error': 'URLs must be a list'}), 400
        
        results = []
        
        for url in urls:
            try:
                # Process each URL (simplified version)
                success, file_path, video_info, error_msg = video_downloader.download_from_url(url)
                
                if success:
                    # Create video record
                    video_id = VideoModel.create(
                        filename=os.path.basename(file_path),
                        source_type='url',
                        **video_info
                    )
                    
                    job_id = ProcessingJobModel.create('video_processing', video_id)
                    process_video_async(video_id, file_path)
                    
                    results.append({
                        'url': url,
                        'success': True,
                        'video_id': video_id,
                        'job_id': job_id
                    })
                else:
                    results.append({
                        'url': url,
                        'success': False,
                        'error': error_msg
                    })
                    
            except Exception as e:
                results.append({
                    'url': url,
                    'success': False,
                    'error': str(e)
                })
        
        return jsonify({
            'results': results,
            'total': len(urls),
            'successful': len([r for r in results if r['success']])
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/cluster_faces', methods=['POST'])
def cluster_faces():
    """Trigger face clustering"""
    try:
        # Create clustering job
        job_id = ProcessingJobModel.create('face_clustering')
        
        # Run clustering
        ProcessingJobModel.update_status(job_id, 'running', 0)
        
        results = face_clusterer.cluster_all_faces()
        
        ProcessingJobModel.update_status(job_id, 'completed', 100)
        
        return jsonify({
            'job_id': job_id,
            'results': results,
            'message': 'Face clustering completed'
        })
        
    except Exception as e:
        ProcessingJobModel.update_status(job_id, 'failed', 0, str(e))
        return jsonify({'error': str(e)}), 500

@api_bp.route('/job_status/<int:job_id>')
def get_job_status(job_id):
    """Get status of a processing job"""
    try:
        # Implementation to get job status from database
        return jsonify({
            'job_id': job_id,
            'status': 'running',  # Placeholder
            'progress': 50  # Placeholder
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/video_info/<int:video_id>')
def get_video_info(video_id):
    """Get video information and processing status"""
    try:
        video = VideoModel.get_by_id(video_id)
        if not video:
            return jsonify({'error': 'Video not found'}), 404
        
        faces = FaceModel.get_by_video(video_id)
        
        return jsonify({
            'video': video,
            'face_count': len(faces),
            'faces': faces[:10]  # Return first 10 faces as preview
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/clusters')
def get_clusters():
    """Get all face clusters"""
    try:
        clusters = FaceClusterModel.get_all_with_faces()
        return jsonify({'clusters': clusters})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/supported_platforms')
def get_supported_platforms():
    """Get supported platforms and tips"""
    try:
        platforms = video_downloader.get_supported_platforms()
        return jsonify({
            'platforms': platforms,
            'tips': {
                'instagram': video_downloader.get_platform_tips('instagram'),
                'tiktok': video_downloader.get_platform_tips('tiktok'),
                'youtube': video_downloader.get_platform_tips('youtube')
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/validate_url', methods=['POST'])
def validate_url():
    """Validate URL and detect platform"""
    try:
        data = request.get_json()
        if not data or 'url' not in data:
            return jsonify({'error': 'URL is required'}), 400
        
        url = data['url']
        is_valid, platform, error_message = video_downloader.validate_url_format(url)
        
        response = {
            'valid': is_valid,
            'platform': platform,
            'message': error_message or f"Valid {platform} URL detected"
        }
        
        if is_valid:
            response['tips'] = video_downloader.get_platform_tips(platform)
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def process_video_async(video_id: int, video_path: str):
    """
    Process video to extract faces (simplified synchronous version)
    In production, this should be run in a background task queue (Celery)
    """
    try:
        # Update video status
        VideoModel.update_status(video_id, 'processing')
        
        # Extract frames
        temp_dir = tempfile.mkdtemp()
        frame_paths = list(video_processor.extract_frames(video_path, temp_dir))
        
        if not frame_paths:
            VideoModel.update_status(video_id, 'failed', 'No frames extracted')
            return
        
        # Extract faces from frames
        faces_dir = os.path.join(current_app.config['FACES_FOLDER'], str(video_id))
        face_data = face_extractor.process_video_frames(
            [fp[1] for fp in frame_paths], 
            faces_dir, 
            video_id
        )
        
        # Save faces to database
        for face in face_data:
            FaceModel.create(
                video_id=face['video_id'],
                frame_timestamp=face['frame_timestamp'],
                face_image_path=face['face_image_path'],
                bbox=face['bbox'],
                confidence=face['confidence'],
                embedding=face['embedding']
            )
        
        # Cleanup
        video_processor.cleanup_frames([fp[1] for fp in frame_paths])
        video_downloader.cleanup_file(video_path)
        
        # Update status
        VideoModel.update_status(video_id, 'completed')
        
        # Trigger clustering if we have enough faces
        if len(face_data) > 0:
            face_clusterer.cluster_all_faces()
        
    except Exception as e:
        VideoModel.update_status(video_id, 'failed', str(e))
        print(f"Error processing video {video_id}: {e}")