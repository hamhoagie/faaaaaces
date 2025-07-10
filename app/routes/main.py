"""
Main web interface routes
"""
from flask import Blueprint, render_template, request, jsonify, current_app, send_from_directory
from app.models.database import VideoModel, FaceClusterModel, ProcessingJobModel
import os

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    """Main dashboard page"""
    # Get recent videos and clusters for display
    videos = VideoModel.get_all()[:10]  # Last 10 videos
    clusters = FaceClusterModel.get_all_with_faces()[:20]  # Top 20 clusters
    active_jobs = ProcessingJobModel.get_active_jobs()
    
    return render_template('index.html', 
                         videos=videos, 
                         clusters=clusters,
                         active_jobs=active_jobs)

@main_bp.route('/upload')
def upload_page():
    """Video upload page"""
    return render_template('upload.html')

@main_bp.route('/clusters')
def clusters_page():
    """Face clusters gallery page"""
    clusters = FaceClusterModel.get_all_with_faces()
    return render_template('clusters.html', clusters=clusters)

@main_bp.route('/videos')
def videos_page():
    """Videos list page"""
    videos = VideoModel.get_all()
    return render_template('videos.html', videos=videos)

@main_bp.route('/video/<int:video_id>')
def video_detail(video_id):
    """Individual video detail page"""
    from app.models.database import FaceModel
    
    video = VideoModel.get_by_id(video_id)
    if not video:
        return "Video not found", 404
    
    faces = FaceModel.get_by_video(video_id)
    return render_template('video_detail.html', video=video, faces=faces)

@main_bp.route('/cluster/<int:cluster_id>')
def cluster_detail(cluster_id):
    """Individual cluster detail page"""
    # Implementation would show all faces in a cluster
    return render_template('cluster_detail.html', cluster_id=cluster_id)

@main_bp.route('/faces/<path:filename>')
def serve_face_image(filename):
    """Serve face images from the faces directory"""
    # The filename already includes the relative path from project root
    # e.g., filename = "1/video_1_frame_30.00s_face_0.jpg"
    # So we serve from the project root directory
    project_root = os.path.dirname(current_app.root_path)
    faces_dir = os.path.join(project_root, 'faces')
    full_path = os.path.join(faces_dir, filename)
    print(f"DEBUG: Trying to serve face image - filename: {filename}, faces_dir: {faces_dir}, full_path: {full_path}, exists: {os.path.exists(full_path)}")
    return send_from_directory(faces_dir, filename)