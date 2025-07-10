"""
FAAAAACES - Face Recognition Video Processing Application
"""
from flask import Flask
from flask_cors import CORS
import os
from dotenv import load_dotenv

load_dotenv()

def create_app():
    app = Flask(__name__)
    
    # Configuration
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key')
    app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER', 'uploads')
    app.config['FACES_FOLDER'] = os.getenv('FACES_FOLDER', 'faces')
    app.config['TEMP_FOLDER'] = os.getenv('TEMP_FOLDER', 'temp')
    app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_VIDEO_SIZE_MB', 500)) * 1024 * 1024
    
    # Enable CORS
    CORS(app)
    
    # Create necessary directories
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['FACES_FOLDER'], exist_ok=True)
    os.makedirs(app.config['TEMP_FOLDER'], exist_ok=True)
    
    # Initialize database
    from app.models.database import init_db
    init_db()
    
    # Register blueprints
    from app.routes.main import main_bp
    from app.routes.api import api_bp
    
    app.register_blueprint(main_bp)
    app.register_blueprint(api_bp, url_prefix='/api')
    
    return app