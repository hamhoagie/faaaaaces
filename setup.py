#!/usr/bin/env python3
"""
Setup script for FAAAAACES application
"""
import os
import sys
import subprocess

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Command: {command}")
        print(f"   Error: {e.stderr}")
        return False

def setup_environment():
    """Setup the application environment"""
    print("🚀 Setting up FAAAAACES application...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Create directories
    print("📁 Creating directories...")
    directories = ['uploads', 'faces', 'temp', 'app/static/css', 'app/static/js']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"   Created: {directory}")
    
    # Create .env file if it doesn't exist
    if not os.path.exists('.env'):
        print("📝 Creating .env file...")
        with open('.env', 'w') as f:
            f.write("""# FAAAAACES Configuration
FLASK_ENV=development
FLASK_DEBUG=True
SECRET_KEY=dev-secret-key-change-in-production

# Database
DATABASE_URL=sqlite:///faaaaaces.db

# Video Processing
MAX_VIDEO_SIZE_MB=500
FRAME_EXTRACTION_INTERVAL=30
MAX_FACES_PER_VIDEO=1000

# Face Recognition
FACE_CONFIDENCE_THRESHOLD=0.7
CLUSTERING_SIMILARITY_THRESHOLD=0.6
FACE_IMAGE_SIZE=224

# File Storage
UPLOAD_FOLDER=uploads
FACES_FOLDER=faces
TEMP_FOLDER=temp

# Video Download
MAX_DOWNLOAD_SIZE_MB=1000
DOWNLOAD_TIMEOUT_SECONDS=300
""")
        print("✅ Created .env file with default settings")
    else:
        print("✅ .env file already exists")
    
    # Install Python dependencies
    if not run_command("pip install -r requirements.txt", "Installing Python dependencies"):
        print("\n💡 Try running with --user flag if you get permission errors:")
        print("   pip install --user -r requirements.txt")
        return False
    
    # Test DeepFace installation
    print("🧪 Testing DeepFace installation...")
    try:
        import deepface
        print("✅ DeepFace imported successfully")
        
        # Preload a model to test
        print("📥 Downloading face recognition models (this may take a few minutes)...")
        from deepface import DeepFace
        import numpy as np
        
        # Create a dummy image to initialize models
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        DeepFace.represent(dummy_img, model_name='VGG-Face', detector_backend='opencv', enforce_detection=False)
        print("✅ Face recognition models loaded successfully")
        
    except ImportError:
        print("❌ DeepFace installation failed")
        return False
    except Exception as e:
        print(f"⚠️ DeepFace model loading failed: {e}")
        print("   Models will be downloaded on first use")
    
    # Test other critical imports
    print("🧪 Testing other dependencies...")
    try:
        import cv2
        import yt_dlp
        import sklearn
        import flask
        print("✅ All critical dependencies loaded successfully")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False
    
    # Initialize database
    print("🗄️ Initializing database...")
    try:
        from app.models.database import init_db
        init_db()
        print("✅ Database initialized successfully")
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        return False
    
    print("\n🎉 Setup completed successfully!")
    print("\n🚀 To start the application:")
    print("   python run.py")
    print("\n🌐 Then visit: http://localhost:5000")
    
    return True

def main():
    """Main setup function"""
    if len(sys.argv) > 1 and sys.argv[1] == '--help':
        print("""
FAAAAACES Setup Script

Usage: python setup.py [options]

Options:
  --help     Show this help message
  
This script will:
1. Check Python version compatibility
2. Create necessary directories
3. Create .env configuration file
4. Install Python dependencies
5. Test DeepFace installation
6. Initialize the database

Requirements:
- Python 3.8 or higher
- pip (Python package installer)
- Internet connection (for downloading dependencies and models)
""")
        return
    
    try:
        success = setup_environment()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n❌ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error during setup: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()