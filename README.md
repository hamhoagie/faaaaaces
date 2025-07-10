# FAAAAACES - Face Recognition Video Processing Application

A Python web application that processes videos to extract and catalog faces using **DeepFace**, **TensorFlow**, and machine learning clustering. Supports multiple video platforms including **YouTube**, **Instagram**, **TikTok**, and direct uploads.

## Features

### 🎥 Video Input Methods
- **File Upload**: Direct upload of video files (MP4, AVI, MOV, etc.)
- **URL Processing**: Download and process videos from multiple platforms:
  - 🔴 **YouTube** (youtube.com, youtu.be)
  - 📸 **Instagram** (posts, reels, IGTV)
  - 🎵 **TikTok** (videos, including short links)
  - 🎬 **Vimeo** and many other platforms
- **Batch Processing**: Process multiple URLs simultaneously from different platforms

### 🔍 Face Processing Pipeline
- **🤖 Advanced AI Models**: Full DeepFace integration with TensorFlow 2.19.0
- **🎯 Multiple Detection Backends**: RetinaFace, MTCNN, OpenCV, SSD, dlib
- **🧠 Neural Network Models**: VGG-Face, Facenet, OpenFace, DeepFace, ArcFace
- **📐 High-Quality Embeddings**: Generate 128/512-dimensional face embeddings
- **🔗 Smart Clustering**: Group similar faces using DBSCAN and Agglomerative clustering
- **⚡ Confidence Scoring**: Quality assessment for each detected face
- **🛡️ Fallback Support**: Basic OpenCV detection when DeepFace unavailable

### 🌐 Web Interface
- **📊 Modern Dashboard**: Clean, responsive interface built with Bootstrap
- **📤 Upload Interface**: Drag-and-drop file upload and URL input with platform detection
- **🖼️ Face Gallery**: Browse discovered faces organized by clusters with confidence scores
- **⏱️ Real-time Processing**: Live updates on video processing status and progress
- **📱 Mobile Responsive**: Works seamlessly on desktop and mobile devices
- **🏷️ Platform Badges**: Visual indicators for Instagram, TikTok, YouTube sources

## Quick Start

### 1. Installation

**⚠️ Python Version Requirement for Full Features**

For **full DeepFace integration**, use **Python 3.12 or earlier**:
```bash
# Recommended: Python 3.12
python3.12 -m venv venv
source venv/bin/activate
```

With **Python 3.13**, the app runs with **basic OpenCV face detection** (DeepFace/TensorFlow not yet compatible).

**✅ Successfully tested with Python 3.12 + DeepFace + TensorFlow 2.19.0**

```bash
# Clone and setup
git clone <repository-url>
cd faaaaaces

# Create virtual environment (Python 3.12 recommended for DeepFace)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Create environment file
cp .env.example .env
# Edit .env with your configuration
```

### 2. Run the Application

```bash
python run.py
```

Visit `http://localhost:5000` to access the web interface.

### 3. Process Your First Video

**Option A: Upload a file**
1. Go to "Upload Video" page
2. Drag and drop video files or click to select
3. Wait for processing to complete

**Option B: Process from URL**
1. Enter a video URL (YouTube, Instagram, TikTok, Vimeo, etc.)
2. Click "Process" 
3. The video will be downloaded and processed automatically

**Supported URL formats:**
- YouTube: `https://youtube.com/watch?v=...` or `https://youtu.be/...`
- Instagram: `https://instagram.com/p/...` or `https://instagram.com/reel/...`
- TikTok: `https://tiktok.com/@user/video/...` or `https://vm.tiktok.com/...`
- Vimeo: `https://vimeo.com/...`

**Option C: Batch processing**
1. Enter multiple URLs (one per line)
2. Click "Process All URLs"
3. Monitor progress on the dashboard

## Architecture

### Core Components

- **Flask Web Framework**: Main application and API endpoints
- **DeepFace**: Face detection, recognition, and embedding generation
- **OpenCV**: Video frame extraction and image processing
- **scikit-learn**: Face clustering algorithms (DBSCAN, Agglomerative)
- **yt-dlp**: Video downloading from web URLs
- **SQLite**: Face metadata and embeddings storage

### Processing Workflow

1. **Video Ingestion** → Upload file or download from URL
2. **Frame Extraction** → Extract frames at configurable intervals
3. **Face Detection** → Find and extract face regions using DeepFace
4. **Embedding Generation** → Create numerical face representations
5. **Face Clustering** → Group similar faces across all videos
6. **Database Storage** → Store face crops, embeddings, and metadata

## Configuration

Key environment variables in `.env`:

```bash
# Processing Settings
FRAME_EXTRACTION_INTERVAL=30  # Extract frame every N seconds
FACE_CONFIDENCE_THRESHOLD=0.7  # Minimum face detection confidence
CLUSTERING_SIMILARITY_THRESHOLD=0.6  # Face similarity threshold

# File Limits
MAX_VIDEO_SIZE_MB=500  # Maximum upload size
MAX_DOWNLOAD_SIZE_MB=1000  # Maximum download size

# Storage Paths
UPLOAD_FOLDER=uploads  # Uploaded videos
FACES_FOLDER=faces     # Extracted face images
TEMP_FOLDER=temp       # Temporary processing files
```

## API Endpoints

### Video Processing
- `POST /api/upload_video` - Upload video file
- `POST /api/process_url` - Process single video URL
- `POST /api/process_urls` - Process multiple video URLs

### Face Management
- `POST /api/cluster_faces` - Trigger face clustering
- `GET /api/clusters` - Get all face clusters
- `GET /api/video_info/<id>` - Get video details and faces

### Job Status
- `GET /api/job_status/<id>` - Get processing job status

## Performance Considerations

### Large-Scale Usage
- **Background Processing**: Consider using Celery + Redis for production
- **Vector Database**: Use Pinecone/Weaviate for efficient similarity search
- **GPU Acceleration**: Enable CUDA for faster DeepFace processing
- **Storage**: Use cloud storage (S3/GCS) for face images in production

### Memory Usage
- Face embeddings are stored in database (JSON format)
- Temporary video files are cleaned up after processing
- Configure frame extraction interval to balance accuracy vs. speed

## Troubleshooting

### Common Issues

**DeepFace Installation**
```bash
# If you get model download errors
pip install deepface --upgrade
python -c "from deepface import DeepFace; DeepFace.build_model('VGG-Face')"
```

**Video Download Issues**
```bash
# Update yt-dlp for latest site support
pip install yt-dlp --upgrade
```

**Platform-Specific Issues**

**Instagram:**
- Only public posts and reels work (private accounts not supported)
- Use full URLs: `instagram.com/p/...` or `instagram.com/reel/...`
- Stories may not be available after 24 hours
- Rate limiting: wait a few minutes between many requests

**TikTok:**
- Only public videos are downloadable
- Both full URLs and short links (`vm.tiktok.com`) work
- Some content may be region-locked
- Private accounts require the content to be public

**YouTube:**
- Private, unlisted, or age-restricted videos cannot be downloaded
- Very long videos may exceed size limits
- Live streams may not be supported

**OpenCV Issues**
```bash
# On macOS/Linux, you might need:
pip install opencv-python-headless
```

### Performance Tuning

**Face Detection Backend**
- `opencv`: Fastest, good for most use cases
- `mtcnn`: More accurate, slower
- `retinaface`: Best accuracy, slowest

**Clustering Algorithm**
- `dbscan`: Good for unknown number of clusters
- `agglomerative`: More deterministic, requires cluster count estimation

## Technical Architecture

### 🏗️ Core Components

**Python Environment Management:**
- **Python 3.12**: Full DeepFace + TensorFlow integration
- **Python 3.13**: Fallback to OpenCV-only mode
- **uv**: Modern Python package and environment manager

**AI/ML Stack:**
- **TensorFlow 2.19.0**: Deep learning framework
- **DeepFace 0.0.93**: Face recognition library  
- **tf-keras**: Compatibility layer for TensorFlow
- **Multiple models**: VGG-Face (580MB), Facenet, OpenFace, ArcFace

**Face Detection Backends:**
- **RetinaFace**: Highest accuracy, production-ready
- **MTCNN**: Multi-task CNN, good balance
- **OpenCV**: Fastest, basic detection
- **SSD**: Single Shot Detector
- **dlib**: Traditional computer vision

### 🗂️ Project Structure
```
faaaaaces/
├── app/
│   ├── models/                  # Database models and schema
│   ├── services/               # Core processing services
│   │   ├── face_extractor.py           # DeepFace integration
│   │   ├── face_extractor_basic.py     # OpenCV fallback
│   │   ├── video_downloader.py         # Multi-platform downloader
│   │   ├── video_processor.py          # Frame extraction
│   │   └── face_clustering.py          # ML clustering
│   ├── routes/                 # Web routes and API endpoints
│   └── templates/              # Responsive HTML templates
├── uploads/                    # User-uploaded videos
├── faces/                      # Extracted face images
├── temp/                       # Temporary video downloads
├── .venv/                      # Python 3.12 virtual environment
├── start_server.sh             # Production server startup
├── stop_server.sh              # Server management
└── requirements.txt            # Secure dependency versions
```

## Development

### Adding New Features

1. **Custom Face Models**: Modify `face_extractor.py` to use different DeepFace models
2. **Additional Video Sources**: Extend `video_downloader.py` for new platforms
3. **Advanced Clustering**: Add new algorithms to `face_clustering.py`
4. **API Extensions**: Add new endpoints in `routes/api.py`

## License

This project is for educational and research purposes. Please ensure compliance with privacy laws and platform terms of service when processing videos.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

---

**Built with ❤️ and lots of faces! 👤👥👫**