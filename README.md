# FAAAAACES - Face Recognition Video Processing Application

> **⚠️ RESEARCH PURPOSES ONLY**
> 
> This application is intended for **academic research, educational purposes, and legitimate security applications only**. Users are responsible for complying with all applicable laws, regulations, and ethical guidelines in their jurisdiction. The developers do not endorse or support any misuse of this technology for unauthorized surveillance, privacy violations, or other harmful activities.

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

## Installation & Deployment

**⚠️ Python Version Requirement**: For **full DeepFace integration**, use **Python 3.12 or earlier**. With **Python 3.13**, the app runs with **basic OpenCV face detection** (DeepFace/TensorFlow not yet compatible).

### 🔧 Local Development Installation (Recommended for most users)

**Use this for:** Development, testing, learning, or running on your personal machine.

```bash
# Clone the repository
git clone <repository-url>
cd faaaaaces

# Quick local installation
./install.sh

# Activate virtual environment
source .venv/bin/activate

# Start development server
python3 run.py

# Open your browser
open http://localhost:5005
```

**What install.sh does:**
- ✅ Sets up Python virtual environment
- ✅ Installs all dependencies (core + optional GPU/mask reconstruction)
- ✅ Initializes SQLite database
- ✅ Creates necessary directories
- ✅ Quick verification (no hanging server tests)
- ✅ **Does NOT download AI models** (downloaded on first use)

### 🚀 Production Deployment

**Use this for:** Production servers, staging environments, or full deployments with system services.

```bash
# Full production deployment
./deploy.sh

# Start the server
./deploy/faaaaaces start

# Setup reverse proxy (nginx/Apache)
./deploy/faaaaaces proxy
```

**What deploy.sh includes (beyond install.sh):**
- ✅ Everything from local installation
- ✅ System service setup (systemd/launchd)
- ✅ Performance optimization & model pre-downloading
- ✅ Full test suite (including server tests)
- ✅ Production environment validation
- ✅ Deployment reporting
- ✅ Binary management system

### 📊 Installation Comparison

| Feature | install.sh (Local) | deploy.sh (Production) |
|---------|-------------------|----------------------|
| **Virtual Environment** | ✅ Yes | ✅ Yes |
| **Core Dependencies** | ✅ Yes | ✅ Yes |
| **Database Setup** | ✅ Yes | ✅ Yes |
| **Directory Structure** | ✅ Yes | ✅ Yes |
| **Quick Verification** | ✅ Simple imports | ✅ Full test suite |
| **Server Tests** | ❌ No (avoids port issues) | ✅ Yes (port 5004) |
| **System Services** | ❌ No | ✅ systemd/launchd |
| **Model Pre-download** | ❌ No | ✅ YOLO models |
| **Performance Optimization** | ❌ No | ✅ Yes |
| **Deployment Binary** | ❌ No | ✅ faaaaaces binary |
| **Installation Time** | ⚡ 2-5 minutes | 🕐 10-15 minutes |

### 🐳 Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f faaaaaces
```

### 🔄 Manual Setup (Advanced)

```bash
# Create virtual environment (Python 3.12 recommended)
python3.12 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-gpu-detection.txt      # Optional: GPU support
pip install -r requirements-mask-reconstruction.txt # Optional: Advanced AI

# Initialize database
python3 -c "from app.models.database import init_db; init_db()"

# Start development server
python3 run_simple.py
```

### 💡 Which Installation Should You Use?

**Choose `install.sh` if you:**
- Want to quickly try out FAAAAACES
- Are developing or testing locally
- Had issues with the old deploy script getting stuck
- Don't need system services or production features

**Choose `deploy.sh` if you:**
- Are deploying to a production server
- Need system service management
- Want full performance optimization
- Require the deployment binary system
- Need nginx/Apache integration

### 🆘 Troubleshooting Previous Installation Issues

If you had problems with the old deploy script getting stuck on port 5004:

```bash
# Kill any stuck processes
pkill -f "python.*5004"
pkill -f "python.*5005"

# Clean and start fresh
./install.sh clean  # or ./deploy.sh clean
./install.sh        # Use the new local installer
```

The application will be available at `http://localhost:5005` (or your configured domain with reverse proxy).

## Process Your First Video

**🎭 Mask Detection Workflow** (Advanced Features)
1. Visit the **Mask Detection Workflow** page
2. **Step 1**: Upload video file, enter URL, or select existing video
3. **Step 2**: Choose detection mode (full video or targeted timestamps)
4. **Step 3**: Review detected faces and select masked faces
5. **Step 4**: Reconstruct faces using AI models
6. **Step 5**: View before/after results and download

**📤 Basic Face Recognition**
1. Go to "Upload Video" page or use URL processing
2. Drag and drop video files or enter video URL
3. Wait for processing to complete
4. View results in the dashboard

**🔗 Supported URL formats:**
- YouTube: `https://youtube.com/watch?v=...` or `https://youtu.be/...`
- Instagram: `https://instagram.com/p/...` or `https://instagram.com/reel/...`
- TikTok: `https://tiktok.com/@user/video/...` or `https://vm.tiktok.com/...`
- Vimeo: `https://vimeo.com/...`

**📊 Monitor Progress**
- Real-time updates on the unified dashboard
- Processing status and face detection results
- Face clustering and similarity analysis

## Deployment Management (Production Only)

**Note:** These commands are only available after running `./deploy.sh`. For local development, use `python3 run.py` directly.

### Binary Commands

The `deploy/faaaaaces` binary provides comprehensive deployment management:

```bash
# Environment setup
./deploy/faaaaaces setup          # Setup virtual environment and dependencies

# Server management
./deploy/faaaaaces start          # Start the server
./deploy/faaaaaces stop           # Stop the server
./deploy/faaaaaces restart        # Restart the server
./deploy/faaaaaces status         # Check server status

# Maintenance
./deploy/faaaaaces test           # Run test suite
./deploy/faaaaaces health         # Perform health check
./deploy/faaaaaces proxy          # Setup reverse proxy (nginx/Apache)
```

### Command Options

```bash
# Custom port
./deploy/faaaaaces start --port 8080

# Debug mode
./deploy/faaaaaces start --debug

# Run in foreground
./deploy/faaaaaces start --foreground
```

### Deployment Scripts

```bash
# Local development installation (recommended)
./install.sh                      # Quick local setup

# Production deployment
./deploy.sh                       # Full automated deployment

# Reverse proxy setup (production)
./deploy/setup-reverse-proxy.sh   # Interactive nginx/Apache setup
```

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
- `GET /api/videos_list` - List all videos

### Face Management
- `POST /api/cluster_faces` - Trigger face clustering
- `GET /api/clusters` - Get all face clusters
- `GET /api/video_info/<id>` - Get video details and faces

### Mask Detection (Advanced)
- `POST /api/enhanced/enhanced_detect_masks/<video_id>` - Detect masks in video
- `POST /api/enhanced/enhanced_extract_timestamps/<video_id>` - Extract specific timestamps
- `GET /api/masked/all_masked_faces` - Get all detected masked faces
- `POST /api/masked/batch_reconstruct` - Reconstruct multiple faces
- `POST /api/masked/reconstruct/<face_id>` - Reconstruct single face

### Legacy Mask Detection
- `POST /api/mask/detect_masks/<video_id>` - Basic mask detection
- `POST /api/mask/extract_timestamps/<video_id>` - Extract targeted frames

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

### Installation Issues

**Getting stuck on port 5004 during installation?**
This was a known issue with the old deploy.sh script. Use the new `install.sh` for local development:
```bash
# Kill any stuck processes
pkill -f "python.*5004"
pkill -f "python.*5005"

# Use the new local installer
./install.sh
```

**Models not downloading?**
Models are downloaded on first use, not during installation:
- DeepFace models: Downloaded to `~/.deepface/weights/` when first processing a video
- YOLO models: Only pre-downloaded in production deployment (`deploy.sh`)

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
├── app/                        # Main application
│   ├── models/                 # Database models and schema
│   ├── services/              # Core processing services
│   │   ├── face_extractor.py          # DeepFace integration
│   │   ├── face_extractor_basic.py    # OpenCV fallback
│   │   ├── video_downloader.py        # Multi-platform downloader
│   │   ├── video_processor.py         # Frame extraction
│   │   ├── face_clustering.py         # ML clustering
│   │   ├── unified_face_mask_detector.py  # Advanced mask detection
│   │   ├── face_reconstructor.py      # AI face reconstruction
│   │   ├── mask_detector.py           # Core mask detection
│   │   └── enhanced_video_processor.py # Optimized processing
│   ├── routes/                # Web routes and API endpoints
│   │   ├── api.py                     # Core API endpoints
│   │   ├── main.py                    # Main web routes
│   │   ├── enhanced_mask_api.py       # Advanced mask detection API
│   │   ├── mask_api.py                # Basic mask detection API
│   │   └── masked_faces_api.py        # Face reconstruction API
│   ├── templates/             # Unified HTML templates
│   │   ├── unified_dashboard.html     # Main dashboard
│   │   ├── unified_mask_operations.html # Mask workflow
│   │   └── masked_faces_gallery.html  # Face gallery
│   └── static/js/             # JavaScript
│       └── unified-mask-workflow.js   # Workflow management
├── deploy/                     # Deployment system
│   ├── faaaaaces              # Main deployment binary
│   ├── setup-reverse-proxy.sh # Reverse proxy setup
│   ├── nginx.conf             # nginx configuration
│   └── apache.conf            # Apache configuration
├── tests/                      # Test suite
├── scripts/                    # Database migrations
├── uploads/                    # User-uploaded videos
├── faces/                      # Extracted face images
├── temp/                       # Temporary video downloads
├── .venv/                      # Python virtual environment
├── deploy.sh                   # Master deployment script
├── run_simple.py               # Application entry point
├── Dockerfile                  # Docker containerization
├── docker-compose.yml          # Docker orchestration
├── requirements.txt            # Core dependencies
├── requirements-gpu-detection.txt     # GPU enhancements
└── requirements-mask-reconstruction.txt # Advanced AI models
```

## Development

### Adding New Features

1. **Custom Face Models**: Modify `face_extractor.py` to use different DeepFace models
2. **Additional Video Sources**: Extend `video_downloader.py` for new platforms
3. **Advanced Clustering**: Add new algorithms to `face_clustering.py`
4. **API Extensions**: Add new endpoints in `routes/` directory
5. **Mask Detection Models**: Extend `unified_face_mask_detector.py` for new detection methods
6. **Face Reconstruction**: Add new AI models to `face_reconstructor.py`

### Testing

```bash
# Local development testing
python -m pytest tests/test_server.py
python -m pytest tests/test_face_reconstruction.py

# Production testing (requires deploy.sh)
./deploy/faaaaaces test
./deploy/faaaaaces health
```

### Development Workflow

```bash
# Local development (recommended)
./install.sh                      # One-time setup
source .venv/bin/activate         # Activate environment
python3 run.py                    # Start dev server

# Production deployment
./deploy.sh                       # Full deployment
./deploy/faaaaaces start          # Start production server
./deploy/faaaaaces proxy          # Setup reverse proxy
```

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