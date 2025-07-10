# FAAAAACES - Face Recognition Video Processing Application

A Python web application that processes videos to extract and catalog faces using DeepFace and machine learning clustering.

## Features

### 🎥 Video Input Methods
- **File Upload**: Direct upload of video files (MP4, AVI, MOV, etc.)
- **URL Processing**: Download and process videos from URLs (YouTube, Vimeo, direct links)
- **Batch Processing**: Process multiple URLs simultaneously

### 🔍 Face Processing Pipeline
- **Face Detection**: Extract faces from video frames using DeepFace
- **Face Recognition**: Generate embeddings for face similarity comparison
- **Smart Clustering**: Group similar faces across all videos using machine learning
- **Known Faces Database**: Build a searchable catalog of unique individuals

### 🌐 Web Interface
- **Dashboard**: Overview of processed videos and face clusters
- **Upload Interface**: Drag-and-drop file upload and URL input
- **Face Gallery**: Browse discovered faces organized by clusters
- **Real-time Processing**: Live updates on video processing status

## Quick Start

### 1. Installation

```bash
# Clone and setup
git clone <repository-url>
cd faaaaaces

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
1. Enter a video URL (YouTube, Vimeo, etc.)
2. Click "Process" 
3. The video will be downloaded and processed automatically

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

## Development

### Project Structure
```
faaaaaces/
├── app/
│   ├── models/          # Database models
│   ├── services/        # Core processing services
│   ├── routes/          # Web routes and API
│   └── templates/       # HTML templates
├── uploads/             # Uploaded videos
├── faces/               # Extracted face images
├── temp/                # Temporary files
└── run.py              # Application entry point
```

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