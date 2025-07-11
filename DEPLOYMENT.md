# FAAAAACES Deployment Guide

Complete deployment guide for the Face Recognition & Mask Detection Platform.

## 🚀 Quick Start

### Option 1: Automated Deployment (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd faaaaaces

# Run automated deployment
./deploy.sh

# Start the server
./deploy/faaaaaces start

# Open browser to http://localhost:5005
```

### Option 2: Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f faaaaaces

# Stop
docker-compose down
```

### Option 3: Manual Deployment

```bash
# Set up virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-gpu-detection.txt    # Optional
pip install -r requirements-mask-reconstruction.txt  # Optional

# Initialize database
python3 -c "from app.models.database import init_db; init_db()"

# Start server
python3 run_simple.py
```

## 📋 System Requirements

### Minimum Requirements
- **OS**: macOS 10.14+, Ubuntu 18.04+, Windows 10+
- **Python**: 3.8+
- **RAM**: 4GB
- **Storage**: 2GB free space
- **Network**: Internet connection for model downloads

### Recommended Requirements
- **OS**: macOS 12+, Ubuntu 20.04+
- **Python**: 3.10+
- **RAM**: 8GB+
- **Storage**: 5GB+ free space
- **GPU**: NVIDIA GPU with CUDA (optional, for enhanced performance)

### Required System Packages
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get update
sudo apt-get install ffmpeg python3-opencv build-essential

# Windows
# Install ffmpeg from https://ffmpeg.org/download.html
```

## 🔧 Deployment Options

### 1. Local Development
```bash
./deploy/faaaaaces start --debug --foreground
```

### 2. Production Server
```bash
./deploy/faaaaaces start
```

### 3. System Service

#### Linux (systemd)
```bash
# Deploy as root to create system service
sudo ./deploy.sh

# Control service
sudo systemctl start faaaaaces
sudo systemctl stop faaaaaces
sudo systemctl status faaaaaces
```

#### macOS (launchd)
```bash
# Deploy normally
./deploy.sh

# Load service
launchctl load ~/Library/LaunchAgents/com.faaaaaces.server.plist

# Unload service
launchctl unload ~/Library/LaunchAgents/com.faaaaaces.server.plist
```

### 4. Docker Production

```bash
# Build image
docker build -t faaaaaces:latest .

# Run container
docker run -d \
  --name faaaaaces \
  -p 5005:5005 \
  -v $(pwd)/faces:/app/faces \
  -v $(pwd)/uploads:/app/uploads \
  -v $(pwd)/faaaaaces.db:/app/faaaaaces.db \
  faaaaaces:latest

# With GPU support
docker run -d \
  --name faaaaaces \
  --gpus all \
  -p 5005:5005 \
  -v $(pwd)/faces:/app/faces \
  -v $(pwd)/uploads:/app/uploads \
  -v $(pwd)/faaaaaces.db:/app/faaaaaces.db \
  faaaaaces:latest
```

## 🎛️ Configuration

### Environment Variables
```bash
# Server configuration
export FLASK_ENV=production        # or development
export FLASK_DEBUG=False          # or True
export FLASK_HOST=0.0.0.0
export FLASK_PORT=5005

# Database configuration
export DATABASE_URL=sqlite:///faaaaaces.db

# Feature flags
export ENABLE_GPU_DETECTION=True
export ENABLE_MASK_RECONSTRUCTION=True
export ENABLE_ADVANCED_MODELS=True
```

### Configuration Files

#### `config.py` (optional)
```python
# Advanced configuration
class Config:
    SECRET_KEY = 'your-secret-key-here'
    DATABASE_URL = 'sqlite:///faaaaaces.db'
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB
    
    # Face detection
    FACE_DETECTION_THRESHOLD = 0.7
    FACE_CLUSTERING_THRESHOLD = 0.6
    
    # Mask detection
    MASK_DETECTION_THRESHOLD = 0.8
    ENABLE_ENHANCED_DETECTION = True
    
    # Performance
    MAX_WORKERS = 4
    ENABLE_CACHING = True
```

## 🔍 Management Commands

### Using the Binary
```bash
# Server management
./deploy/faaaaaces start          # Start server
./deploy/faaaaaces stop           # Stop server
./deploy/faaaaaces restart        # Restart server
./deploy/faaaaaces status         # Check status

# Maintenance
./deploy/faaaaaces test           # Run tests
./deploy/faaaaaces health         # Health check
./deploy/faaaaaces setup          # Setup environment
./deploy/faaaaaces proxy          # Setup reverse proxy
```

### Using Shell Scripts (Legacy)
```bash
# Start server
./start_server.sh

# Stop server
./stop_server.sh
```

## 📊 Monitoring

### Health Checks
```bash
# Built-in health check
curl http://localhost:5005/status

# Binary health check
./deploy/faaaaaces health

# Docker health check
docker inspect faaaaaces-server | grep Health
```

### Log Files
```bash
# Server logs
tail -f server.log

# Deployment logs
tail -f deployment.log

# Docker logs
docker-compose logs -f faaaaaces
```

### Performance Monitoring
```bash
# System resources
htop

# Disk usage
df -h

# Database size
ls -lh faaaaaces.db

# Face storage
du -sh faces/
```

## 🐛 Troubleshooting

### Common Issues

#### 1. Python Version Error
```bash
# Check Python version
python3 --version

# Update Python (macOS)
brew upgrade python

# Update Python (Ubuntu)
sudo apt-get update && sudo apt-get upgrade python3
```

#### 2. OpenCV Import Error
```bash
# Reinstall OpenCV
pip uninstall opencv-python opencv-contrib-python
pip install opencv-python opencv-contrib-python

# System OpenCV (Ubuntu)
sudo apt-get install python3-opencv
```

#### 3. FFmpeg Not Found
```bash
# Install FFmpeg
# macOS: brew install ffmpeg
# Ubuntu: sudo apt-get install ffmpeg
# Windows: Download from https://ffmpeg.org/
```

#### 4. Database Errors
```bash
# Reset database
rm faaaaaces.db
python3 -c "from app.models.database import init_db; init_db()"

# Check database
sqlite3 faaaaaces.db ".tables"
```

#### 5. Port Already in Use
```bash
# Find process using port
lsof -i :5005

# Kill process
kill -9 <PID>

# Use different port
./deploy/faaaaaces start --port 5006
```

#### 6. Permission Errors
```bash
# Fix permissions
chmod +x deploy.sh
chmod +x deploy/faaaaaces
chmod +x start_server.sh
chmod +x stop_server.sh

# Fix directory permissions
chmod -R 755 faces/ uploads/ temp/
```

### Performance Issues

#### 1. Slow Video Processing
- Enable GPU acceleration
- Reduce video resolution
- Use targeted timestamps instead of full video analysis

#### 2. High Memory Usage
- Reduce batch size
- Enable memory caching
- Monitor with `htop` or `top`

#### 3. Database Performance
- Regular database maintenance
- Consider PostgreSQL for large datasets
- Enable database indexing

## 🔧 Advanced Configuration

### GPU Support
```bash
# Install CUDA (NVIDIA)
# Follow NVIDIA CUDA installation guide

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Enable GPU in configuration
export ENABLE_GPU_DETECTION=True
```

### Reverse Proxy Setup

#### Automated Setup (Recommended)
```bash
# Interactive reverse proxy setup
./deploy/faaaaaces proxy

# Or directly run the setup script
./deploy/setup-reverse-proxy.sh
```

#### Manual nginx Setup
```bash
# Install nginx
sudo apt-get install nginx  # Ubuntu/Debian
brew install nginx          # macOS

# Copy configuration
sudo cp deploy/nginx.conf /etc/nginx/sites-available/faaaaaces
sudo ln -s /etc/nginx/sites-available/faaaaaces /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

#### Manual Apache Setup
```bash
# Install Apache
sudo apt-get install apache2  # Ubuntu/Debian
brew install httpd           # macOS

# Copy configuration
sudo cp deploy/apache.conf /etc/apache2/sites-available/faaaaaces.conf
sudo a2ensite faaaaaces.conf
sudo systemctl reload apache2
```
```

### SSL/TLS Configuration
```bash
# Generate self-signed certificate
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

# Update nginx configuration for HTTPS
# Add SSL certificate paths
```

## 📈 Scaling

### Horizontal Scaling
```bash
# Multiple instances with load balancer
# Instance 1
./deploy/faaaaaces start --port 5005

# Instance 2
./deploy/faaaaaces start --port 5006

# Instance 3
./deploy/faaaaaces start --port 5007
```

### Database Scaling
```bash
# PostgreSQL setup
pip install psycopg2-binary
export DATABASE_URL=postgresql://user:password@localhost:5432/faaaaaces

# Redis for caching
pip install redis
export REDIS_URL=redis://localhost:6379
```

## 🔄 Updates

### Application Updates
```bash
# Pull latest code
git pull origin main

# Update dependencies
pip install -r requirements.txt --upgrade

# Restart server
./deploy/faaaaaces restart
```

### Database Migrations
```bash
# Run migration scripts
python3 scripts/upgrade_database_for_masks.py

# Backup database before migrations
cp faaaaaces.db faaaaaces.db.backup
```

## 📋 Maintenance

### Regular Tasks
```bash
# Clean temporary files
find temp/ -type f -mtime +7 -delete

# Backup database
cp faaaaaces.db backups/faaaaaces-$(date +%Y%m%d).db

# Check disk usage
df -h

# Update system packages
sudo apt-get update && sudo apt-get upgrade  # Ubuntu
brew update && brew upgrade                   # macOS
```

### Security Updates
```bash
# Update Python packages
pip list --outdated
pip install --upgrade package_name

# Check for security vulnerabilities
pip audit

# Update system packages regularly
```

## 🆘 Support

### Getting Help
1. Check the troubleshooting section above
2. Review log files for error messages
3. Run health checks and tests
4. Check system requirements
5. Search for similar issues online

### Reporting Issues
When reporting issues, include:
- Operating system and version
- Python version
- Error messages from logs
- Steps to reproduce
- Output of `./deploy/faaaaaces health`

## 📚 Additional Resources

- [README.md](README.md) - Project overview
- [MASK_DETECTION_GUIDE.md](MASK_DETECTION_GUIDE.md) - Mask detection usage
- [HARDWARE_REQUIREMENTS.md](HARDWARE_REQUIREMENTS.md) - Hardware specifications
- [requirements.txt](requirements.txt) - Python dependencies