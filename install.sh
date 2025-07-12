#!/bin/bash

# FAAAAACES Local Development Installation Script
# Simple, focused installation for local development

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="FAAAAACES"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$PROJECT_ROOT/.venv"
PYTHON_CMD="$VENV_PATH/bin/python3"
PIP_CMD="$VENV_PATH/bin/pip"

# Print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_header() {
    echo -e "${PURPLE}🎭 $1${NC}"
}

# Check basic system requirements
check_system_requirements() {
    print_header "Checking Basic System Requirements"
    
    # Check Python version
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
        
        if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 8 ]; then
            print_status "Python $PYTHON_VERSION found"
        else
            print_error "Python 3.8+ required (found $PYTHON_VERSION)"
            exit 1
        fi
    else
        print_error "Python 3 not found. Please install Python 3.8+"
        exit 1
    fi
    
    # Check for ffmpeg (optional but recommended)
    if command -v ffmpeg &> /dev/null; then
        print_status "FFmpeg found"
    else
        print_warning "FFmpeg not found (recommended for video processing)"
        print_info "Install with: brew install ffmpeg (macOS) or apt-get install ffmpeg (Ubuntu)"
        read -p "Continue without FFmpeg? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_info "Please install FFmpeg and run this script again"
            exit 1
        fi
    fi
}

# Setup virtual environment
setup_virtual_environment() {
    print_header "Setting Up Virtual Environment"
    
    if [ -d "$VENV_PATH" ]; then
        print_info "Virtual environment already exists"
        read -p "Recreate virtual environment? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_info "Removing existing virtual environment..."
            rm -rf "$VENV_PATH"
        else
            print_info "Using existing virtual environment"
            return 0
        fi
    fi
    
    print_info "Creating virtual environment..."
    python3 -m venv "$VENV_PATH"
    
    print_info "Upgrading pip..."
    "$PIP_CMD" install --upgrade pip
    
    print_status "Virtual environment created successfully"
}

# Install Python dependencies
install_dependencies() {
    print_header "Installing Python Dependencies"
    
    # Core requirements (essential)
    if [ -f "$PROJECT_ROOT/requirements.txt" ]; then
        print_info "Installing core requirements..."
        "$PIP_CMD" install -r "$PROJECT_ROOT/requirements.txt"
        print_status "Core requirements installed"
    else
        print_error "requirements.txt not found"
        exit 1
    fi
    
    # Optional GPU detection requirements
    if [ -f "$PROJECT_ROOT/requirements-gpu-detection.txt" ]; then
        print_info "Installing GPU detection requirements (optional)..."
        "$PIP_CMD" install -r "$PROJECT_ROOT/requirements-gpu-detection.txt" || {
            print_warning "GPU detection requirements failed (continuing without GPU support)"
        }
    fi
    
    # Optional mask reconstruction requirements
    if [ -f "$PROJECT_ROOT/requirements-mask-reconstruction.txt" ]; then
        print_info "Installing mask reconstruction requirements (optional)..."
        "$PIP_CMD" install -r "$PROJECT_ROOT/requirements-mask-reconstruction.txt" || {
            print_warning "Mask reconstruction requirements failed (continuing with basic reconstruction)"
        }
    fi
    
    print_status "Dependencies installation completed"
}

# Initialize database
setup_database() {
    print_header "Setting Up Database"
    
    print_info "Initializing database schema..."
    cd "$PROJECT_ROOT"
    "$PYTHON_CMD" -c "from app.models.database import init_db; init_db()" || {
        print_error "Database initialization failed"
        exit 1
    }
    
    print_status "Database initialized successfully"
}

# Create basic directory structure
create_directories() {
    print_header "Creating Directory Structure"
    
    # Create essential directories
    mkdir -p "$PROJECT_ROOT/logs"
    mkdir -p "$PROJECT_ROOT/uploads"
    mkdir -p "$PROJECT_ROOT/faces"
    mkdir -p "$PROJECT_ROOT/faces/reconstructed"
    mkdir -p "$PROJECT_ROOT/temp"
    
    print_status "Directory structure created"
}

# Simple health check
verify_installation() {
    print_header "Verifying Installation"
    
    print_info "Testing basic imports..."
    cd "$PROJECT_ROOT"
    
    # Test basic imports without starting servers
    "$PYTHON_CMD" -c "
import app
from app.models.database import DatabaseManager
from app.services.face_detector import FaceDetector
print('✅ Core imports successful')
" || {
        print_error "Basic import test failed"
        exit 1
    }
    
    print_info "Testing database connection..."
    "$PYTHON_CMD" -c "
from app.models.database import DatabaseManager
db = DatabaseManager()
print('✅ Database connection successful')
" || {
        print_error "Database connection test failed"
        exit 1
    }
    
    print_status "Installation verification completed"
}

# Main installation function
install() {
    print_header "Installing FAAAAACES for Local Development"
    
    echo -e "${BLUE}This script will install FAAAAACES for local development.${NC}"
    echo -e "${BLUE}For production deployment, use deploy.sh instead.${NC}"
    echo ""
    
    # Run installation steps
    check_system_requirements
    setup_virtual_environment
    install_dependencies
    setup_database
    create_directories
    verify_installation
    
    print_header "Installation Complete!"
    print_status "FAAAAACES has been successfully installed for local development"
    print_info ""
    print_info "Quick Start:"
    print_info "  1. Activate virtual environment: source .venv/bin/activate"
    print_info "  2. Start development server: python3 run.py"
    print_info "  3. Open browser: http://localhost:5005"
    print_info "  4. Upload videos and detect faces!"
    print_info ""
    print_info "Development Commands:"
    print_info "  • Start server: python3 run.py"
    print_info "  • Run tests: python3 -m pytest tests/"
    print_info "  • Check status: curl http://localhost:5005/health"
    print_info ""
    print_warning "This is a development installation. For production deployment,"
    print_warning "please use the deploy.sh script instead."
}

# Command line interface
case "${1:-install}" in
    "install")
        install
        ;;
    "clean")
        print_header "Cleaning installation"
        rm -rf "$VENV_PATH"
        rm -rf "$PROJECT_ROOT/logs"
        rm -rf "$PROJECT_ROOT/temp"
        print_status "Installation cleaned (uploads and faces preserved)"
        ;;
    "help")
        echo "FAAAAACES Local Development Installation Script"
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  install   - Install for local development (default)"
        echo "  clean     - Clean installation files"
        echo "  help      - Show this help"
        echo ""
        echo "For production deployment, use deploy.sh instead."
        ;;
    *)
        echo "Unknown command: $1"
        echo "Run '$0 help' for usage information"
        exit 1
        ;;
esac