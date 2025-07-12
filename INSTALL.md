# FAAAAACES Installation Guide

FAAAAACES provides two installation methods depending on your use case:

## 🔧 Local Development Installation

**Use this for:** Development, testing, learning, or running on your personal machine.

```bash
./install.sh
```

### What it does:
- ✅ Sets up Python virtual environment
- ✅ Installs core dependencies
- ✅ Initializes database
- ✅ Creates basic directory structure
- ✅ Simple verification (no server tests)
- ✅ Quick and reliable setup

### What it doesn't include:
- ❌ System service setup
- ❌ Performance optimization
- ❌ Complex server tests that can hang
- ❌ Production-specific configurations

### After installation:
```bash
# Activate virtual environment
source .venv/bin/activate

# Start development server
python3 run.py

# Open browser
open http://localhost:5005
```

---

## 🚀 Production Deployment

**Use this for:** Production servers, staging environments, or full deployments.

```bash
./deploy.sh
```

### What it includes:
- ✅ Everything from local installation
- ✅ System service setup (systemd/launchd)
- ✅ Performance optimization
- ✅ Full test suite
- ✅ Production environment validation
- ✅ Deployment reporting
- ✅ Service management

### After deployment:
- **Linux:** `systemctl start faaaaaces`
- **macOS:** `launchctl load ~/Library/LaunchAgents/com.faaaaaces.server.plist`
- **Manual:** `./deploy/faaaaaces start`

---

## 🆘 Troubleshooting

### Previous Installation Issues

If you previously had problems with the old deploy script getting stuck on port 5004:

1. **Kill any running servers:**
   ```bash
   pkill -f "python.*5004"
   pkill -f "python.*5005"
   ```

2. **Clean previous installation:**
   ```bash
   ./install.sh clean
   # or
   ./deploy.sh clean
   ```

3. **Start fresh with the appropriate script:**
   ```bash
   ./install.sh    # For local development
   ./deploy.sh     # For production only
   ```

### Common Issues

- **Python version errors:** Ensure Python 3.8+ is installed
- **Permission errors:** Don't run as root unless deploying to production
- **FFmpeg missing:** Install with `brew install ffmpeg` (macOS) or `apt-get install ffmpeg` (Ubuntu)
- **Port conflicts:** Make sure ports 5005 is available

### Getting Help

```bash
./install.sh help    # Local development help
./deploy.sh help     # Production deployment help
```

---

## 📋 Quick Reference

| Task | Command |
|------|---------|
| **Local Development** | `./install.sh` |
| **Production Deployment** | `./deploy.sh` |
| **Start Development Server** | `python3 run.py` |
| **Clean Installation** | `./install.sh clean` |
| **Get Help** | `./install.sh help` |

Choose the right script for your needs:
- 🔧 **install.sh** = Simple, fast, local development
- 🚀 **deploy.sh** = Complete, production-ready deployment