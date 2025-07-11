# Mask Detection and Face Reconstruction Guide

## Overview

FAAAAACES now supports advanced mask detection and face reconstruction capabilities. This allows you to:

1. **Detect masked faces** in videos automatically
2. **Reconstruct underlying faces** from masked faces using AI
3. **Identify people** who are wearing masks
4. **Analyze mask usage** across your video library

## Features

### 🔍 Mask Detection
- **Computer Vision fallback** - Works out of the box with OpenCV
- **Machine Learning detection** - Higher accuracy with trained models
- **Mask type classification** - Surgical, cloth, N95, etc.
- **Confidence scoring** - Reliability metrics for each detection

### 🔧 Face Reconstruction  
- **GFPGAN integration** - State-of-the-art face restoration
- **CodeFormer support** - Robust face reconstruction
- **Classical inpainting** - OpenCV fallback for basic reconstruction
- **Quality assessment** - Automatic quality scoring of reconstructions

### 📊 Analytics
- **Mask statistics** - Percentage of masked vs unmasked faces
- **Reconstruction success rates** - Track reconstruction quality
- **Mask type distribution** - Analyze types of masks detected
- **Video-level analysis** - Per-video mask detection results

## Installation

### Basic Installation (Computer Vision)
The basic mask detection works out of the box with existing dependencies.

### Advanced Installation (AI Models)
For better accuracy and face reconstruction:

```bash
# Install mask detection and reconstruction dependencies
pip install -r requirements-mask-reconstruction.txt

# For GPU acceleration (recommended)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Model Downloads
Models are downloaded automatically on first use:
- **GFPGAN**: ~280MB - Face restoration and enhancement
- **VGG-Face**: ~580MB - Already installed for face recognition
- **Mask Detection**: ~50MB - Optional, uses CV fallback if not available

## API Endpoints

### Get Mask Statistics
```bash
GET /api/masks/mask_statistics
```
Returns overall mask detection statistics across all videos.

### Get Masked Faces
```bash
GET /api/masks/masked_faces?video_id=1
```
Returns all faces detected as wearing masks, optionally filtered by video.

### Get Reconstructed Faces  
```bash
GET /api/masks/reconstructed_faces?video_id=1
```
Returns all faces that have been successfully reconstructed.

### Detect Masks in Video
```bash
POST /api/masks/detect_masks/1
```
Analyzes all faces in video ID 1 to detect masks.

### Reconstruct Faces in Video
```bash
POST /api/masks/reconstruct_faces/1
```
Reconstructs all masked faces found in video ID 1.

### Service Health Check
```bash
GET /api/masks/mask_detection_health
```
Returns status of mask detection and reconstruction services.

## Database Schema

New fields added to the `faces` table:

### Mask Detection Fields
- `is_masked` (BOOLEAN) - Whether face is wearing a mask
- `mask_confidence` (REAL) - Detection confidence (0.0-1.0)
- `mask_type` (TEXT) - Type of mask (surgical, cloth, N95, etc.)
- `mask_detection_method` (TEXT) - Detection method used

### Reconstruction Fields  
- `has_reconstruction` (BOOLEAN) - Whether face has been reconstructed
- `reconstructed_image_path` (TEXT) - Path to reconstructed face image
- `reconstruction_quality` (REAL) - Quality score (0.0-1.0)
- `reconstruction_method` (TEXT) - Reconstruction method used

## Usage Examples

### Basic Mask Detection
```python
from app.services.mask_detector import MaskDetector
import cv2

detector = MaskDetector()
face_image = cv2.imread('path/to/face.jpg')
result = detector.detect_mask(face_image)

print(f"Is masked: {result['is_masked']}")
print(f"Confidence: {result['confidence']:.2f}")
print(f"Mask type: {result['mask_type']}")
```

### Face Reconstruction
```python
from app.services.face_reconstructor import FaceReconstructor
import cv2

reconstructor = FaceReconstructor()
masked_face = cv2.imread('path/to/masked_face.jpg')
result = reconstructor.reconstruct_masked_face(masked_face)

if result['success']:
    cv2.imwrite('reconstructed_face.jpg', result['reconstructed_image'])
    print(f"Quality score: {result['quality_score']:.2f}")
```

### Integrated Processing
```python
from app.services.face_extractor import FaceExtractor

# Enable mask detection and reconstruction
extractor = FaceExtractor(
    enable_mask_detection=True,
    enable_face_reconstruction=True
)

# Process video frames
face_data = extractor.process_video_frames(frame_paths, output_dir, video_id)

# Process masked faces
mask_results = extractor.process_masked_faces(face_data)
print(f"Found {mask_results['masked_count']} masked faces")
print(f"Reconstructed {mask_results['reconstructed_count']} faces")
```

## Performance Considerations

### Processing Speed
- **Mask Detection**: 50-100ms per face (CV), 200-500ms per face (ML)
- **Face Reconstruction**: 1-5 seconds per face (GFPGAN), 10+ seconds (CodeFormer)
- **Memory Usage**: 2-4GB additional RAM for AI models

### Quality Factors
- **Image resolution** - Higher resolution = better reconstruction
- **Mask type** - Surgical masks easier to reconstruct than N95
- **Face angle** - Frontal faces reconstruct better than profile
- **Lighting conditions** - Good lighting improves results

### Batch Processing
For large video libraries:
1. **Process overnight** - Reconstruction is computationally intensive
2. **Use GPU acceleration** - 5-10x speedup with CUDA
3. **Prioritize videos** - Process most important content first
4. **Monitor quality** - Review reconstruction results for accuracy

## Ethical Considerations

### Privacy and Consent
- Ensure proper authorization for unmasking individuals
- Consider data protection regulations (GDPR, CCPA, etc.)
- Implement access controls for sensitive reconstructed data
- Document usage for compliance auditing

### Accuracy and Limitations
- AI reconstruction is not 100% accurate
- False positives/negatives in mask detection
- Reconstructed faces may not match actual appearance
- Quality varies based on input conditions

### Responsible Usage
- Use for legitimate security/safety purposes
- Avoid creating misleading reconstructions
- Respect individual privacy rights
- Consider bias in AI models

## Troubleshooting

### Common Issues

**Mask detection not working:**
- Check image quality and resolution
- Verify face is clearly visible
- Try different detection methods

**Reconstruction failing:**
- Ensure sufficient GPU memory
- Check model installation
- Verify input image format

**Poor reconstruction quality:**
- Use higher resolution input images  
- Try different reconstruction models
- Adjust reconstruction parameters

### Performance Issues

**Slow processing:**
- Enable GPU acceleration
- Reduce batch size
- Use faster models (classical inpainting)

**High memory usage:**
- Process videos individually
- Clear cache between processing
- Use CPU-only mode if needed

## Future Enhancements

### Planned Features
- **Real-time mask detection** in video streams
- **Custom mask detection models** for specific use cases
- **Advanced reconstruction** with StyleGAN/diffusion models
- **Face matching** with reconstructed faces
- **Batch processing UI** for large video libraries

### Model Improvements
- **Higher accuracy** mask detection models
- **Faster reconstruction** algorithms
- **Better quality assessment** metrics
- **Multi-modal** face reconstruction (using multiple angles)

---

*For technical support or questions about mask detection capabilities, please file an issue on the GitHub repository.*