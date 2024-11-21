# Advanced Image Signal Processing (ISP) Pipeline

## Overview
This project implements an advanced Image Signal Processing (ISP) pipeline with comprehensive image enhancement techniques, focusing on denoising, sharpening, and quality assessment for RAW images.

## Assignment Details
### Assignment 2 Objectives
- Implement advanced denoising and sharpness techniques
- Compare traditional and AI-based image processing methods
- Compute comprehensive image quality metrics

## Features

### Image Processing Techniques
1. **Demosaicing**
   - Edge-aware interpolation for GRBG Bayer pattern
   - Converts 12-bit RAW to RGB

2. **White Balance**
   - Gray world algorithm to remove color cast

3. **Gamma Correction**
   - sRGB gamma transformation
   - Converts 12-bit to 8-bit image

4. **Denoising Methods**
   - Gaussian Filtering
   - Median Filtering
   - Bilateral Filtering
   - Deep Learning Denoising (DnCNN)

5. **Sharpening Techniques**
   - Unsharp Mask
   - Laplacian Sharpening

### Metrics Computation
- Signal-to-Noise Ratio (SNR)
- Edge Strength
- Region of Interest (ROI) Analysis

## Requirements
- Python 3.8+
- Libraries:
  - OpenCV
  - NumPy
  - PyTorch
  - Streamlit
  - Rich

## Installation
```bash
git clone https://github.com/bharshavardhanreddy924/assign2.git
cd assign2
pip install -r requirements.txt
```

## Usage
### Streamlit Web Application
```bash
streamlit run app.py
```

### Command Line Processing
```bash
python pipe.py
```

## Project Structure
- `app.py`: Streamlit web interface
- `pipe.py`: Core ISP pipeline implementation
- `dncnn_custom.pth`: Pre-trained DnCNN weights

## Workflow
1. Upload 12-bit RAW image
2. Select processing techniques
3. Adjust parameters interactively
4. View processed images and quality metrics
5. Download results

## Input Specifications
- Format: 12-bit RAW
- Bayer Pattern: GRBG
- Resolution: 1920x1280

## Output
- Processed RGB images
- CSV with image quality metrics
- Visualizations of image processing stages

## Metrics Tracked
- SNR for dark/mid/bright regions
- Edge strength
- Comparative performance of processing methods

## Limitations
- Requires pre-trained DnCNN weights file
- Performance depends on input image quality
- Computational intensity for deep learning methods

## Future Work
- Expand deep learning denoising models
- Add more advanced sharpening techniques
- Implement adaptive processing algorithms
