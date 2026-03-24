# Microplastic Analyzer Pro

A Python application for detecting, analyzing, and classifying microplastic particles in microscopy images using computer vision and machine learning.

**Platforms:** Windows, macOS, Linux  
**Python:** 3.8+

## Features

### Analysis Methods

| Method | Speed | Use Case |
|--------|-------|----------|
| **Quick (Basic)** | Fast | Real-time screening, Otsu thresholding |
| **Deep (Advanced)** | Medium | Multi-channel RGB with CLAHE enhancement |
| **Watershed** | Slow | Separating touching/overlapping particles |
| **Edge** | Fast | Well-defined particle boundaries |
| **Adaptive** | Medium | Uneven illumination, complex backgrounds |

### Microscopy Support

- **Brightfield:** White background, dark particles
- **Fluorescent:** Dark background, bright particles (auto-detected)

### Shape Classification

Particles are classified into four categories based on morphometric parameters:

| Shape | Circularity | Eccentricity | Aspect Ratio |
|-------|-------------|--------------|--------------|
| Microbead/Pellet | ≥ 0.82 | ≤ 0.60 | < 2.0 |
| Fiber/Filament | < 0.55 | ≥ 0.70 | > 2.5 |
| Fragment | 0.38 - 0.72 | 0.20 - 0.75 | 1.0 - 3.0 |
| Irregular | < 0.55 | variable | < 3.0 |

### Additional Features

- **Synthetic Data Generation:** Create training datasets with known ground truth
- **YOLO Dataset Export:** Generate YOLOv8-compatible training data
- **Benchmark Reports:** HTML reports with precision, recall, F1-score
- **Comparison Mode:** Side-by-side Quick vs Deep analysis
- **Color Analysis:** HSV-based classification with fluorescent support

## Quick Start

### Installation

```bash
# Clone or download the repository
cd MircrosPlastic_Software

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

### Basic Usage

**GUI Mode (recommended):**
```bash
python main.py
```

**Command Line:**
```bash
# Quick analysis
python main.py --quick path/to/image.png

# Deep analysis
python main.py --deep path/to/image.png

# Batch processing
python main.py --batch path/to/folder --method advanced
```

### Fluorescent Microscopy

1. Load your image
2. Click **"Fluorescent Microscopy"** preset button
3. Run **"Deep Analysis"**

The software automatically detects dark backgrounds and optimizes thresholding.

## Project Structure

```
MircrosPlastic_Software/
├── main.py                 # Application entry point
├── requirements.txt        # Python dependencies
├── config/
│   ├── constants.py        # Shape thresholds, color definitions
│   └── settings.py         # Preprocessing parameters
├── src/
│   ├── analysis/           # Quick and Deep analyzers
│   ├── core/               # Image processing, shape analysis
│   ├── data_generation/    # Synthetic image generator
│   ├── gui/                # PyQt5 interface
│   └── ml/                 # YOLO integration
├── models/                 # Pre-trained YOLO models
├── docs/                   # Documentation
└── benchmark_results/      # Generated reports
```

## Preprocessing Methods

### Quick Analysis (basic)
```python
# Fast Otsu thresholding
# Best for: Real-time processing, clean images
method='basic'
```

### Deep Analysis (advanced)
```python
# Multi-channel RGB with CLAHE
# Best for: Color variation, mixed samples
method='advanced'
```

### Watershed
```python
# Marker-based segmentation
# Best for: Touching particles
method='watershed'
```

### Edge Detection
```python
# Canny + Scharr edge detection
# Best for: Clear particle boundaries
method='edge'
```

### Adaptive
```python
# Background subtraction + adaptive thresholding
# Best for: Uneven illumination
method='adaptive'
```

## Configuration

Key settings in `config/constants.py`:

```python
# Shape classification thresholds
SHAPE_THRESHOLDS = {
    'Microbead': {'circularity_min': 0.60, 'eccentricity_max': 0.60},
    'Fiber': {'aspect_ratio_min': 2.5, 'eccentricity_min': 0.70},
    'Fragment': {'circularity_min': 0.38, 'circularity_max': 0.72},
    'Irregular': {'circularity_max': 0.50}
}

# Background detection
BG_WHITE_THRESHOLD = 200  # Brightfield detection
BG_BLACK_THRESHOLD = 50   # Fluorescent detection
```

## Documentation

| Document | Description |
|----------|-------------|
| [docs/INSTALL.md](docs/INSTALL.md) | Detailed installation guide |
| [docs/USAGE.md](docs/USAGE.md) | Complete usage reference |
| [docs/PREPROCESSING_GUIDE.md](docs/PREPROCESSING_GUIDE.md) | Preprocessing method details |
| [BUILD_MACOS_GUIDE.md](BUILD_MACOS_GUIDE.md) | Building macOS .app and .dmg |
| [BUILD_EXE_GUIDE_VI.md](BUILD_EXE_GUIDE_VI.md) | Building Windows .exe |
| [FLUORESCENT_MICROSCOPY_GUIDE.md](FLUORESCENT_MICROSCOPY_GUIDE.md) | Fluorescent image optimization |
| [YOLO_TRAINING_GUIDE.md](YOLO_TRAINING_GUIDE.md) | Custom model training |
| [MORPHOMETRIC_PARAMETERS_REFERENCE.md](MORPHOMETRIC_PARAMETERS_REFERENCE.md) | Shape classification formulas |

## Requirements

- Python 3.8+
- OpenCV 4.x
- NumPy, SciPy, scikit-image
- PyQt5 (GUI)
- Ultralytics (optional, for YOLO)

See `requirements.txt` for complete list.

## License

MIT License
