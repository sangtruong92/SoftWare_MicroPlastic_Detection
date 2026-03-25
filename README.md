# Fluorescence-based Microplastic Analyzer (FL-MPA)

**Advanced computer vision and machine learning software for detecting, analyzing, and classifying microplastic particles in microscopy images.**

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📋 Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Requirements & Dependencies](#requirements--dependencies)
- [Project Structure](#project-structure)
- [Usage Guide](#usage-guide)
- [Analysis Methods](#analysis-methods)
- [Image Generation](#image-generation)
- [Machine Learning Models](#machine-learning-models)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Features

### Analysis Methods
- **Quick Analysis**: Fast screening with Otsu thresholding
- **Deep Analysis**: Comprehensive shape detection with RGB multi-channel processing
- **Watershed Segmentation**: Separates touching/overlapping particles
- **Adaptive Analysis**: Handles uneven illumination and complex backgrounds
- **Edge Detection**: Identifies well-defined particle boundaries

### Microscopy Support
- **Brightfield Microscopy**: White background, dark particles
- **Fluorescent Microscopy**: Dark background, bright particles (auto-detected)
- **USB Camera Integration**: Real-time capture and analysis
- **ESP32-CAM Support**: Network-based image capture

### Shape Classification
Automatic classification into 5 morphometric categories:

| Shape | Characteristics | Aspect Ratio | Circularity | Use Case |
|-------|-----------------|--------------|-------------|----------|
| **Microbead/Pellet** | Spherical | 1.0-1.5 | > 0.82 | Microbeads, pellets |
| **Fiber/Filament** | Elongated threads | > 3.0 | < 0.55 | Fibers, filaments |
| **Fragment** | Irregular pieces | 1.2-3.0 | 0.38-0.72 | Broken particles |
| **Film** | Thin sheets | Variable | < 0.50 | Plastic films |
| **Irregular** | Miscellaneous | < 3.0 | < 0.55 | Unclassified |

### Additional Features
- Color analysis (HSV-based with fluorescent support)
- Synthetic dataset generation with ground truth
- YOLO v8 dataset export for training
- HTML benchmark reports with precision/recall/F1-score
- Side-by-side comparison mode
- Batch processing capabilities

---

## 📦 Installation

### Prerequisites
- **Python 3.8 or higher**
- **Git** (for cloning the repository)
- **pip** (Python package manager)

### Windows Installation

#### Step 1: Clone Repository
```powershell
cd D:\software_project\Python_Project
git clone https://github.com/sangtruong92/SoftWare_MicroPlastic_Detection.git
cd "Fluorescence-based Microplastic Analyzer"
```

#### Step 2: Create Virtual Environment (Recommended)
```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1
```

#### Step 3: Install Dependencies
```powershell
# Upgrade pip
python -m pip install --upgrade pip

# Install required packages
pip install -r requirements.txt
```

#### Step 4: Verify Installation
```powershell
python main.py --system-info
```

### macOS Installation

#### Step 1: Clone Repository
```bash
cd ~/Documents
git clone https://github.com/sangtruong92/SoftWare_MicroPlastic_Detection.git
cd "Fluorescence-based Microplastic Analyzer"
```

#### Step 2: Create Virtual Environment
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

#### Step 3: Install Dependencies
```bash
# Upgrade pip
python3 -m pip install --upgrade pip

# Install required packages
pip3 install -r requirements.txt

# For macOS: Install additional system dependencies
brew install libomp  # Required for OpenCV
```

#### Step 4: Verify Installation
```bash
python main.py --system-info
```

### Linux Installation

#### Step 1: Clone Repository
```bash
cd ~/Projects
git clone https://github.com/sangtruong92/SoftWare_MicroPlastic_Detection.git
cd "Fluorescence-based Microplastic Analyzer"
```

#### Step 2: Install System Dependencies
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y python3-dev python3-pip
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0  # For OpenCV
sudo apt-get install -y libxkbcommon-x11-0  # For PyQt5

# Fedora/RHEL
sudo dnf install -y python3-devel
sudo dnf install -y mesa-libGL glib2  # For OpenCV
```

#### Step 3: Create Virtual Environment
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

#### Step 4: Install Python Dependencies
```bash
# Upgrade pip
python3 -m pip install --upgrade pip

# Install required packages
pip3 install -r requirements.txt
```

#### Step 5: Verify Installation
```bash
python main.py --system-info
```

---

## 📚 Requirements & Dependencies

### Core Dependencies

```
opencv-python>=4.6.0          # Computer vision library
numpy>=1.22.0                 # Numerical computing
scipy>=1.8.0                  # Scientific computing
scikit-image>=0.19.0          # Image processing
matplotlib>=3.5.0             # Plotting and visualization
pandas>=1.4.0                 # Data manipulation
PyQt5>=5.15.0                 # GUI framework
Pillow>=9.0.0                 # Image I/O
```

### Machine Learning (Optional)

```
torch>=1.12.0                 # PyTorch - Deep learning framework
torchvision>=0.13.0           # Vision models
ultralytics>=8.0.0            # YOLO v8 models
tensorboard>=2.9.0            # TensorBoard visualization
```

### Check Installed Packages

```bash
# List all installed packages
pip list

# Check specific package version
pip show package_name

# Export requirements (for future use)
pip freeze > requirements_current.txt
```

---

## 🗂️ Project Structure

```
Fluorescence-based Microplastic Analyzer/
│
├── src/                              # Main source code
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── quick_analyzer.py        # Fast analysis algorithm
│   │   ├── deep_analyzer.py         # Deep analysis with morphometrics
│   │   ├── ml_benchmark_analyzer.py # ML-based analysis
│   │   └── watershed_analyzer.py    # Watershed segmentation
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── image_processor.py       # Image preprocessing
│   │   ├── feature_extractor.py     # Feature extraction
│   │   ├── shape_classifier.py      # Shape classification
│   │   └── color_analyzer.py        # Color analysis
│   │
│   ├── data_generation/
│   │   ├── __init__.py
│   │   ├── synthetic_generator.py   # Synthetic image generation
│   │   ├── yolo_exporter.py         # YOLO dataset export
│   │   └── ground_truth_generator.py # Ground truth data
│   │
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── yolo_detector.py         # YOLO v8 model interface
│   │   ├── model_trainer.py         # Model training scripts
│   │   └── model_evaluator.py       # Evaluation metrics
│   │
│   └── gui/
│       ├── __init__.py
│       ├── main_window.py           # Main GUI window
│       ├── dialogs/                 # Dialog windows
│       └── styles/                  # GUI themes and styles
│
├── config/                           # Configuration files
│   ├── __init__.py
│   ├── constants.py                 # Application constants
│   ├── settings.py                  # Default settings
│   └── shape_thresholds.py          # Shape classification thresholds
│
├── models/                           # Pre-trained models
│   ├── YoloV8s_v3/
│   │   ├── best.pt                  # Best YOLOv8s model
│   │   └── last.pt                  # Latest checkpoint
│   ├── Yolov8Seg_v1/                # Segmentation models
│   └── FPT_Team_Yolo26/             # Custom models
│
├── main.py                          # Application entry point
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
└── .gitignore                       # Git ignore rules
```

---

## 🚀 Quick Start

### Launch GUI Application
```powershell
# Windows
python main.py --gui

# macOS/Linux
python3 main.py --gui
```

### Command Line Quickstart

#### Quick Analysis
```bash
python main.py --quick-analyze image.jpg --output results.json
```

#### Deep Analysis
```bash
python main.py --deep-analyze image.jpg --output detailed_results.json
```

#### Generate Synthetic Data
```bash
# Generate 100 mixed-shape fluorescent images
python main.py --generate-data 100 --output-dir data/synthetic

# Generate 50 fiber-only images
python main.py --generate-data 50 --shape-type "Fiber only" --num-particles 20
```

#### Show System Info
```bash
python main.py --system-info
```

#### Display Classification Thresholds
```bash
python main.py --show-thresholds
```

---

## 🔧 Analysis Methods

### 1. Quick Analysis (Basic)
**Best for:** Real-time screening, large batches

```bash
python main.py --quick-analyze image.jpg --output results.json
```

**Features:**
- Otsu thresholding for fast segmentation
- Basic morphometric parameters
- Suitable for uniform illumination
- ~100-200 particles per second

### 2. Deep Analysis (Advanced)
**Best for:** High-accuracy detection, detailed classification. It offers different agorithms with: Watershed Segmentation, Adaptive Analysis

```bash
python main.py --deep-analyze image.jpg --output detailed_results.json
```

**Features:**
- Multi-channel RGB analysis
- CLAHE enhancement for uneven illumination
- Complete morphometric metrics
- Shape classification (5 categories)
- ~10-50 particles per second

### 2.1 Watershed Segmentation
**Best for:** Touching/overlapping particles

**Parameters:**
- Separates clustered particles
- Slower processing
- Higher accuracy for complex samples

### 2.2 Adaptive Analysis
**Best for:** Variable illumination

**Features:**
- Automatic threshold adaptation
- Local contrast enhancement
- Robust to lighting variations

---

## 🎨 Image Generation

### Generate Synthetic Training Data

#### Basic Syntax
```bash
python main.py --generate-data N [OPTIONS]
```

#### Options

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `--shape-type` | Mixed, Fiber, Filament, Microbead, Clean, Custom | Mixed | Which shapes to generate |
| `--color-type` | Fluorescent, Natural, Mixed Colors | Fluorescent | Color scheme |
| `--num-particles` | 1-50+ | 15 | Particles per image |

#### Examples

**Generate 100 mixed images:**
```bash
python main.py --generate-data 100 --output-dir data/synthetic
```

**Generate 50 fiber-only images:**
```bash
python main.py --generate-data 50 --shape-type "Fiber only" --num-particles 20 --output-dir data/fibers
```

**Generate images with natural colors:**
```bash
python main.py --generate-data 100 --color-type "Natural" --output-dir data/natural
```

### Generated Files

Each generated image produces:
- `synthetic_XXXX.png` - The synthetic image
- `synthetic_XXXX_gt.json` - Ground truth annotations

**Ground Truth Format:**
```json
{
  "image_size": [512, 512],
  "particles": [
    {
      "id": 1,
      "shape": "Fiber",
      "color": "Green",
      "bounding_box": [x, y, width, height],
      "mask": "polygon_coordinates",
      "area": 245.5,
      "circularity": 0.23,
      "aspect_ratio": 4.2
    }
  ]
}
```

---

## 🔍 Preprocessing Methods

### Available Methods

1. **Otsu Thresholding** (Automatic)
   - Best for: Uniform backgrounds
   - No parameter tuning needed
   
2. **Adaptive Thresholding**
   - Best for: Uneven illumination
   - Block size: 11-51 pixels (default: 31)
   
3. **CLAHE (Contrast Limited Adaptive Histogram Equalization)**
   - Best for: Low contrast images
   - Improves visibility of faint particles
   
4. **Bilateral Filtering**
   - Smooths noise while preserving edges
   - Parameters: diameter=9, sigmaColor=75, sigmaSpace=75

### Preprocessing Parameters

```python
from config.settings import PreprocessingParams

# Create parameters
params = PreprocessingParams(
    method='advanced',           # 'quick', 'advanced', 'adaptive'
    min_area=10,                 # Minimum particle area (pixels²)
    max_area=10000,              # Maximum particle area
    threshold_value=127,         # Manual threshold (0-255)
    use_clahe=True,              # Enhanced contrast
    clahe_clip_limit=2.0,        # CLAHE strength
    morphological_kernel=5        # Kernel size for morphological ops
)
```

---

## 🤖 Machine Learning Models

### Pre-trained Models

#### YOLOv8s (Nano)
- **File:** `models/YoloV8s_v3/best.pt`
- **Size:** ~21 MB
- **Speed:** 45-100 FPS
- **Best for:** General detection, real-time processing

#### YOLOv8 Segmentation
- **File:** `models/Yolov8Seg_v1/best.pt`
- **Size:** ~40 MB
- **Speed:** 30-60 FPS
- **Best for:** Precise boundaries, pixel-level masks

### Using Pre-trained Models

```python
from src.ml import YoloDetector

# Initialize detector
detector = YoloDetector(model_path='models/YoloV8s_v3/best.pt')

# Run detection
results = detector.detect(image)

# Get predictions
for detection in results:
    print(f"Class: {detection['class']}")
    print(f"Confidence: {detection['confidence']:.2%}")
    print(f"Box: {detection['bbox']}")
```

### Training Custom Models

```bash
# Export dataset in YOLO format
python main.py --export-yolo --input data/annotated --output data/yolo_dataset

# Train model (requires ultralytics and PyTorch)
yolo detect train data=data/yolo_dataset/data.yaml model=yolov8s.pt epochs=100
```

---

## ✅ Testing Models

### Evaluate Pre-trained Models

```python
from src.ml import ModelEvaluator
from src.analysis import DeepAnalyzer

# Load test dataset
test_images = ['test1.jpg', 'test2.jpg', 'test3.jpg']

# Run analysis on test set
analyzer = DeepAnalyzer()
results = []

for image_path in test_images:
    result = analyzer.analyze(image_path)
    results.append(result)

# Generate benchmark report
evaluator = ModelEvaluator()
evaluator.generate_report(results, output_path='benchmark_report.html')
```

### Accuracy Metrics

Metrics calculated:
- **Precision:** True Positives / (True Positives + False Positives)
- **Recall:** True Positives / (True Positives + False Negatives)
- **F1-Score:** 2 × (Precision × Recall) / (Precision + Recall)
- **IoU (Intersection over Union):** For bounding box accuracy
- **Dice Coefficient:** Segmentation mask overlap

---

## 📖 Usage Guide

### GUI Application

#### 1. Launch Application
```bash
python main.py --gui
```

#### 2. Capture or Load Image
- **Capture from Camera:** USB Camera or ESP32-CAM
- **Load from File:** Select image from disk
- **Batch Process:** Process multiple images

#### 3. Select Analysis Method
- **Quick:** Fast screening
- **Deep:** Detailed analysis
- **Adaptive:** Variable lighting
- **ML:** Machine learning detection

#### 4. Configure Parameters
- Adjust preprocessing settings
- Set classification thresholds
- Choose output format

#### 5. Run Analysis
- Click "Analyze" button
- Monitor progress
- View results in real-time

#### 6. Export Results
- Save as JSON, CSV, or images
- Generate HTML report
- Export YOLO dataset

### Python API

#### Basic Analysis
```python
import cv2
from src.analysis import QuickAnalyzer
from config.settings import PreprocessingParams

# Load image
image = cv2.imread('sample.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Analyze
analyzer = QuickAnalyzer()
params = PreprocessingParams(method='advanced')
result = analyzer.analyze(image, params)

# Process results
print(f"Detected: {result.num_detections} particles")
for feature in result.features:
    print(f"  {feature['shape']}: Area={feature['area']:.1f}")
```

#### Batch Processing
```python
from pathlib import Path
from src.analysis import DeepAnalyzer

# Process multiple images
image_dir = Path('images/')
analyzer = DeepAnalyzer()

for image_file in image_dir.glob('*.jpg'):
    result = analyzer.analyze(str(image_file))
    print(f"{image_file.name}: {result.num_detections} particles")
```

---

## 📊 API Reference

### Main Classes

#### QuickAnalyzer
```python
from src.analysis import QuickAnalyzer

analyzer = QuickAnalyzer()
result = analyzer.analyze(image, preprocessing_params)
```

**Returns:**
- `num_detections` - Number of particles found
- `features` - List of detected particles with properties
- `processing_time` - Time taken in seconds

#### DeepAnalyzer
```python
from src.analysis import DeepAnalyzer

analyzer = DeepAnalyzer()
result = analyzer.analyze(image, preprocessing_params)
```

**Returns:**
- All QuickAnalyzer results plus:
- `fiber_count` - Number of detected fibers
- `avg_processing_per_object` - Time per particle

#### SyntheticDataGenerator
```python
from src.data_generation import SyntheticDataGenerator
from config.settings import SyntheticImageParams

generator = SyntheticDataGenerator()
params = SyntheticImageParams(
    shape_type='Mixed',
    color_type='Fluorescent',
    num_particles=15
)
image, ground_truth = generator.generate(params)
```

---

## 💡 Examples

### Example 1: Analyze Single Image
```bash
# Windows
python main.py --deep-analyze sample.jpg --output result.json

# macOS/Linux
python3 main.py --deep-analyze sample.jpg --output result.json
```

### Example 2: Generate Training Dataset
```bash
# Create 500 synthetic images for training
python main.py --generate-data 500 --output-dir data/training --num-particles 20
```

### Example 3: Batch Analysis with Python
```python
import json
from pathlib import Path
from src.analysis import DeepAnalyzer
from config.settings import PreprocessingParams

analyzer = DeepAnalyzer()
params = PreprocessingParams(method='advanced', min_area=5)

results = {}
for image_path in Path('images').glob('*.jpg'):
    result = analyzer.analyze(str(image_path), params)
    results[image_path.name] = {
        'num_particles': result.num_detections,
        'processing_time': result.processing_time,
        'particles': [
            {
                'id': f['id'],
                'shape': f['shape'],
                'area': f['area']
            }
            for f in result.features
        ]
    }

# Save results
with open('batch_results.json', 'w') as f:
    json.dump(results, f, indent=2)
```

---

## 🐛 Troubleshooting

### Common Issues

#### Issue: "ModuleNotFoundError: No module named 'PyQt5'"
**Solution:**
```bash
pip install PyQt5
# or use conda
conda install pyqt
```

#### Issue: "ImportError: libGL.so.1" (Linux)
**Solution:**
```bash
sudo apt-get install libgl1-mesa-glx
```

#### Issue: "ImportError: No module named 'torch'" (ML features)
**Solution:**
```bash
# CPU version
pip install torch torchvision

# GPU version (NVIDIA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Issue: Application runs slowly
**Solution:**
1. Use Quick Analysis instead of Deep Analysis
2. Reduce image resolution
3. Decrease `min_area` parameter
4. Use GPU acceleration if available

### Getting Help

- Check `requirements.txt` for dependency versions
- Review error messages in console output
- See `docs/` folder for detailed guides
- Open an issue on GitHub with:
  - Python version
  - OS and OS version
  - Full error message
  - Steps to reproduce

---

## 📝 Command Line Reference

```bash
# Launch GUI
python main.py --gui

# Quick analysis
python main.py --quick-analyze IMAGE.jpg --output results.json

# Deep analysis
python main.py --deep-analyze IMAGE.jpg --output results.json

# Generate synthetic data
python main.py --generate-data 100 --output-dir data/synthetic

# Show shape thresholds
python main.py --show-thresholds

# Display system information
python main.py --system-info

# Help
python main.py --help
```

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## � Contributors
- **Sang Truong** - Contributor - [sangtruong92](https://github.com/sangtruong92)
- **CV Son** - Contributor - cvson@utexas.edu

---

## �📄 License

This project is licensed under the MIT License - see LICENSE file for details.

---

## 📬 Contact & Support

- **GitHub:** [github.com/sangtruong92/SoftWare_MicroPlastic_Detection](https://github.com/sangtruong92/SoftWare_MicroPlastic_Detection)
- **Issues:** Report bugs and request features via [GitHub Issues](https://github.com/sangtruong92/SoftWare_MicroPlastic_Detection/issues)

---

## 📚 References

- OpenCV Documentation: https://docs.opencv.org/
- YOLOv8 Documentation: https://docs.ultralytics.com/
- PyQt5 Documentation: https://www.riverbankcomputing.com/static/Docs/PyQt5/
- Scientific Papers on Microplastic Analysis: See `docs/` folder

---

## ✨ Acknowledgments

Built with contributions from the microplastic research community.

**Last Updated:** March 24, 2026

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
