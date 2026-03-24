"""
Machine Learning Module
This module contains YOLO detection and model training functionality.
"""

# ML functionality is optional and requires additional dependencies
import sys

YOLO_AVAILABLE = False
YOLO = None
YOLO_ERROR_MSG = ""

# Try multiple strategies to import YOLO
try:
    # Strategy 1: Direct import
    from ultralytics import YOLO as YOLOClass
    import torch
    YOLO = YOLOClass
    YOLO_AVAILABLE = True
    print(f"✓ YOLO loaded successfully (PyTorch {torch.__version__})")
except ImportError as e:
    YOLO_ERROR_MSG = f"YOLO not installed. Install with: pip install ultralytics torch\nError: {e}"
except (OSError, RuntimeError) as e:
    # PyTorch DLL loading issues (common in Python 3.13)
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    YOLO_ERROR_MSG = (
        f"YOLO unavailable: PyTorch compatibility issue with Python {python_version}\n"
        f"Error: {str(e)[:200]}\n\n"
        f"Solutions:\n"
        f"1. Use Python 3.10 or 3.11 (recommended):\n"
        f"   - Download Python 3.11: https://www.python.org/downloads/\n"
        f"   - Install dependencies: pip install ultralytics torch torchvision\n"
        f"   - Run with: py -3.11 main.py\n\n"
        f"2. Wait for PyTorch Python 3.13 support (future release)\n\n"
        f"3. Use virtual environment with Python 3.11:\n"
        f"   - python3.11 -m venv venv_ml\n"
        f"   - venv_ml\\Scripts\\activate (Windows) or source venv_ml/bin/activate (Linux/Mac)\n"
        f"   - pip install ultralytics torch torchvision\n\n"
        f"Note: The application works without ML features for traditional computer vision analysis."
    )

# If YOLO not available, show warning once
if not YOLO_AVAILABLE and YOLO_ERROR_MSG:
    print("\n" + "=" * 70)
    print("WARNING: YOLO Machine Learning Features Not Available")
    print("=" * 70)
    print(YOLO_ERROR_MSG)
    print("=" * 70 + "\n")

__all__ = ['YOLO_AVAILABLE', 'YOLO']

if YOLO_AVAILABLE:
    # Import ML modules when available
    # from .yolo_detector import YOLODetector
    # from .model_trainer import ModelTrainer
    # __all__.extend(['YOLODetector', 'ModelTrainer'])
    pass
