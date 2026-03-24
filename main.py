"""
Microplastic Analyzer - Main Entry Point

This is the main entry point for the Microplastic Analyzer application.
It provides both GUI and command-line interfaces for analyzing microplastic images.

Cross-platform compatible: Works on Windows, macOS, and Linux.
"""

import sys
import argparse
import platform
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def get_platform_info():
    """Get current platform information"""
    return {
        'system': platform.system(),
        'release': platform.release(),
        'version': platform.version(),
        'machine': platform.machine(),
        'python_version': platform.python_version()
    }


def run_gui():
    """Launch the GUI application"""
    try:
        from PyQt5 import QtWidgets
        from src.gui import MicroplasticAnalyzerGUI
        
        print("Launching Microplastic Analyzer GUI...")
        print(f"Platform: {platform.system()} {platform.release()}")
        print(f"Python: {platform.python_version()}")
        
        app = QtWidgets.QApplication(sys.argv)
        
        # Set application style based on platform
        if platform.system() == 'Darwin':  # macOS
            app.setStyle('Macintosh')
        elif platform.system() == 'Windows':
            app.setStyle('Fusion')
        else:  # Linux and others
            app.setStyle('Fusion')
        
        window = MicroplasticAnalyzerGUI()
        window.show()
        sys.exit(app.exec_())
        
    except ImportError as e:
        print(f"Error: Unable to load GUI. {e}")
        print("Please install PyQt5: pip install PyQt5")
        print("\nIf you have the original GUI, you can also run:")
        print("  python MicroPlastic_GUI_ML_V7.py")
        sys.exit(1)
    except Exception as e:
        print(f"Error launching GUI: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def run_quick_analysis(image_path: str, output_path: str = None):
    """Run quick analysis on an image
    
    Uses literature-based morphometric parameters (grouped categories):
    - Microbead/Pellet: AR 1.0-1.5, C > 0.8, E < 0.4
    - Fiber/Filament: AR > 3.0, E ≥ 0.7 OR C ≤ 0.3 (relaxed criteria)
    - Fragment: AR 1.2-3.0, C 0.4-0.7, E 0.5-0.8
    - Film: AR variable, C < 0.5
    - Irregular: AR < 3.0, C < 0.5
    """
    import cv2
    from config.settings import PreprocessingParams
    from src.analysis import QuickAnalyzer
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        sys.exit(1)
    
    # Convert to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Run analysis
    print(f"Analyzing image: {image_path}")
    analyzer = QuickAnalyzer()
    params = PreprocessingParams(method='advanced', min_area=10)
    result = analyzer.analyze(image, params)
    
    # Print results
    print(f"\nAnalysis completed in {result.processing_time:.2f} seconds")
    print(f"Detected {result.num_detections} particles")
    print(f"Background color: {result.background_color}")
    print(f"\nDetected particles:")
    
    for feature in result.features:
        print(f"  Particle {feature['id']}: {feature['shape']} - {feature['color']} "
              f"(Area: {feature['area']:.1f} pixels)")
    
    # Save results if output path specified
    if output_path:
        import json
        with open(output_path, 'w') as f:
            # Convert result to JSON-serializable format
            output_data = {
                'num_detections': result.num_detections,
                'processing_time': result.processing_time,
                'background_color': result.background_color,
                'features': [
                    {
                        'id': f['id'],
                        'shape': f['shape'],
                        'color': f['color'],
                        'area': f['area'],
                        'circularity': f['circularity'],
                        'centroid': f['centroid']
                    }
                    for f in result.features
                ]
            }
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {output_path}")


def run_deep_analysis(image_path: str, output_path: str = None):
    """Run deep analysis on an image with detailed morphometric metrics
    
    Provides comprehensive shape classification using grouped categories:
    - Aspect Ratio (AR): Length/Width ratio
    - Circularity (C): 4π×Area/Perimeter² (0-1)
    - Eccentricity (E): Elongation measure (0-1)
    - Solidity, Rectangularity, and other advanced metrics
    
    Grouped shape categories (simplified from 7 to 5):
    - Microbead/Pellet: Spherical particles
    - Fiber/Filament: Elongated thread-like structures
    - Fragment, Film, Irregular: Maintained as separate categories
    
    See MORPHOMETRIC_PARAMETERS_REFERENCE.md and SHAPE_GROUPING_UPDATE.md for details.
    """
    import cv2
    from config.settings import PreprocessingParams
    from src.analysis import DeepAnalyzer
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        sys.exit(1)
    
    # Convert to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Run analysis
    print(f"Performing deep analysis on: {image_path}")
    analyzer = DeepAnalyzer()
    params = PreprocessingParams(method='advanced', min_area=3)
    result = analyzer.analyze(image, params)
    
    # Print results
    print(f"\nAnalysis completed in {result.processing_time:.2f} seconds")
    print(f"Detected {result.num_detections} particles")
    print(f"Fiber count: {result.fiber_count}")
    print(f"Background color: {result.background_color}")
    print(f"Average time per particle: {result.avg_processing_per_object:.4f} seconds")
    print(f"\nDetailed particle information:")
    
    for feature in result.features:
        print(f"\nParticle {feature['id']}:")
        print(f"  Shape: {feature['shape']}")
        print(f"  Color: {feature['color']}")
        print(f"  Area: {feature['area']:.1f} pixels")
        print(f"  Circularity: {feature['circularity']:.3f}")
        print(f"  Eccentricity: {feature['eccentricity']:.3f}")
        print(f"  Aspect Ratio: {feature['aspect_ratio']:.2f}")
        print(f"  Solidity: {feature['solidity']:.3f}")
    
    # Save results if output path specified
    if output_path:
        import json
        with open(output_path, 'w') as f:
            output_data = {
                'num_detections': result.num_detections,
                'processing_time': result.processing_time,
                'fiber_count': result.fiber_count,
                'background_color': result.background_color,
                'avg_processing_per_object': result.avg_processing_per_object,
                'features': [
                    {
                        'id': f['id'],
                        'shape': f['shape'],
                        'color': f['color'],
                        'area': f['area'],
                        'circularity': f['circularity'],
                        'eccentricity': f['eccentricity'],
                        'aspect_ratio': f['aspect_ratio'],
                        'solidity': f['solidity'],
                        'centroid': f['centroid']
                    }
                    for f in result.features
                ]
            }
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {output_path}")


def generate_synthetic_data(num_images: int, output_dir: str, shape_type: str = 'Mixed', 
                           color_type: str = 'Fluorescent', num_particles: int = 15):
    """Generate synthetic training data with literature-based morphometric parameters
    
    Generates particles matching scientific literature criteria:
    - Fiber: Very elongated (AR > 3.0, length 40-90px, width 2-5px)
    - Filament: Extremely elongated (AR > 5.0, length 60-120px, width 2-4px)
    - Microbead: Nearly spherical (AR 1.0-1.2, radius 3-7px)
    - Pellet: Slightly oval (AR 1.0-1.5, radius 18-28px)
    - Fragment: Moderate irregular (AR 1.2-3.0)
    - Film: Thin sheets (variable AR, 40-80 × 12-28px)
    
    Args:
        num_images: Number of images to generate
        output_dir: Directory to save images
        shape_type: 'Mixed', 'Fiber only', 'Filament only', etc.
        color_type: 'Fluorescent', 'Natural', or 'Mixed Colors'
        num_particles: Number of particles per image
    """
    import cv2
    from pathlib import Path
    from config.settings import SyntheticImageParams
    from src.data_generation import SyntheticImageGenerator
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {num_images} synthetic images...")
    print(f"  Shape type: {shape_type}")
    print(f"  Color type: {color_type}")
    print(f"  Particles per image: {num_particles}")
    print(f"  Using literature-based morphometric parameters")
    generator = SyntheticImageGenerator()
    params = SyntheticImageParams(
        shape_type=shape_type,
        color_type=color_type,
        num_particles=num_particles
    )
    
    for i in range(num_images):
        # Generate image
        image, ground_truth = generator.generate(params)
        
        # Save image
        image_file = output_path / f"synthetic_{i+1:04d}.png"
        cv2.imwrite(str(image_file), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        # Save ground truth
        import json
        gt_file = output_path / f"synthetic_{i+1:04d}_gt.json"
        with open(gt_file, 'w') as f:
            json.dump(ground_truth, f, indent=2)
        
        if (i + 1) % 10 == 0:
            print(f"  Generated {i+1}/{num_images} images...")
    
    print(f"\nGeneration complete! Images saved to: {output_dir}")
    print(f"\nNote: Generated images use literature-based morphometric parameters.")
    print(f"See MORPHOMETRIC_PARAMETERS_REFERENCE.md for parameter details.")


def show_shape_thresholds():
    """Display shape classification thresholds"""
    from config.constants import SHAPE_THRESHOLDS
    
    print("=" * 70)
    print("MICROPLASTIC SHAPE CLASSIFICATION THRESHOLDS")
    print("Based on Scientific Literature (Updated Feb 2, 2026)")
    print("=" * 70)
    print("\nAR = Aspect Ratio, C = Circularity, E = Eccentricity\n")
    
    for shape_name, thresholds in SHAPE_THRESHOLDS.items():
        print(f"\n{shape_name}:")
        print(f"  {'─' * 60}")
        for param, value in thresholds.items():
            param_display = param.replace('_', ' ').title()
            if isinstance(value, float):
                print(f"  {param_display:.<25} {value:.2f}")
            else:
                print(f"  {param_display:.<25} {value}")
    
    print("\n" + "=" * 70)
    print("Reference: MORPHOMETRIC_PARAMETERS_REFERENCE.md")
    print("=" * 70)


def show_system_info():
    """Display system and dependency information"""
    import cv2
    import numpy as np
    import matplotlib
    
    print("=" * 60)
    print("MICROPLASTIC ANALYZER - SYSTEM INFORMATION")
    print("=" * 60)
    
    # Platform info
    info = get_platform_info()
    print(f"\nPlatform:")
    print(f"  OS: {info['system']} {info['release']}")
    print(f"  Architecture: {info['machine']}")
    print(f"  Python: {info['python_version']}")
    
    # Python packages
    print(f"\nCore Dependencies:")
    print(f"  NumPy: {np.__version__}")
    print(f"  OpenCV: {cv2.__version__}")
    print(f"  Matplotlib: {matplotlib.__version__}")
    
    try:
        from PyQt5.QtCore import QT_VERSION_STR, PYQT_VERSION_STR
        print(f"  PyQt5: {PYQT_VERSION_STR} (Qt {QT_VERSION_STR})")
    except ImportError:
        print(f"  PyQt5: Not installed")
    
    try:
        from scipy import __version__ as scipy_version
        print(f"  SciPy: {scipy_version}")
    except ImportError:
        print(f"  SciPy: Not installed")
    
    try:
        from skimage import __version__ as skimage_version
        print(f"  scikit-image: {skimage_version}")
    except ImportError:
        print(f"  scikit-image: Not installed")
    
    try:
        import pandas as pd
        print(f"  Pandas: {pd.__version__}")
    except ImportError:
        print(f"  Pandas: Not installed")
    
    # ML dependencies
    print(f"\nMachine Learning (Optional):")
    try:
        import torch
        print(f"  PyTorch: {torch.__version__}")
        print(f"  CUDA Available: {torch.cuda.is_available()}")
    except (ImportError, OSError):
        print(f"  PyTorch: Not installed or incompatible")
    
    try:
        from ultralytics import __version__ as yolo_version
        print(f"  Ultralytics YOLO: {yolo_version}")
    except (ImportError, OSError):
        print(f"  Ultralytics YOLO: Not installed or incompatible")
    
    # Platform-specific notes
    print(f"\nPlatform Notes:")
    if platform.system() == 'Darwin':
        print(f"  macOS detected - Using native Macintosh UI style")
        print(f"  For best performance, ensure Xcode Command Line Tools are installed")
    elif platform.system() == 'Windows':
        print(f"  Windows detected - Using Fusion UI style")
        if sys.version_info >= (3, 13):
            print(f"  Note: Python 3.13+ has PyTorch compatibility issues")
            print(f"  For ML features, consider using Python 3.10 or 3.11")
    else:
        print(f"  Linux/Unix detected - Using Fusion UI style")
        print(f"  Ensure required system libraries are installed (libGL, etc.)")
    
    print("\n" + "=" * 60)
    print("For installation instructions, see README.md or INSTALL.md")
    print("=" * 60)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Microplastic Analyzer - Detect and analyze microplastic particles in images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch GUI
  python main.py --gui
  
  # Quick analysis
  python main.py --quick-analyze image.jpg --output results.json
  
  # Deep analysis
  python main.py --deep-analyze image.jpg --output detailed_results.json
  
  # Generate synthetic data (mixed shapes, fluorescent colors)
  python main.py --generate-data 100 --output-dir data/synthetic
  
  # Generate fiber-only synthetic data
  python main.py --generate-data 50 --shape-type "Fiber only" --num-particles 20
  
  # Show shape classification thresholds
  python main.py --show-thresholds
  
  # Show system information
  python main.py --system-info

Cross-platform: Works on Windows, macOS, and Linux
Shape parameters based on scientific literature (Feb 2026 update)
        """
    )
    
    parser.add_argument('--gui', action='store_true',
                       help='Launch GUI application')
    parser.add_argument('--quick-analyze', metavar='IMAGE',
                       help='Perform quick analysis on an image')
    parser.add_argument('--deep-analyze', metavar='IMAGE',
                       help='Perform deep analysis on an image')
    parser.add_argument('--generate-data', type=int, metavar='N',
                       help='Generate N synthetic images')
    parser.add_argument('--shape-type', metavar='TYPE', default='Mixed',
                       help='Shape type for synthetic data (Mixed, Fiber only, etc.)')
    parser.add_argument('--color-type', metavar='TYPE', default='Fluorescent',
                       help='Color type for synthetic data (Fluorescent, Natural, Mixed Colors)')
    parser.add_argument('--num-particles', type=int, metavar='N', default=15,
                       help='Number of particles per synthetic image (default: 15)')
    parser.add_argument('--output', '-o', metavar='PATH',
                       help='Output file path for analysis results')
    parser.add_argument('--output-dir', metavar='DIR',
                       help='Output directory for generated data')
    parser.add_argument('--system-info', action='store_true',
                       help='Display system and dependency information')
    parser.add_argument('--show-thresholds', action='store_true',
                       help='Display shape classification thresholds')
    
    args = parser.parse_args()
    
    # Show system info if requested
    if args.system_info:
        show_system_info()
        return
    
    # Show shape thresholds if requested
    if args.show_thresholds:
        show_shape_thresholds()
        return
    
    # If no arguments, launch GUI (same as original behavior)
    if len(sys.argv) == 1:
        print("No arguments provided - launching GUI interface...")
        run_gui()
        sys.exit(0)
    
    # Execute requested action
    if args.gui:
        run_gui()
    
    elif args.quick_analyze:
        run_quick_analysis(args.quick_analyze, args.output)
    
    elif args.deep_analyze:
        run_deep_analysis(args.deep_analyze, args.output)
    
    elif args.generate_data:
        output_dir = args.output_dir or 'data/synthetic'
        generate_synthetic_data(
            args.generate_data, 
            output_dir,
            shape_type=args.shape_type,
            color_type=args.color_type,
            num_particles=args.num_particles
        )
    
    else:
        print("Error: No action specified. Use --help for usage information.")
        sys.exit(1)


if __name__ == '__main__':
    main()
