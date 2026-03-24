"""
Main Window - Microplastic Analyzer GUI
Complete GUI implementation using modular components
"""

import sys
import platform
import cv2
import numpy as np
import json
from pathlib import Path
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFileDialog, QTextEdit,
                             QComboBox, QSpinBox, QGroupBox, QTabWidget,
                             QTableWidget, QTableWidgetItem, QProgressBar,
                             QMessageBox, QSplitter, QDoubleSpinBox, QCheckBox,
                             QGridLayout, QFormLayout, QScrollArea, QApplication)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting

from config.settings import PreprocessingParams, SyntheticImageParams
from config.constants import SHAPE_GROUP_MAPPING
from src.analysis import QuickAnalyzer, DeepAnalyzer, MLBenchmarkAnalyzer, StatisticsComparator
from src.data_generation import SyntheticImageGenerator

# Machine Learning imports - handle PyTorch compatibility issues
import sys
from src.ml import YOLO_AVAILABLE, YOLO

# Build detailed error message for UI
YOLO_ERROR_MESSAGE = ""
if not YOLO_AVAILABLE:
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    YOLO_ERROR_MESSAGE = (
        f"YOLO ML features unavailable (Python {python_version})\n\n"
        f"PyTorch doesn't fully support Python 3.13 yet.\n\n"
        f"SOLUTIONS:\n"
        f"1. Use Python 3.10 or 3.11:\n"
        f"   • Download: https://www.python.org/downloads/\n"
        f"   • Install: pip install ultralytics torch\n"
        f"   • Run: py -3.11 main.py\n\n"
        f"2. Create Python 3.11 virtual environment:\n"
        f"   • python3.11 -m venv venv_ml\n"
        f"   • Activate and install packages\n\n"
        f"The app works without ML for traditional CV analysis."
    )


def map_to_grouped_shape(shape_name: str) -> str:
    """
    Map individual shape names to grouped categories.
    Handles legacy individual names from ground truth data.
    
    Args:
        shape_name: Individual or grouped shape name
        
    Returns:
        Grouped shape name
    """
    return SHAPE_GROUP_MAPPING.get(shape_name, shape_name)


def _get_camera_backend() -> int:
    """
    Get the appropriate OpenCV camera backend for the current OS.
    
    Returns:
        OpenCV camera backend constant:
        - cv2.CAP_AVFOUNDATION: macOS (AVFoundation framework)
        - cv2.CAP_DSHOW: Windows (DirectShow)
        - cv2.CAP_V4L2: Linux (Video4Linux2)
    """
    system = platform.system()
    
    if system == 'Darwin':
        # macOS uses AVFoundation
        return cv2.CAP_AVFOUNDATION
    elif system == 'Windows':
        # Windows uses DirectShow
        return cv2.CAP_DSHOW
    else:
        # Linux and other Unix-like systems use Video4Linux2
        return cv2.CAP_V4L2


class AnalysisThread(QThread):
    """Background thread for image analysis"""
    finished = pyqtSignal(object)
    progress = pyqtSignal(str)
    
    def __init__(self, image, params, analysis_type='quick'):
        super().__init__()
        self.image = image
        self.params = params
        self.analysis_type = analysis_type
        
    def run(self):
        try:
            self.progress.emit("Analyzing image...")
            
            if self.analysis_type == 'quick':
                analyzer = QuickAnalyzer()
            else:
                analyzer = DeepAnalyzer()
            
            result = analyzer.analyze(self.image, self.params)
            self.finished.emit(result)
            
        except Exception as e:
            self.progress.emit(f"Error: {str(e)}")
            self.finished.emit(None)


class ImageLabel(QLabel):
    """Custom label for displaying images with zoom support"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 300)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("border: 2px solid #ccc; background-color: #f0f0f0;")
        self.setText("No Image Loaded")
        self.setMouseTracking(True)
        self.setScaledContents(False)
        
        # Zoom state
        self.zoom_factor = 1.0
        self.original_pixmap = None
        self.original_image = None
        self.zoomed_pixmap = None  # Store cropped region
        self.zoom_region = None  # Store the current zoom region (x, y, w, h)
        self.is_panning = False
        self.last_mouse_pos = None
        self.zoom_box_start = None
        self.zoom_box_enabled = False
        self.parent_scroll = None
        
        # Rubber band for zoom box
        self.rubber_band = QtWidgets.QRubberBand(QtWidgets.QRubberBand.Rectangle, self)
        
    def set_scroll_area(self, scroll_area):
        """Set parent scroll area for proper sizing"""
        self.parent_scroll = scroll_area
        
    def set_image(self, image):
        """Display numpy array as image"""
        if image is None:
            return
            
        # Store original image
        self.original_image = image.copy()
        
        # Convert to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        
        # Convert to QImage
        h, w, ch = image.shape
        bytes_per_line = ch * w
        qt_image = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Store original pixmap
        self.original_pixmap = QPixmap.fromImage(qt_image)
        self.zoom_factor = 1.0
        self.update_display()
        
    def update_display(self):
        """Update the displayed image with current zoom"""
        if self.original_pixmap is None:
            return
        
        # Use zoomed region if available, otherwise use full original
        display_pixmap = self.zoomed_pixmap if self.zoomed_pixmap else self.original_pixmap
            
        if self.zoom_factor == 1.0:
            # At 100% zoom, fit to available space in scroll area
            if self.parent_scroll:
                available_size = self.parent_scroll.viewport().size()
                # Leave some margin
                available_size.setWidth(available_size.width() - 10)
                available_size.setHeight(available_size.height() - 10)
                scaled = display_pixmap.scaled(available_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            else:
                scaled = display_pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            self.setPixmap(scaled)
            self.resize(scaled.size())
        else:
            # Apply zoom - show at original size * zoom factor
            new_size = display_pixmap.size() * self.zoom_factor
            scaled = display_pixmap.scaled(new_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.setPixmap(scaled)
            self.resize(scaled.size())
            
    def zoom_in(self):
        """Zoom in by 20%"""
        self.zoom_factor *= 1.2
        self.update_display()
        
    def zoom_out(self):
        """Zoom out by 20%"""
        self.zoom_factor /= 1.2
        if self.zoom_factor < 1.0:
            self.zoom_factor = 1.0
        self.update_display()
        
    def reset_zoom(self):
        """Reset zoom to 100% and show full original image"""
        self.zoom_factor = 1.0
        self.zoom_region = None
        self.zoomed_pixmap = None
        self.update_display()
        
    def enable_zoom_box(self, enabled):
        """Enable/disable zoom box selection"""
        self.zoom_box_enabled = enabled
        self.setCursor(Qt.CrossCursor if enabled else Qt.ArrowCursor)
        
    def mousePressEvent(self, event):
        """Handle mouse press for zoom and pan"""
        if self.zoom_box_enabled and event.button() == Qt.LeftButton:
            # Start zoom box
            self.zoom_box_start = event.pos()
            self.rubber_band.setGeometry(QtCore.QRect(self.zoom_box_start, QtCore.QSize()))
            self.rubber_band.show()
        elif event.button() == Qt.MiddleButton:
            # Start panning
            self.is_panning = True
            self.last_mouse_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
        elif self.zoom_box_enabled and event.button() == Qt.RightButton:
            # Right click to zoom out
            self.zoom_out()
            
    def mouseMoveEvent(self, event):
        """Handle mouse move for zoom box and pan"""
        if self.zoom_box_start is not None:
            # Update rubber band
            self.rubber_band.setGeometry(QtCore.QRect(self.zoom_box_start, event.pos()).normalized())
        elif self.is_panning and self.last_mouse_pos is not None:
            # Pan the image (would need scroll area support)
            self.last_mouse_pos = event.pos()
            
    def mouseReleaseEvent(self, event):
        """Handle mouse release"""
        if event.button() == Qt.LeftButton and self.zoom_box_start is not None:
            # Complete zoom box
            self.rubber_band.hide()
            zoom_rect = QtCore.QRect(self.zoom_box_start, event.pos()).normalized()
            
            if zoom_rect.width() > 10 and zoom_rect.height() > 10 and self.original_pixmap:
                # Convert screen coordinates to image coordinates
                current_pixmap = self.pixmap()
                if current_pixmap:
                    # Calculate scaling factor from displayed image to original
                    scale_x = self.original_pixmap.width() / current_pixmap.width()
                    scale_y = self.original_pixmap.height() / current_pixmap.height()
                    
                    # Convert zoom box to original image coordinates
                    img_x = int(zoom_rect.x() * scale_x)
                    img_y = int(zoom_rect.y() * scale_y)
                    img_w = int(zoom_rect.width() * scale_x)
                    img_h = int(zoom_rect.height() * scale_y)
                    
                    # Clamp to image bounds
                    img_x = max(0, min(img_x, self.original_pixmap.width() - 1))
                    img_y = max(0, min(img_y, self.original_pixmap.height() - 1))
                    img_w = min(img_w, self.original_pixmap.width() - img_x)
                    img_h = min(img_h, self.original_pixmap.height() - img_y)
                    
                    # Crop the region from original image
                    if img_w > 10 and img_h > 10:
                        self.zoom_region = (img_x, img_y, img_w, img_h)
                        cropped = self.original_pixmap.copy(img_x, img_y, img_w, img_h)
                        self.zoomed_pixmap = cropped
                        self.zoom_factor = 1.0  # Reset zoom factor for the cropped region
                        self.update_display()
                    
            self.zoom_box_start = None
        elif event.button() == Qt.MiddleButton:
            self.is_panning = False
            self.setCursor(Qt.CrossCursor if self.zoom_box_enabled else Qt.ArrowCursor)
            
    def wheelEvent(self, event):
        """Handle mouse wheel for zoom"""
        if event.modifiers() & Qt.ControlModifier:
            # Zoom with Ctrl + wheel
            if event.angleDelta().y() > 0:
                self.zoom_in()
            else:
                self.zoom_out()
            event.accept()
            
    def resizeEvent(self, event):
        """Handle resize to update display at zoom 1.0"""
        super().resizeEvent(event)
        if self.zoom_factor == 1.0:
            self.update_display()


class MicroplasticAnalyzerGUI(QMainWindow):
    """Main GUI window for Microplastic Analyzer"""
    
    def __init__(self):
        super().__init__()
        self.current_image = None
        self.current_result = None
        self.ground_truth = None  # Store ground truth particles
        self.ground_truth_count = 0
        self.esp_camera_ip = "192.168.1.100"  # Default ESP32-CAM IP
        self.yolo_model = None
        self.yolo_model_path = None
        
        # Statistics comparator for multi-method comparison
        self.stats_comparator = StatisticsComparator()
        self.quick_result = None
        self.deep_result = None
        self.ml_result = None
        
        self.initUI()
        
    def initUI(self):
        """Initialize the user interface"""
        self.setWindowTitle('Microplastic Analyzer Pro')
        self.setGeometry(100, 100, 1400, 900)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel - Controls
        left_panel = self.create_control_panel()
        main_layout.addWidget(left_panel, 1)
        
        # Right panel - Display
        right_panel = self.create_display_panel()
        main_layout.addWidget(right_panel, 2)
        
        # Status bar
        self.statusBar().showMessage('Ready')
        
        # Menu bar
        self.create_menu_bar()
        
    def create_menu_bar(self):
        """Create menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        open_action = QtWidgets.QAction('Open Image', self)
        open_action.setShortcut('Ctrl+O')
        open_action.triggered.connect(self.load_image)
        file_menu.addAction(open_action)
        
        save_action = QtWidgets.QAction('Save Results', self)
        save_action.setShortcut('Ctrl+S')
        save_action.triggered.connect(self.save_results)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        exit_action = QtWidgets.QAction('Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Tools menu
        tools_menu = menubar.addMenu('Tools')
        
        generate_action = QtWidgets.QAction('Generate Synthetic Data', self)
        generate_action.triggered.connect(self.generate_synthetic)
        tools_menu.addAction(generate_action)
        
        # Machine Learning menu
        ml_menu = menubar.addMenu('Machine Learning')
        
        load_model_action = QtWidgets.QAction('Load YOLO Model', self)
        load_model_action.triggered.connect(self.load_yolo_model)
        ml_menu.addAction(load_model_action)
        
        run_ml_action = QtWidgets.QAction('Run ML Detection', self)
        run_ml_action.triggered.connect(self.run_yolo_detection)
        ml_menu.addAction(run_ml_action)
        
        # Help menu
        help_menu = menubar.addMenu('Help')
        
        about_action = QtWidgets.QAction('About', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
        # Add keyboard shortcuts for zoom (working on any tab)
        from PyQt5.QtWidgets import QShortcut
        from PyQt5.QtGui import QKeySequence
        
        # Shortcuts for Annotated view (when active)
        zoom_in_shortcut = QShortcut(QKeySequence('+'), self)
        zoom_in_shortcut.activated.connect(lambda: self.zoom_in_annotated() if self.tabs.currentIndex() == 2 else self.zoom_in_image())
        
        zoom_out_shortcut = QShortcut(QKeySequence('-'), self)
        zoom_out_shortcut.activated.connect(lambda: self.zoom_out_annotated() if self.tabs.currentIndex() == 2 else self.zoom_out_image())
        
        reset_zoom_shortcut = QShortcut(QKeySequence('0'), self)
        reset_zoom_shortcut.activated.connect(lambda: self.reset_zoom_annotated() if self.tabs.currentIndex() == 2 else self.reset_zoom_image())
        
    def create_control_panel(self):
        """Create left control panel with comprehensive parameter controls"""
        panel = QWidget()
        main_layout = QVBoxLayout(panel)
        
        # Create scroll area for parameters
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        layout = QVBoxLayout(scroll_content)
        
        # Title
        title = QLabel("Microplastic Analyzer Pro")
        title.setStyleSheet("font-size: 18px; font-weight: bold; padding: 10px;")
        layout.addWidget(title)
        
        # ESP Camera button
        esp_cam_btn = QPushButton("Capture from Camera")
        esp_cam_btn.setStyleSheet("padding: 10px; font-size: 14px; font-weight: bold; background-color: #2196F3;")
        esp_cam_btn.setToolTip("Capture image from USB Camera or ESP32-CAM.\nSupports direct USB connection or network HTTP capture.")
        esp_cam_btn.clicked.connect(self.capture_from_esp_camera)
        layout.addWidget(esp_cam_btn)
        
        # Load Image button
        load_btn = QPushButton("Load Image")
        load_btn.setStyleSheet("padding: 10px; font-size: 14px; font-weight: bold; background-color: #4CAF50;")
        load_btn.clicked.connect(self.load_image)
        layout.addWidget(load_btn)
        
        # Generate Synthetic Image button
        generate_btn = QPushButton("Generate Synthetic Image")
        generate_btn.setStyleSheet("padding: 10px; font-size: 14px; font-weight: bold; background-color: #FF9800;")
        generate_btn.setToolTip("Generate a single synthetic image with automatic ground truth.\n"
                               "Perfect for testing and comparing analysis methods.")
        generate_btn.clicked.connect(self.generate_synthetic_single)
        layout.addWidget(generate_btn)
        
        # Export YOLO Dataset button
        yolo_export_btn = QPushButton("Export YOLO Training Dataset")
        yolo_export_btn.setStyleSheet("padding: 10px; font-size: 14px; font-weight: bold; background-color: #9C27B0; color: white;")
        yolo_export_btn.setToolTip("Generate and export synthetic images in YOLOv8 format.\n"
                                   "Creates images/ and labels/ folders with annotations\n"
                                   "for training custom YOLOv8 detection models.")
        yolo_export_btn.clicked.connect(self.export_yolo_dataset)
        layout.addWidget(yolo_export_btn)
        
        # ===== SYNTHETIC IMAGE GENERATION PARAMETERS =====
        # (Moved here from below, right after Generate button)
        synth_group = QGroupBox("Synthetic Image Parameters")
        synth_layout = QGridLayout()
        
        # Row 1: Basic parameters
        synth_layout.addWidget(QLabel("Num Particles:"), 0, 0)
        self.synth_num_particles = QSpinBox()
        self.synth_num_particles.setRange(1, 100)
        self.synth_num_particles.setValue(50)
        synth_layout.addWidget(self.synth_num_particles, 0, 1)
        
        synth_layout.addWidget(QLabel("Shape Type:"), 0, 2)
        self.synth_shape_type = QComboBox()
        self.synth_shape_type.addItems(['Mixed', 'Fiber/Filament only', 'Fragment only', 
                                       'Microbead/Pellet only', 'Irregular only'])
        synth_layout.addWidget(self.synth_shape_type, 0, 3)
        
        # Row 2: Color and blur
        synth_layout.addWidget(QLabel("Color Type:"), 1, 0)
        self.synth_color_type = QComboBox()
        self.synth_color_type.addItems(['Fluorescent', 'Natural', 'Mixed Colors'])
        synth_layout.addWidget(self.synth_color_type, 1, 1)
        
        self.synth_enable_blur = QCheckBox("Enable Blur")
        self.synth_enable_blur.setChecked(True)
        synth_layout.addWidget(self.synth_enable_blur, 1, 2)
        
        synth_layout.addWidget(QLabel("Blur Kernel:"), 1, 3)
        self.synth_blur_kernel = QSpinBox()
        self.synth_blur_kernel.setRange(1, 31)
        self.synth_blur_kernel.setValue(15)
        self.synth_blur_kernel.setSingleStep(2)
        synth_layout.addWidget(self.synth_blur_kernel, 1, 4)
        
        # Row 3: Background and Particle Brightness
        synth_layout.addWidget(QLabel("BG Brightness:"), 2, 0)
        self.synth_brightness = QSpinBox()
        self.synth_brightness.setRange(0, 255)
        self.synth_brightness.setValue(0)
        self.synth_brightness.setToolTip("Background brightness:\n• 0 = Dark (fluorescent microscopy)\n• 255 = White (brightfield microscopy)")
        synth_layout.addWidget(self.synth_brightness, 2, 1)
        
        synth_layout.addWidget(QLabel("Objects Brightness:"), 2, 2)
        self.synth_particle_brightness = QSpinBox()
        self.synth_particle_brightness.setRange(0, 255)
        self.synth_particle_brightness.setValue(255)
        self.synth_particle_brightness.setToolTip("Particle/object brightness:\n• 255 = Full brightness\n• 128 = Half brightness\n• Lower values = darker particles")
        synth_layout.addWidget(self.synth_particle_brightness, 2, 3)
        
        # Row 4: Image dimensions (compact)
        synth_layout.addWidget(QLabel("Size:"), 3, 0)
        self.synth_width = QSpinBox()
        self.synth_width.setRange(100, 5000)
        self.synth_width.setValue(1280)
        synth_layout.addWidget(self.synth_width, 3, 1)
        synth_layout.addWidget(QLabel("x"), 3, 2)
        self.synth_height = QSpinBox()
        self.synth_height.setRange(100, 5000)
        self.synth_height.setValue(1280)
        synth_layout.addWidget(self.synth_height, 3, 3)
        
        synth_group.setLayout(synth_layout)
        layout.addWidget(synth_group)
        
        # Create hidden parameters with default values (not shown in UI for simplicity)
        self.synth_enable_glow = QCheckBox()
        self.synth_enable_glow.setChecked(True)
        self.synth_glow_intensity = QDoubleSpinBox()
        self.synth_glow_intensity.setValue(0.4)
        self.synth_glow_sigma_min = QDoubleSpinBox()
        self.synth_glow_sigma_min.setValue(2.5)
        self.synth_glow_sigma_max = QDoubleSpinBox()
        self.synth_glow_sigma_max.setValue(6.0)
        self.synth_bg_noise_min = QDoubleSpinBox()
        self.synth_bg_noise_min.setValue(0.0)
        self.synth_bg_noise_max = QDoubleSpinBox()
        self.synth_bg_noise_max.setValue(0.02)
        
        # ===== QUICK ANALYSIS PARAMETERS =====
        quick_group = QGroupBox("Quick Analysis Parameters")
        quick_layout = QGridLayout()
        
        # Row 1: Blur and Distance Threshold
        # Row 0: Preprocess Method
        quick_layout.addWidget(QLabel("Preprocess:"), 0, 0)
        self.quick_method = QComboBox()
        self.quick_method.addItems(['basic'])  # Quick: only basic method
        self.quick_method.setCurrentIndex(0)  # Default to 'basic'
        self.quick_method.setToolTip(
            "Quick Analysis Method:\n"
            "• basic: Simple Otsu thresholding (fastest)"
        )
        quick_layout.addWidget(self.quick_method, 0, 1, 1, 3)
        
        # Row 1: Blur and Distance Threshold
        quick_layout.addWidget(QLabel("Blur:"), 1, 0)
        self.quick_blur = QSpinBox()
        self.quick_blur.setRange(3, 100)
        self.quick_blur.setValue(5)
        quick_layout.addWidget(self.quick_blur, 1, 1)
        
        quick_layout.addWidget(QLabel("Distance Threshold:"), 1, 2)
        self.quick_distance_thresh = QDoubleSpinBox()
        self.quick_distance_thresh.setRange(0.1, 1.0)
        self.quick_distance_thresh.setSingleStep(0.1)
        self.quick_distance_thresh.setValue(0.5)
        quick_layout.addWidget(self.quick_distance_thresh, 1, 3)
        
        # Row 2: Min and Max Area
        quick_layout.addWidget(QLabel("Min Area:"), 2, 0)
        self.quick_min_area = QSpinBox()
        self.quick_min_area.setRange(1, 100000)
        self.quick_min_area.setValue(20)
        quick_layout.addWidget(self.quick_min_area, 2, 1)
        
        quick_layout.addWidget(QLabel("Max Area:"), 2, 2)
        self.quick_max_area = QSpinBox()
        self.quick_max_area.setRange(1, 100000)
        self.quick_max_area.setValue(10000)
        quick_layout.addWidget(self.quick_max_area, 2, 3)
        
        quick_group.setLayout(quick_layout)
        layout.addWidget(quick_group)
        
        # ===== DEEP ANALYSIS PARAMETERS =====
        deep_group = QGroupBox("Deep Analysis Parameters")
        deep_layout = QGridLayout()
        
        # Row 0: Preprocess Method
        deep_layout.addWidget(QLabel("Preprocess:"), 0, 0)
        self.deep_method = QComboBox()
        self.deep_method.addItems(['adaptive', 'advanced', 'watershed'])  # Deep: 3 methods
        self.deep_method.setCurrentIndex(1)  # Default to 'advanced' (auto-detects background)
        self.deep_method.setToolTip(
            "Deep Analysis Methods:\n"
            "• adaptive: CLAHE + local threshold (fluorescent)\n"
            "• advanced: Multi-channel RGB (auto-detects background)\n"
            "• watershed: Separates touching particles"
        )
        deep_layout.addWidget(self.deep_method, 0, 1, 1, 3)
        
        # Row 1: Morphology Kernel and Blur
        deep_layout.addWidget(QLabel("Morphology Kernel:"), 1, 0)
        self.deep_marker_size = QSpinBox()
        self.deep_marker_size.setRange(3, 100)
        self.deep_marker_size.setValue(5)
        self.deep_marker_size.setSingleStep(2)
        self.deep_marker_size.setToolTip(
            "For watershed method: Controls marker erosion\n"
            "• Smaller (2-3): Tighter object boundaries\n"
            "• Larger (4-5): More conservative separation"
        )
        deep_layout.addWidget(self.deep_marker_size, 1, 1)
        
        deep_layout.addWidget(QLabel("Blur:"), 1, 2)
        self.deep_blur = QSpinBox()
        self.deep_blur.setRange(1, 100)
        self.deep_blur.setValue(3)
        deep_layout.addWidget(self.deep_blur, 1, 3)
        
        # Row 2: Distance Threshold and Min Area
        deep_layout.addWidget(QLabel("Distance Threshold:"), 2, 0)
        self.deep_distance_thresh = QDoubleSpinBox()
        self.deep_distance_thresh.setRange(0.1, 1.0)
        self.deep_distance_thresh.setSingleStep(0.1)
        self.deep_distance_thresh.setValue(0.4)
        self.deep_distance_thresh.setToolTip(
            "For watershed method: Controls object separation\n"
            "• Lower (0.3-0.4): More aggressive separation (recommended)\n"
            "• Medium (0.5): Balanced\n"
            "• Higher (0.6-0.7): Conservative separation"
        )
        deep_layout.addWidget(self.deep_distance_thresh, 2, 1)
        
        deep_layout.addWidget(QLabel("Min Area:"), 2, 2)
        self.deep_min_area = QSpinBox()
        self.deep_min_area.setRange(1, 100000)
        self.deep_min_area.setValue(20)
        deep_layout.addWidget(self.deep_min_area, 2, 3)
        
        # Row 3: Max Area
        deep_layout.addWidget(QLabel("Max Area:"), 3, 0)
        self.deep_max_area = QSpinBox()
        self.deep_max_area.setRange(1, 100000)
        self.deep_max_area.setValue(10000)
        deep_layout.addWidget(self.deep_max_area, 3, 1)
        
        # Row 4: Brightness and Sharpness thresholds
        deep_layout.addWidget(QLabel("Min Brightness:"), 4, 0)
        self.brightness_threshold = QSpinBox()
        self.brightness_threshold.setRange(0, 255)
        self.brightness_threshold.setValue(0)
        self.brightness_threshold.setToolTip("Minimum brightness (0=disabled, 20-80 typical)")
        deep_layout.addWidget(self.brightness_threshold, 4, 1)
        
        deep_layout.addWidget(QLabel("Min Sharpness:"), 4, 2)
        self.min_blur_score = QDoubleSpinBox()
        self.min_blur_score.setRange(0.0, 500.0)
        self.min_blur_score.setValue(0.0)
        self.min_blur_score.setSingleStep(10.0)
        self.min_blur_score.setToolTip("Minimum sharpness score (0=disabled, 50-300 typical)")
        deep_layout.addWidget(self.min_blur_score, 4, 3)
        
        # Row 5: Adaptive C value
        deep_layout.addWidget(QLabel("Adaptive C:"), 5, 0)
        self.adaptive_c_value = QSpinBox()
        self.adaptive_c_value.setRange(1, 20)
        self.adaptive_c_value.setValue(2)
        self.adaptive_c_value.setToolTip("C parameter for adaptive thresholding. Lower = more sensitive")
        deep_layout.addWidget(self.adaptive_c_value, 5, 1)
        
        # Store labels for dynamic enable/disable
        self.deep_morph_label = deep_layout.itemAtPosition(1, 0).widget()
        self.deep_blur_label = deep_layout.itemAtPosition(1, 2).widget()
        self.deep_dist_label = deep_layout.itemAtPosition(2, 0).widget()
        self.deep_adaptive_label = deep_layout.itemAtPosition(5, 0).widget()
        
        # Connect preprocessing method change to update parameter states
        self.deep_method.currentTextChanged.connect(self._update_deep_params_state)
        
        deep_group.setLayout(deep_layout)
        layout.addWidget(deep_group)
        
        # Initialize parameter states based on default method
        self._update_deep_params_state(self.deep_method.currentText())
        
        # ===== QUICK SETTINGS BUTTONS =====
        settings_group = QGroupBox("Quick Settings")
        settings_layout = QHBoxLayout()
        
        fluorescent_btn = QPushButton("Fluorescent Microscopy")
        fluorescent_btn.setToolTip("Optimize parameters for fluorescent microscopy (black background)")
        fluorescent_btn.setStyleSheet("padding: 6px; font-weight: bold; background-color: #2196F3; color: white;")
        fluorescent_btn.clicked.connect(self.set_fluorescent_params)
        settings_layout.addWidget(fluorescent_btn)
        
        brightfield_btn = QPushButton("Brightfield Microscopy")
        brightfield_btn.setToolTip("Optimize parameters for brightfield microscopy (white background)")
        brightfield_btn.setStyleSheet("padding: 6px; font-weight: bold; background-color: #2196F3; color: white;")
        brightfield_btn.clicked.connect(self.set_brightfield_params)
        settings_layout.addWidget(brightfield_btn)
        
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)
        
        # ===== ANALYSIS BUTTONS =====
        analysis_group = QGroupBox("Analysis")
        analysis_layout = QVBoxLayout()
        
        self.quick_btn = QPushButton("Quick Analysis")
        self.quick_btn.setEnabled(False)
        self.quick_btn.setStyleSheet("padding: 8px; font-weight: bold; background-color: #4CAF50;")
        self.quick_btn.clicked.connect(self.run_quick_analysis)
        analysis_layout.addWidget(self.quick_btn)
        
        self.deep_btn = QPushButton("Deep Analysis")
        self.deep_btn.setEnabled(False)
        self.deep_btn.setStyleSheet("padding: 8px; font-weight: bold; background-color: #4CAF50;")
        self.deep_btn.clicked.connect(self.run_deep_analysis)
        analysis_layout.addWidget(self.deep_btn)
        
        self.both_btn = QPushButton("Run Both & Compare")
        self.both_btn.setEnabled(False)
        self.both_btn.setStyleSheet("padding: 8px; font-weight: bold; background-color: #4CAF50;")
        self.both_btn.setToolTip("Run Quick and Deep analysis on current single image and compare results.\n"
                                "Use synthetic images for automatic ground truth metrics.")
        self.both_btn.clicked.connect(self.run_both_analyses)
        analysis_layout.addWidget(self.both_btn)
        
        # Benchmark button
        self.benchmark_btn = QPushButton("Run Benchmark")
        self.benchmark_btn.setStyleSheet("padding: 8px; font-weight: bold; background-color: #4CAF50;")
        self.benchmark_btn.setToolTip("Run batch analysis on 200+ images.\n"
                                      "Compares Quick vs Deep across many images.\n"
                                      "Generates HTML report with aggregated statistics.")
        self.benchmark_btn.clicked.connect(self.run_benchmark)
        analysis_layout.addWidget(self.benchmark_btn)
        
        analysis_group.setLayout(analysis_layout)
        layout.addWidget(analysis_group)
        
        # ===== MACHINE LEARNING =====
        if YOLO_AVAILABLE:
            ml_group = QGroupBox("Machine Learning (YOLO)")
            ml_layout = QVBoxLayout()
            
            load_model_btn = QPushButton("Load YOLO Model")
            load_model_btn.setStyleSheet("padding: 8px; font-weight: bold; background-color: #4CAF50;")
            load_model_btn.clicked.connect(self.load_yolo_model)
            ml_layout.addWidget(load_model_btn)
            
            self.run_ml_btn = QPushButton("Run ML Detection")
            self.run_ml_btn.setEnabled(False)
            self.run_ml_btn.setStyleSheet("padding: 8px; font-weight: bold; background-color: #4CAF50;")
            self.run_ml_btn.clicked.connect(self.run_yolo_detection)
            ml_layout.addWidget(self.run_ml_btn)
            
            self.run_ml_benchmark_btn = QPushButton("Run ML Benchmark")
            self.run_ml_benchmark_btn.setEnabled(False)
            self.run_ml_benchmark_btn.setStyleSheet("padding: 8px; font-weight: bold; background-color: #2196F3;")
            self.run_ml_benchmark_btn.setToolTip("Run ML detection with full shape/color analysis.\nComparable with Quick and Deep analysis.")
            self.run_ml_benchmark_btn.clicked.connect(self.run_ml_benchmark)
            ml_layout.addWidget(self.run_ml_benchmark_btn)
            
            self.ml_model_label = QLabel("No model loaded")
            self.ml_model_label.setStyleSheet("color: gray; font-style: italic;")
            ml_layout.addWidget(self.ml_model_label)
            
            ml_group.setLayout(ml_layout)
            layout.addWidget(ml_group)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Results summary
        results_group = QGroupBox("Results Summary")
        results_layout = QVBoxLayout()
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(200)
        results_layout.addWidget(self.results_text)
        
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)
        
        # Export buttons
        export_layout = QHBoxLayout()
        
        export_json_btn = QPushButton("Export JSON")
        export_json_btn.clicked.connect(lambda: self.export_results('json'))
        export_layout.addWidget(export_json_btn)
        
        export_csv_btn = QPushButton("Export CSV")
        export_csv_btn.clicked.connect(lambda: self.export_results('csv'))
        export_layout.addWidget(export_csv_btn)
        
        layout.addLayout(export_layout)
        
        layout.addStretch()
        
        scroll.setWidget(scroll_content)
        main_layout.addWidget(scroll)
        
        return panel
        
    def create_display_panel(self):
        """Create right display panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Tab widget for different views
        self.tabs = QTabWidget()
        
        # Image tab with zoom controls
        image_tab = QWidget()
        image_layout = QVBoxLayout(image_tab)
        
        # Zoom controls toolbar
        zoom_toolbar = QHBoxLayout()
        
        self.zoom_in_btn = QPushButton("Zoom In (+)")
        self.zoom_in_btn.setToolTip("Zoom in by 20% (Shortcut: + key)")
        self.zoom_in_btn.clicked.connect(self.zoom_in_image)
        zoom_toolbar.addWidget(self.zoom_in_btn)
        
        self.zoom_out_btn = QPushButton("Zoom Out (-)")
        self.zoom_out_btn.setToolTip("Zoom out by 20% (Shortcut: - key)")
        self.zoom_out_btn.clicked.connect(self.zoom_out_image)
        zoom_toolbar.addWidget(self.zoom_out_btn)
        
        self.zoom_reset_btn = QPushButton("Reset Zoom")
        self.zoom_reset_btn.setToolTip("Reset to original view (Shortcut: 0 key)")
        self.zoom_reset_btn.clicked.connect(self.reset_zoom_image)
        zoom_toolbar.addWidget(self.zoom_reset_btn)
        
        self.zoom_box_btn = QPushButton("Box Zoom")
        self.zoom_box_btn.setCheckable(True)
        self.zoom_box_btn.setToolTip("Enable box zoom: Drag a rectangle to zoom into that area.\nRight-click to zoom out.\nCtrl+Wheel to zoom incrementally.")
        self.zoom_box_btn.toggled.connect(self.toggle_zoom_box)
        zoom_toolbar.addWidget(self.zoom_box_btn)
        
        self.zoom_level_label = QLabel("Zoom: 100%")
        zoom_toolbar.addWidget(self.zoom_level_label)
        
        zoom_toolbar.addStretch()
        
        image_layout.addLayout(zoom_toolbar)
        
        # Scroll area for image
        self.image_scroll_area = QScrollArea()
        self.image_scroll_area.setWidgetResizable(False)
        self.image_scroll_area.setAlignment(Qt.AlignCenter)
        
        self.image_label = ImageLabel()
        self.image_label.set_scroll_area(self.image_scroll_area)
        self.image_scroll_area.setWidget(self.image_label)
        image_layout.addWidget(self.image_scroll_area)
        
        self.tabs.addTab(image_tab, "Original Image")
        
        # Results tab
        results_tab = QWidget()
        results_layout = QVBoxLayout(results_tab)
        
        self.mask_label = ImageLabel()
        results_layout.addWidget(self.mask_label)
        
        self.tabs.addTab(results_tab, "Segmentation Mask")
        
        # Annotated tab with zoom controls
        annotated_tab = QWidget()
        annotated_layout = QVBoxLayout(annotated_tab)
        
        # Zoom controls toolbar for annotated view
        annotated_zoom_toolbar = QHBoxLayout()
        
        self.annotated_zoom_in_btn = QPushButton("Zoom In (+)")
        self.annotated_zoom_in_btn.setToolTip("Zoom in by 20% (Shortcut: + key)")
        self.annotated_zoom_in_btn.clicked.connect(self.zoom_in_annotated)
        annotated_zoom_toolbar.addWidget(self.annotated_zoom_in_btn)
        
        self.annotated_zoom_out_btn = QPushButton("Zoom Out (-)")
        self.annotated_zoom_out_btn.setToolTip("Zoom out by 20% (Shortcut: - key)")
        self.annotated_zoom_out_btn.clicked.connect(self.zoom_out_annotated)
        annotated_zoom_toolbar.addWidget(self.annotated_zoom_out_btn)
        
        self.annotated_zoom_reset_btn = QPushButton("Reset Zoom")
        self.annotated_zoom_reset_btn.setToolTip("Reset to full annotated view (Shortcut: 0 key)")
        self.annotated_zoom_reset_btn.clicked.connect(self.reset_zoom_annotated)
        annotated_zoom_toolbar.addWidget(self.annotated_zoom_reset_btn)
        
        self.annotated_zoom_box_btn = QPushButton("Box Zoom")
        self.annotated_zoom_box_btn.setCheckable(True)
        self.annotated_zoom_box_btn.setToolTip("Enable box zoom:\n• Left-click and drag to select area to zoom\n• Right-click to zoom out\n• Ctrl+Wheel to zoom incrementally\n• Press '0' to reset zoom")
        self.annotated_zoom_box_btn.toggled.connect(self.toggle_zoom_box_annotated)
        annotated_zoom_toolbar.addWidget(self.annotated_zoom_box_btn)
        
        self.annotated_zoom_level_label = QLabel("Zoom: 100%")
        annotated_zoom_toolbar.addWidget(self.annotated_zoom_level_label)
        
        annotated_zoom_toolbar.addStretch()
        
        annotated_layout.addLayout(annotated_zoom_toolbar)
        
        # Scroll area for annotated image
        self.annotated_scroll_area = QScrollArea()
        self.annotated_scroll_area.setWidgetResizable(False)
        self.annotated_scroll_area.setAlignment(Qt.AlignCenter)
        
        self.annotated_label = ImageLabel()
        self.annotated_label.set_scroll_area(self.annotated_scroll_area)
        self.annotated_scroll_area.setWidget(self.annotated_label)
        annotated_layout.addWidget(self.annotated_scroll_area)
        
        self.tabs.addTab(annotated_tab, "Annotated Results")
        
        # Table tab
        table_tab = QWidget()
        table_layout = QVBoxLayout(table_tab)
        
        self.results_table = QTableWidget()
        table_layout.addWidget(self.results_table)
        
        self.tabs.addTab(table_tab, "Data Table")
        
        # Statistics tab with multiple plots
        stats_tab = QWidget()
        stats_layout = QVBoxLayout(stats_tab)
        
        # Chart control buttons
        chart_controls = QHBoxLayout()
        
        self.refresh_charts_btn = QPushButton("Refresh Charts")
        self.refresh_charts_btn.clicked.connect(self.update_charts_full)
        chart_controls.addWidget(self.refresh_charts_btn)
        
        self.chart_type_combo = QComboBox()
        self.chart_type_combo.addItems(['All Charts', 'Shape Distribution', 'Size Distribution', 
                                        'Color Distribution', 'Circularity Distribution', 'Scatter Plots'])
        self.chart_type_combo.setCurrentIndex(0)  # Default to 'All Charts'
        self.chart_type_combo.currentTextChanged.connect(self.update_charts_full)
        chart_controls.addWidget(QLabel("Chart Type:"))
        chart_controls.addWidget(self.chart_type_combo)
        
        chart_controls.addStretch()
        stats_layout.addLayout(chart_controls)
        
        # Create matplotlib figure with subplots
        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)
        stats_layout.addWidget(self.canvas)
        
        self.tabs.addTab(stats_tab, "Statistics")
        
        layout.addWidget(self.tabs)
        
        return panel
        
    def load_image(self):
        """Load image file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", 
            "Image Files (*.png *.jpg *.jpeg *.tiff *.bmp);;All Files (*)"
        )
        
        if file_path:
            try:
                # Load image
                image = cv2.imread(file_path)
                if image is None:
                    raise ValueError("Failed to load image")
                
                # Convert to RGB
                self.current_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Display
                self.image_label.set_image(self.current_image)
                
                # Clear ground truth when loading real image
                self.ground_truth = None
                self.ground_truth_count = 0
                
                # Enable analysis buttons
                self.quick_btn.setEnabled(True)
                self.deep_btn.setEnabled(True)
                self.both_btn.setEnabled(True)
                
                # Auto-detect fluorescent microscopy (black background)
                gray = cv2.cvtColor(self.current_image, cv2.COLOR_RGB2GRAY)
                mean_brightness = np.mean(gray)
                
                if mean_brightness < 50:  # Very dark image - likely fluorescent microscopy
                    self.statusBar().showMessage(
                        f'Loaded: {Path(file_path).name} - Dark background detected (fluorescent microscopy)'
                    )
                    # Show tip for better detection
                    QMessageBox.information(
                        self,
                        "Fluorescent Microscopy Detected",
                        "Dark background image detected!\n\n"
                        "For best results with fluorescent microscopy:\n"
                        "1. Lower the 'Min Area' to 5-20\n"
                        "2. Try 'adaptive' preprocessing method\n"
                        "3. Adjust 'Blur' as needed (default: 15)\n\n"
                        "Current settings may miss small particles."
                    )
                else:
                    self.statusBar().showMessage(f'Loaded: {Path(file_path).name}')
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load image: {str(e)}")
    
    def _update_deep_params_state(self, method):
        """Enable/disable deep analysis parameters based on selected preprocessing method"""
        # All methods use min_area and max_area - always enabled
        
        # Watershed uses: blur, distance_thresh, marker_size
        is_watershed = (method == 'watershed')
        self.deep_blur.setEnabled(is_watershed)
        self.deep_blur_label.setEnabled(is_watershed)
        self.deep_distance_thresh.setEnabled(is_watershed)
        self.deep_dist_label.setEnabled(is_watershed)
        self.deep_marker_size.setEnabled(is_watershed)
        self.deep_morph_label.setEnabled(is_watershed)
        
        # Adaptive uses: adaptive_c_value
        is_adaptive = (method == 'adaptive')
        self.adaptive_c_value.setEnabled(is_adaptive)
        self.deep_adaptive_label.setEnabled(is_adaptive)
        
        # Update tooltips to show when parameters are inactive
        if not is_watershed:
            self.deep_blur.setToolTip("Only used by 'watershed' preprocessing")
            self.deep_distance_thresh.setToolTip("Only used by 'watershed' preprocessing")
            self.deep_marker_size.setToolTip("Only used by 'watershed' preprocessing")
        else:
            self.deep_blur.setToolTip("Gaussian blur kernel size")
            self.deep_distance_thresh.setToolTip(
                "For watershed method: Controls object separation\n"
                "• Lower (0.3-0.4): More aggressive separation\n"
                "• Medium (0.5): Balanced (default)\n"
                "• Higher (0.6-0.7): Conservative separation"
            )
            self.deep_marker_size.setToolTip(
                "For watershed method: Controls marker erosion\n"
                "• Smaller (2-3): Tighter object boundaries\n"
                "• Larger (4-5): More conservative separation"
            )
        
        if not is_adaptive:
            self.adaptive_c_value.setToolTip("Only used by 'adaptive' preprocessing")
        else:
            self.adaptive_c_value.setToolTip("C parameter for adaptive thresholding. Lower = more sensitive")
    
    def capture_from_esp_camera(self):
        """Capture image from ESP32-CAM via HTTP or USB Camera"""
        from PyQt5.QtWidgets import QInputDialog
        import urllib.request
        import urllib.error
        
        # Ask user to choose camera source
        items = ["USB Camera (Direct)", "ESP32-CAM (Network)"]
        source, ok = QInputDialog.getItem(
            self,
            "Camera Source",
            "Select camera source:",
            items, 0, False
        )
        
        if not ok:
            return
        
        if "USB" in source:
            # USB Camera Mode
            self._capture_from_usb_camera()
            return
        
        # ESP32-CAM Network Mode
        # Ask for ESP32-CAM IP address
        ip, ok = QInputDialog.getText(
            self,
            "ESP32-CAM Configuration",
            f"Enter ESP32-CAM IP address:",
            text=self.esp_camera_ip
        )
        
        if not ok or not ip:
            return
        
        self.esp_camera_ip = ip
        
        try:
            # Construct ESP32-CAM capture URL
            url = f"http://{self.esp_camera_ip}/capture"
            
            self.statusBar().showMessage(f"Connecting to ESP32-CAM at {self.esp_camera_ip}...")
            QApplication.processEvents()
            
            # Request image from ESP32-CAM
            with urllib.request.urlopen(url, timeout=5) as response:
                img_array = np.asarray(bytearray(response.read()), dtype=np.uint8)
                image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("Failed to decode image from ESP32-CAM")
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Store and display
            self.current_image = image
            self.ground_truth = None
            self.ground_truth_count = 0
            
            # Display image using custom ImageLabel method
            self.image_label.set_image(image)
            
            # Enable analysis buttons
            self.quick_btn.setEnabled(True)
            self.deep_btn.setEnabled(True)
            
            h, w, c = image.shape
            self.statusBar().showMessage(f'Captured from ESP32-CAM ({w}x{h})')
            self.results_text.clear()
            self.results_text.append(f"✓ Image captured from ESP32-CAM at {self.esp_camera_ip}")
            self.results_text.append(f"  Resolution: {w}x{h}")
            
        except urllib.error.URLError as e:
            QMessageBox.critical(
                self,
                "Connection Error",
                f"Failed to connect to ESP32-CAM at {self.esp_camera_ip}\\n\\n"
                f"Error: {str(e)}\\n\\n"
                f"Make sure:\\n"
                f"1. ESP32-CAM is powered on\\n"
                f"2. Connected to the same network\\n"
                f"3. IP address is correct\\n"
                f"4. ESP32-CAM firmware supports /capture endpoint"
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to capture image: {str(e)}")
    
    def _capture_from_usb_camera(self):
        """Open camera preview window with capture button"""
        from PyQt5.QtWidgets import QInputDialog, QDialog, QVBoxLayout, QPushButton, QLabel
        from PyQt5.QtCore import QTimer
        
        # Detect available cameras
        self.statusBar().showMessage("Detecting available cameras...")
        QApplication.processEvents()
        
        # Suppress OpenCV error messages during camera detection
        import os
        old_stderr = os.dup(2)  # Save stderr file descriptor
        os.close(2)  # Close stderr temporarily
        
        available_cameras = []
        backend = _get_camera_backend()
        for i in range(3):  # Check first 3 camera indices (faster)
            try:
                cap = cv2.VideoCapture(i, backend)
                if cap.isOpened():
                    # Quick check without reading full frame
                    ret = cap.grab()
                    if ret:
                        available_cameras.append(i)
                    cap.release()
            except:
                pass
        
        # Restore stderr
        os.dup2(old_stderr, 2)
        os.close(old_stderr)
        
        if not available_cameras:
            QMessageBox.warning(
                self,
                "No Cameras Found",
                "No cameras detected.\n\n"
                "Make sure:\n"
                "• Camera is connected\n"
                "• Camera drivers are installed\n"
                "• Camera is not being used by another application"
            )
            return
        
        # Create camera selection dialog
        camera_names = []
        for idx in available_cameras:
            if idx == 0:
                camera_names.append(f"Camera {idx} (Built-in/Default)")
            else:
                camera_names.append(f"Camera {idx} (USB/External)")
        
        camera_choice, ok = QInputDialog.getItem(
            self,
            "Select Camera",
            f"Found {len(available_cameras)} camera(s). Select one:",
            camera_names,
            0,
            False
        )
        
        if not ok:
            return
        
        # Extract camera index from selection
        camera_id = available_cameras[camera_names.index(camera_choice)]
        
        try:
            # Open camera with platform-specific backend
            backend = _get_camera_backend()
            cap = cv2.VideoCapture(camera_id, backend)
            
            if not cap.isOpened():
                raise ValueError(f"Cannot open camera {camera_id}")
            
            # Set camera resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            # Flush camera buffer - grab just 2 frames to ensure fresh feed
            cap.grab()
            cap.grab()
            
            # Create camera preview dialog
            dialog = QDialog(self)
            dialog.setWindowTitle(f"USB Camera {camera_id} - Live Preview")
            dialog.setModal(True)
            dialog.resize(800, 650)
            
            layout = QVBoxLayout()
            
            # Video preview label
            video_label = QLabel()
            video_label.setMinimumSize(640, 480)
            video_label.setStyleSheet("border: 2px solid #333; background-color: black;")
            layout.addWidget(video_label)
            
            # Capture button
            capture_btn = QPushButton("📷 Capture Image")
            capture_btn.setStyleSheet("padding: 10px; font-size: 14px; font-weight: bold; background-color: #4CAF50;")
            layout.addWidget(capture_btn)
            
            # Close button
            close_btn = QPushButton("Close Camera")
            close_btn.setStyleSheet("padding: 8px; font-size: 12px;")
            layout.addWidget(close_btn)
            
            dialog.setLayout(layout)
            
            # Variables to store captured image
            captured_image = [None]  # Use list to allow modification in nested function
            
            def update_frame():
                """Update video preview"""
                ret, frame = cap.read()
                if ret:
                    # Convert BGR to RGB for display
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, c = frame_rgb.shape
                    bytes_per_line = c * w
                    # Make a copy to ensure data is not overwritten
                    qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
                    pixmap = QPixmap.fromImage(qt_image)
                    # Scale to fit label while maintaining aspect ratio
                    scaled_pixmap = pixmap.scaled(video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    video_label.setPixmap(scaled_pixmap)
            
            def capture_image():
                """Capture current frame"""
                ret, frame = cap.read()
                if ret:
                    # Make a copy to ensure the image is not overwritten
                    captured_image[0] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).copy()
                    dialog.accept()
            
            def close_camera():
                """Close camera and dialog"""
                dialog.reject()
            
            # Connect buttons
            capture_btn.clicked.connect(capture_image)
            close_btn.clicked.connect(close_camera)
            
            # Setup timer for video preview
            timer = QTimer()
            timer.timeout.connect(update_frame)
            timer.start(30)  # Update every 30ms (~33 fps)
            
            # Show dialog and wait for user action
            result = dialog.exec_()
            
            # Stop timer and release camera
            timer.stop()
            cap.release()
            
            # If image was captured, load it
            if result == QDialog.Accepted and captured_image[0] is not None:
                image = captured_image[0]
                
                # Store and display
                self.current_image = image
                self.ground_truth = None
                self.ground_truth_count = 0
                
                # Display image using custom ImageLabel method
                self.image_label.set_image(image)
                
                # Enable analysis buttons
                self.quick_btn.setEnabled(True)
                self.deep_btn.setEnabled(True)
                
                h, w, c = image.shape
                self.statusBar().showMessage(f'Captured from USB Camera {camera_id} ({w}x{h})')
                self.results_text.clear()
                self.results_text.append(f"✓ Image captured from USB Camera {camera_id}")
                self.results_text.append(f"  Resolution: {w}x{h}")
            
        except Exception as e:
            # Build platform-specific error message
            system = platform.system()
            if system == 'Darwin':
                platform_help = (
                    f"macOS-specific checks:\\n"
                    f"1. Check System Preferences → Security & Privacy → Camera\\n"
                    f"   to allow terminal/IDE access\\n"
                    f"2. Restart the application after granting permissions\\n"
                    f"3. Ensure camera is not in use by another app\\n"
                )
            elif system == 'Windows':
                platform_help = (
                    f"Windows-specific checks:\\n"
                    f"1. Update camera drivers\\n"
                    f"2. Disable conflicting applications (Zoom, Skype, etc.)\\n"
                    f"3. Check Device Manager for camera errors\\n"
                )
            else:  # Linux
                platform_help = (
                    f"Linux-specific checks:\\n"
                    f"1. Check device permissions: ls -la /dev/video*\\n"
                    f"2. Add user to video group: sudo usermod -a -G video $USER\\n"
                    f"3. Install v4l-utils: sudo apt-get install v4l-utils\\n"
                    f"4. Log out and back in for group changes to take effect\\n"
                )
            
            QMessageBox.critical(
                self,
                "Camera Error",
                f"Failed to open USB camera {camera_id}\\n\\n"
                f"Error: {str(e)}\\n\\n"
                f"Make sure:\\n"
                f"• USB camera is connected\\n"
                f"• Camera is not in use by another application\\n\\n"
                f"{platform_help}"
                f"\\n4. Try a different camera index (0, 1, 2...)"
            )
                
    def get_analysis_params(self, analysis_type='quick'):
        """Get current analysis parameters based on type"""
        if analysis_type == 'quick':
            return PreprocessingParams(
                method=self.quick_method.currentText(),
                min_area=self.quick_min_area.value(),
                max_area=self.quick_max_area.value(),
                blur=self.quick_blur.value(),
                distance_thresh=self.quick_distance_thresh.value()
            )
        else:  # deep
            return PreprocessingParams(
                method=self.deep_method.currentText(),
                min_area=self.deep_min_area.value(),
                max_area=self.deep_max_area.value(),
                blur=self.deep_blur.value(),
                distance_thresh=self.deep_distance_thresh.value(),
                marker_size=self.deep_marker_size.value(),
                brightness_threshold=self.brightness_threshold.value(),
                min_blur_score=self.min_blur_score.value(),
                adaptive_c_value=self.adaptive_c_value.value()
            )
        
    def run_quick_analysis(self):
        """Run quick analysis"""
        if self.current_image is None:
            return
            
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.statusBar().showMessage('Running quick analysis...')
        
        params = self.get_analysis_params('quick')
        
        self.analysis_thread = AnalysisThread(self.current_image, params, 'quick')
        self.analysis_thread.finished.connect(self.on_analysis_complete)
        self.analysis_thread.progress.connect(self.statusBar().showMessage)
        self.analysis_thread.start()
        
    def run_deep_analysis(self):
        """Run deep analysis"""
        if self.current_image is None:
            return
            
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.statusBar().showMessage('Running deep analysis...')
        
        params = self.get_analysis_params('deep')
        
        self.analysis_thread = AnalysisThread(self.current_image, params, 'deep')
        self.analysis_thread.finished.connect(self.on_analysis_complete)
        self.analysis_thread.progress.connect(self.statusBar().showMessage)
        self.analysis_thread.start()
        
    def run_both_analyses(self):
        """Run both Quick and Deep analyses and compare results"""
        if self.current_image is None:
            return
        
        # Check if ground truth is set
        if self.ground_truth_count == 0:
            from PyQt5.QtWidgets import QMessageBox
            reply = QMessageBox.warning(
                self, 
                "No Ground Truth Set",
                "⚠️ No ground truth is set for this image.\n\n"
                "Without ground truth, Precision, Recall, and F1-Score will be 0.00.\n\n"
                "Options:\n"
                "• Use 'Generate Synthetic Image' for automatic ground truth\n\n"
                "Note: For batch analysis of 200 images, use the 'Benchmark' button instead.\n\n"
                "Continue anyway?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.No:
                return
            
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.statusBar().showMessage('Running both analyses...')
        self.results_text.clear()
        self.results_text.append("Running both Quick and Deep analyses...")
        
        import time
        start_time = time.time()
        
        # Run Quick Analysis
        quick_params = self.get_analysis_params('quick')
        quick_analyzer = QuickAnalyzer()
        quick_result = quick_analyzer.analyze(self.current_image, quick_params)
        
        # Run Deep Analysis
        deep_params = self.get_analysis_params('deep')
        deep_analyzer = DeepAnalyzer()
        deep_result = deep_analyzer.analyze(self.current_image, deep_params)
        
        total_time = time.time() - start_time
        
        # Store both results
        self.quick_result = quick_result
        self.deep_result = deep_result
        self.current_result = deep_result  # Use deep as primary
        
        # Create comparison visualization
        self.create_comparison_visualization(quick_result, deep_result)
        
        # Compare and display results with statistics
        self.compare_analysis_results(quick_result, deep_result, total_time)
        
        # Create comprehensive statistics chart comparing both methods
        self.create_comparison_statistics_chart(quick_result, deep_result)
        
        # Update table with deep results
        self.update_results_table(deep_result)
        
        self.progress_bar.setVisible(False)
        self.statusBar().showMessage(f'Both analyses complete in {total_time:.2f}s')
        
    def create_comparison_visualization(self, quick_result, deep_result):
        """Create visualization comparing Quick and Deep analysis"""
        comparison_image = self.current_image.copy()
        
        # Draw Quick analysis boundaries in blue
        for feature in quick_result.features:
            bbox = feature.get('bounding_box')
            if bbox:
                x, y, w, h = bbox
                cv2.rectangle(comparison_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
                centroid = feature.get('centroid', (x + w//2, y + h//2))
                cv2.circle(comparison_image, (int(centroid[0]), int(centroid[1])), 3, (255, 0, 0), -1)
        
        # Draw Deep analysis boundaries in green
        for feature in deep_result.features:
            bbox = feature.get('bounding_box')
            if bbox:
                x, y, w, h = bbox
                cv2.rectangle(comparison_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                centroid = feature.get('centroid', (x + w//2, y + h//2))
                cv2.circle(comparison_image, (int(centroid[0]), int(centroid[1])), 3, (0, 255, 0), -1)
        
        # Add legend
        cv2.putText(comparison_image, "Quick Analysis (Blue)", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(comparison_image, "Deep Analysis (Green)", (10, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Display in annotated tab
        self.annotated_label.set_image(comparison_image)
        self.tabs.setCurrentIndex(2)  # Switch to annotated tab
        
    def compare_analysis_results(self, quick_result, deep_result, total_time):
        """Compare Quick and Deep analysis results"""
        import math
        
        quick_count = quick_result.num_detections
        deep_count = deep_result.num_detections
        
        text = "\n=== COMPARISON RESULTS ===\n"
        
        # Ground Truth Comparison (if available)
        if self.ground_truth_count > 0:
            text += f"\nGROUND TRUTH: {self.ground_truth_count} particles\n"
            text += "-" * 50 + "\n"
            
            # Quick vs Ground Truth
            quick_precision = min(quick_count, self.ground_truth_count) / quick_count if quick_count > 0 else 0
            quick_recall = min(quick_count, self.ground_truth_count) / self.ground_truth_count
            quick_f1 = 2 * (quick_precision * quick_recall) / (quick_precision + quick_recall) if (quick_precision + quick_recall) > 0 else 0
            
            text += f"\nQuick Analysis vs Ground Truth:\n"
            text += f"  Detected: {quick_count} | True: {self.ground_truth_count}\n"
            text += f"  Precision: {quick_precision:.3f} ({quick_precision*100:.1f}%)\n"
            text += f"  Recall: {quick_recall:.3f} ({quick_recall*100:.1f}%)\n"
            text += f"  F1-Score: {quick_f1:.3f} ({quick_f1*100:.1f}%)\n"
            
            # Deep vs Ground Truth
            deep_precision = min(deep_count, self.ground_truth_count) / deep_count if deep_count > 0 else 0
            deep_recall = min(deep_count, self.ground_truth_count) / self.ground_truth_count
            deep_f1 = 2 * (deep_precision * deep_recall) / (deep_precision + deep_recall) if (deep_precision + deep_recall) > 0 else 0
            
            text += f"\nDeep Analysis vs Ground Truth:\n"
            text += f"  Detected: {deep_count} | True: {self.ground_truth_count}\n"
            text += f"  Precision: {deep_precision:.3f} ({deep_precision*100:.1f}%)\n"
            text += f"  Recall: {deep_recall:.3f} ({deep_recall*100:.1f}%)\n"
            text += f"  F1-Score: {deep_f1:.3f} ({deep_f1*100:.1f}%)\n"
        else:
            text += "\n" + "=" * 50 + "\n"
            text += "⚠️  NO GROUND TRUTH SET\n"
            text += "=" * 50 + "\n"
            text += "To enable precision/recall metrics:\n"
            text += "  • Use 'Generate Synthetic Image' (includes ground truth)\n"
            text += "\nWithout ground truth, you can still compare:\n"
            text += "  • Detection counts between methods\n"
            text += "  • Shape/color distributions\n"
            text += "  • Processing times\n"
            text += "=" * 50 + "\n"
            
            text += "\n" + "=" * 50 + "\n"
        
        text += f"Quick Analysis: {quick_count} particles in {quick_result.processing_time:.2f}s\n"
        text += f"Deep Analysis: {deep_count} particles in {deep_result.processing_time:.2f}s\n"
        text += f"Total time: {total_time:.2f} seconds\n"
        
        # Find common detections based on centroid proximity
        common_detections = 0
        quick_only = 0
        deep_only = 0
        threshold_distance = 15  # pixels
        
        matched_deep = set()
        
        for q_feature in quick_result.features:
            q_centroid = q_feature.get('centroid', (0, 0))
            matched = False
            
            for i, d_feature in enumerate(deep_result.features):
                if i in matched_deep:
                    continue
                d_centroid = d_feature.get('centroid', (0, 0))
                distance = math.sqrt((q_centroid[0] - d_centroid[0])**2 +
                                    (q_centroid[1] - d_centroid[1])**2)
                
                if distance < threshold_distance:
                    matched = True
                    matched_deep.add(i)
                    break
            
            if matched:
                common_detections += 1
            else:
                quick_only += 1
        
        deep_only = deep_count - common_detections
        
        text += f"\nDetection Comparison:\n"
        text += f"  Common detections: {common_detections}\n"
        text += f"  Quick only: {quick_only}\n"
        text += f"  Deep only: {deep_only}\n"
        
        # Calculate agreement percentages
        if quick_count > 0 and deep_count > 0:
            agreement_percentage = (common_detections / min(quick_count, deep_count)) * 100
            text += f"  Detection agreement: {agreement_percentage:.1f}%\n"
        
        # Compare shape classifications for common detections
        if common_detections > 0:
            shape_agreement = 0
            color_agreement = 0
            
            matched_deep_for_class = set()
            
            for q_feature in quick_result.features:
                q_centroid = q_feature.get('centroid', (0, 0))
                q_shape = q_feature.get('shape', '')
                q_color = q_feature.get('color', '')
                
                for i, d_feature in enumerate(deep_result.features):
                    if i in matched_deep_for_class:
                        continue
                    d_centroid = d_feature.get('centroid', (0, 0))
                    distance = math.sqrt((q_centroid[0] - d_centroid[0])**2 +
                                        (q_centroid[1] - d_centroid[1])**2)
                    
                    if distance < threshold_distance:
                        d_shape = d_feature.get('shape', '')
                        d_color = d_feature.get('color', '')
                        
                        if q_shape == d_shape:
                            shape_agreement += 1
                        if q_color == d_color:
                            color_agreement += 1
                        matched_deep_for_class.add(i)
                        break
            
            shape_agreement_pct = (shape_agreement / common_detections) * 100
            color_agreement_pct = (color_agreement / common_detections) * 100
            
            text += f"\nClassification Agreement:\n"
            text += f"  Shape agreement: {shape_agreement_pct:.1f}%\n"
            text += f"  Color agreement: {color_agreement_pct:.1f}%\n"
        
        self.results_text.append(text)
    
    def create_comparison_statistics_chart(self, quick_result, deep_result):
        """Create comprehensive comparison statistics chart in Statistics tab"""
        self.figure.clear()
        
        from collections import Counter
        import numpy as np
        
        # Calculate metrics for both methods
        quick_count = quick_result.num_detections
        deep_count = deep_result.num_detections
        gt_count = self.ground_truth_count
        
        # Create 3x2 subplot layout (6 panels)
        ax1 = self.figure.add_subplot(3, 2, 1)  # Detection counts
        ax2 = self.figure.add_subplot(3, 2, 2)  # Metrics (Precision/Recall/F1)
        ax3 = self.figure.add_subplot(3, 2, 3)  # Shape distribution comparison
        ax4 = self.figure.add_subplot(3, 2, 4)  # Color distribution comparison
        ax5 = self.figure.add_subplot(3, 2, 5)  # Area (pixel) distribution
        ax6 = self.figure.add_subplot(3, 2, 6)  # Processing time
        
        # 1. Detection Count Comparison
        if self.ground_truth and len(self.ground_truth) > 0:
            # 3-way comparison with ground truth
            methods = ['Quick', 'Deep', 'Ground\nTruth']
            counts = [quick_count, deep_count, len(self.ground_truth)]
            colors_bar = ['#2196F3', '#4CAF50', '#FF5722']
        else:
            # 2-way comparison
            methods = ['Quick', 'Deep']
            counts = [quick_count, deep_count]
            colors_bar = ['#2196F3', '#4CAF50']
        
        bars = ax1.bar(methods, counts, color=colors_bar, alpha=0.8, edgecolor='black', linewidth=2)
        
        ax1.set_ylabel('Particle Count', fontsize=11, fontweight='bold')
        ax1.set_title('Detection Count Comparison', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (bar, count) in enumerate(zip(bars, counts)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(counts) * 0.02,
                    f'{int(count)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 2. Performance Metrics (if ground truth available)
        if gt_count > 0:
            # Calculate metrics
            quick_precision = min(quick_count, gt_count) / quick_count if quick_count > 0 else 0
            quick_recall = min(quick_count, gt_count) / gt_count if gt_count > 0 else 0
            quick_f1 = 2 * (quick_precision * quick_recall) / (quick_precision + quick_recall) if (quick_precision + quick_recall) > 0 else 0
            
            deep_precision = min(deep_count, gt_count) / deep_count if deep_count > 0 else 0
            deep_recall = min(deep_count, gt_count) / gt_count if gt_count > 0 else 0
            deep_f1 = 2 * (deep_precision * deep_recall) / (deep_precision + deep_recall) if (deep_precision + deep_recall) > 0 else 0
            
            x = np.arange(3)
            width = 0.35
            
            quick_metrics = [quick_precision, quick_recall, quick_f1]
            deep_metrics = [deep_precision, deep_recall, deep_f1]
            
            bars1 = ax2.bar(x - width/2, quick_metrics, width, label='Quick', color='#2196F3', alpha=0.8)
            bars2 = ax2.bar(x + width/2, deep_metrics, width, label='Deep', color='#4CAF50', alpha=0.8)
            
            ax2.set_ylabel('Score', fontsize=11, fontweight='bold')
            ax2.set_title('Performance Metrics vs Ground Truth', fontsize=12, fontweight='bold')
            ax2.set_xticks(x)
            ax2.set_xticklabels(['Precision', 'Recall', 'F1-Score'], fontsize=9)
            ax2.legend(fontsize=9)
            ax2.grid(True, alpha=0.3, axis='y')
            ax2.set_ylim(0, 1.1)
            
            # Add value labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                            f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        else:
            ax2.text(0.5, 0.5, 'No Ground Truth\nAvailable', 
                    ha='center', va='center', fontsize=12, transform=ax2.transAxes)
            ax2.set_title('Performance Metrics', fontsize=12, fontweight='bold')
        
        # 3. Shape Distribution Comparison
        quick_shapes = Counter([f['shape'] for f in quick_result.features]) if quick_result.features else {}
        deep_shapes = Counter([f['shape'] for f in deep_result.features]) if deep_result.features else {}
        
        # Add ground truth shapes if available
        if self.ground_truth and len(self.ground_truth) > 0:
            # Map ground truth shapes to grouped names
            gt_shapes = Counter([map_to_grouped_shape(p['shape']) for p in self.ground_truth])
            all_shapes = sorted(set(list(quick_shapes.keys()) + list(deep_shapes.keys()) + list(gt_shapes.keys())))
        else:
            gt_shapes = None
            all_shapes = sorted(set(list(quick_shapes.keys()) + list(deep_shapes.keys())))
        
        if all_shapes:
            x = np.arange(len(all_shapes))
            width = 0.25 if gt_shapes else 0.35
            
            quick_counts = [quick_shapes.get(shape, 0) for shape in all_shapes]
            deep_counts = [deep_shapes.get(shape, 0) for shape in all_shapes]
            
            if gt_shapes:
                # 3-way comparison
                gt_counts = [gt_shapes.get(shape, 0) for shape in all_shapes]
                ax3.bar(x - width, quick_counts, width, label='Quick', color='#2196F3', alpha=0.8)
                ax3.bar(x, deep_counts, width, label='Deep', color='#4CAF50', alpha=0.8)
                ax3.bar(x + width, gt_counts, width, label='Ground Truth', color='#FF5722', alpha=0.8)
            else:
                # 2-way comparison
                ax3.bar(x - width/2, quick_counts, width, label='Quick', color='#2196F3', alpha=0.8)
                ax3.bar(x + width/2, deep_counts, width, label='Deep', color='#4CAF50', alpha=0.8)
            
            ax3.set_ylabel('Count', fontsize=11, fontweight='bold')
            ax3.set_title('Shape Distribution Comparison', fontsize=12, fontweight='bold')
            ax3.set_xticks(x)
            ax3.set_xticklabels(all_shapes, rotation=45, ha='right', fontsize=9)
            ax3.legend(fontsize=9)
            ax3.grid(True, alpha=0.3, axis='y')
        else:
            ax3.text(0.5, 0.5, 'No Particles\nDetected', 
                    ha='center', va='center', fontsize=12, transform=ax3.transAxes)
            ax3.set_title('Shape Distribution', fontsize=12, fontweight='bold')
        
        # 4. Color Distribution Comparison
        quick_colors = Counter([f['color'] for f in quick_result.features]) if quick_result.features else {}
        deep_colors = Counter([f['color'] for f in deep_result.features]) if deep_result.features else {}
        
        # Add ground truth colors if available
        if self.ground_truth and len(self.ground_truth) > 0:
            gt_colors = Counter([p['color_label'] for p in self.ground_truth])
            all_colors = sorted(set(list(quick_colors.keys()) + list(deep_colors.keys()) + list(gt_colors.keys())))
        else:
            gt_colors = None
            all_colors = sorted(set(list(quick_colors.keys()) + list(deep_colors.keys())))
        
        if all_colors:
            x = np.arange(len(all_colors))
            width = 0.25 if gt_colors else 0.35
            
            quick_color_counts = [quick_colors.get(color, 0) for color in all_colors]
            deep_color_counts = [deep_colors.get(color, 0) for color in all_colors]
            
            if gt_colors:
                # 3-way comparison
                gt_color_counts = [gt_colors.get(color, 0) for color in all_colors]
                ax4.bar(x - width, quick_color_counts, width, label='Quick', color='#2196F3', alpha=0.8)
                ax4.bar(x, deep_color_counts, width, label='Deep', color='#4CAF50', alpha=0.8)
                ax4.bar(x + width, gt_color_counts, width, label='Ground Truth', color='#FF5722', alpha=0.8)
            else:
                # 2-way comparison
                ax4.bar(x - width/2, quick_color_counts, width, label='Quick', color='#2196F3', alpha=0.8)
                ax4.bar(x + width/2, deep_color_counts, width, label='Deep', color='#4CAF50', alpha=0.8)
            
            ax4.set_ylabel('Count', fontsize=11, fontweight='bold')
            ax4.set_title('Color Distribution Comparison', fontsize=12, fontweight='bold')
            ax4.set_xticks(x)
            ax4.set_xticklabels(all_colors, rotation=45, ha='right', fontsize=9)
            ax4.legend(fontsize=9)
            ax4.grid(True, alpha=0.3, axis='y')
        else:
            ax4.text(0.5, 0.5, 'No Particles\nDetected', 
                    ha='center', va='center', fontsize=12, transform=ax4.transAxes)
            ax4.set_title('Color Distribution', fontsize=12, fontweight='bold')
        
        # 5. Area (Object Pixels) Distribution Comparison
        quick_areas = [f['area'] for f in quick_result.features] if quick_result.features else []
        deep_areas = [f['area'] for f in deep_result.features] if deep_result.features else []
        
        # Add ground truth areas if available
        if self.ground_truth and len(self.ground_truth) > 0:
            gt_areas = [p['area'] for p in self.ground_truth]
        else:
            gt_areas = []
        
        if quick_areas or deep_areas or gt_areas:
            # Create histogram bins
            all_areas = quick_areas + deep_areas + gt_areas
            if all_areas:
                bins = np.linspace(min(all_areas), max(all_areas), 15)
                
                ax5.hist(quick_areas, bins=bins, alpha=0.5, label='Quick', color='#2196F3', edgecolor='black')
                ax5.hist(deep_areas, bins=bins, alpha=0.5, label='Deep', color='#4CAF50', edgecolor='black')
                if gt_areas:
                    ax5.hist(gt_areas, bins=bins, alpha=0.5, label='Ground Truth', color='#FF5722', edgecolor='black')
                
                ax5.set_xlabel('Area (pixels)', fontsize=11, fontweight='bold')
                ax5.set_ylabel('Frequency', fontsize=11, fontweight='bold')
                ax5.set_title('Object Pixel Area Distribution', fontsize=12, fontweight='bold')
                ax5.legend(fontsize=9)
                ax5.grid(True, alpha=0.3, axis='y')
                
                # Add statistics
                if quick_areas:
                    quick_mean = np.mean(quick_areas)
                    ax5.axvline(quick_mean, color='#2196F3', linestyle='--', linewidth=2, alpha=0.7)
                if deep_areas:
                    deep_mean = np.mean(deep_areas)
                    ax5.axvline(deep_mean, color='#4CAF50', linestyle='--', linewidth=2, alpha=0.7)
                if gt_areas:
                    gt_mean = np.mean(gt_areas)
                    ax5.axvline(gt_mean, color='#FF5722', linestyle='--', linewidth=2, alpha=0.7)
        else:
            ax5.text(0.5, 0.5, 'No Particles\nDetected', 
                    ha='center', va='center', fontsize=12, transform=ax5.transAxes)
            ax5.set_title('Object Pixel Area Distribution', fontsize=12, fontweight='bold')
        
        # 6. Processing Time Comparison (only Quick and Deep, no ground truth time)
        time_methods = ['Quick', 'Deep']
        times = [quick_result.processing_time, deep_result.processing_time]
        time_colors = ['#2196F3', '#4CAF50']
        bars = ax6.bar(time_methods, times, color=time_colors, alpha=0.8, edgecolor='black', linewidth=2)
        
        ax6.set_ylabel('Time (seconds)', fontsize=11, fontweight='bold')
        ax6.set_title('Processing Time Comparison', fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='y')
        
        for bar, time_val in zip(bars, times):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + max(times) * 0.02,
                    f'{time_val:.2f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        self.figure.tight_layout()
        self.canvas.draw()
        
        # Switch to Statistics tab to show the charts
        self.tabs.setCurrentIndex(3)
    
    def create_comprehensive_comparison_charts(self):
        """Create comprehensive comparison charts for all available methods"""
        self.figure.clear()
        
        # Get all methods and their data
        methods = self.stats_comparator.get_all_methods()
        num_methods = len(methods)
        
        if num_methods < 2:
            return
        
        # Create subplot layout (2x2 grid - 4 main graphs)
        gs = self.figure.add_gridspec(2, 2, hspace=0.35, wspace=0.35)
        ax1 = self.figure.add_subplot(gs[0, 0])  # Detection counts
        ax2 = self.figure.add_subplot(gs[0, 1])  # Color distribution
        ax3 = self.figure.add_subplot(gs[1, 0])  # Shape distribution
        ax4 = self.figure.add_subplot(gs[1, 1])  # 3D Morphometric parameters
        
        # Color scheme for different methods
        color_map = {
            'Quick Analysis': '#2196F3',
            'Deep Analysis': '#4CAF50',
            'ML Benchmark': '#9C27B0',
            'Synthetic Ground Truth': '#FF5722'
        }
        
        # 1. Detection Counts
        comparison_table = self.stats_comparator.get_comparison_table()
        counts = [comparison_table[m]['num_detections'] for m in methods]
        colors = [color_map.get(m, '#757575') for m in methods]
        
        bars = ax1.bar(range(len(methods)), counts, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        ax1.set_ylabel('Detection Count', fontsize=11, fontweight='bold')
        ax1.set_title('Detection Count Comparison', fontsize=12, fontweight='bold')
        ax1.set_xticks(range(len(methods)))
        ax1.set_xticklabels([m.replace(' ', '\n') for m in methods], fontsize=9)
        ax1.grid(True, alpha=0.3, axis='y')
        
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(counts) * 0.02,
                    f'{int(count)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 2. Color Distribution (Stacked Bar Chart for 4 fluorescent colors)
        color_data = self.stats_comparator.get_parameter_comparison('color')
        if color_data:
            # Count 4 main colors: Red, Green, Blue, Yellow
            main_colors = ['Red', 'Green', 'Blue', 'Yellow']
            color_counts = {method: {c: 0 for c in main_colors} for method in methods}
            
            for method in methods:
                if method in color_data:
                    from collections import Counter
                    colors_list = color_data[method]
                    color_counter = Counter(colors_list)
                    
                    for color_name, count in color_counter.items():
                        # Map to 4 main colors
                        if 'Red' in color_name or 'Pink' in color_name or 'Orange' in color_name:
                            color_counts[method]['Red'] += count
                        elif 'Green' in color_name or 'Cyan' in color_name:
                            color_counts[method]['Green'] += count
                        elif 'Blue' in color_name or 'Purple' in color_name or 'Violet' in color_name:
                            color_counts[method]['Blue'] += count
                        elif 'Yellow' in color_name:
                            color_counts[method]['Yellow'] += count
            
            # Create stacked bar chart
            x_pos = np.arange(len(methods))
            bar_width = 0.6
            
            # Color mapping for visual representation
            visual_colors = {'Red': '#FF0000', 'Green': '#00FF00', 'Blue': '#0000FF', 'Yellow': '#FFFF00'}
            
            bottom = np.zeros(len(methods))
            for color_name in main_colors:
                counts_for_color = [color_counts[m][color_name] for m in methods]
                ax2.bar(x_pos, counts_for_color, bar_width, label=color_name,
                       bottom=bottom, color=visual_colors[color_name], alpha=0.8,
                       edgecolor='black', linewidth=1)
                bottom += counts_for_color
            
            ax2.set_ylabel('Count', fontsize=11, fontweight='bold')
            ax2.set_title('Color Distribution (4 Fluorescent Colors)', fontsize=12, fontweight='bold')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels([m.replace(' ', '\n') for m in methods], fontsize=9)
            ax2.legend(loc='upper right', fontsize=9)
            ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Shape Distribution Comparison
        all_shapes = set()
        for method in methods:
            shapes = comparison_table[method]['shapes']
            all_shapes.update(shapes.keys())
        all_shapes = sorted(all_shapes)
        
        if all_shapes:
            x = np.arange(len(all_shapes))
            width = 0.8 / num_methods
            
            for i, method in enumerate(methods):
                shapes = comparison_table[method]['shapes']
                counts = [shapes.get(shape, 0) for shape in all_shapes]
                offset = (i - num_methods/2 + 0.5) * width
                ax3.bar(x + offset, counts, width, label=method.replace(' Analysis', '').replace(' Benchmark', ''),
                       color=color_map.get(method, '#757575'), alpha=0.7)
            
            ax3.set_ylabel('Count', fontsize=11, fontweight='bold')
            ax3.set_title('Shape Distribution Comparison', fontsize=12, fontweight='bold')
            ax3.set_xticks(x)
            ax3.set_xticklabels(all_shapes, rotation=45, ha='right', fontsize=9)
            ax3.legend(fontsize=9, loc='upper right')
            ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. 3D Morphometric Parameters (Circularity, Aspect Ratio, Eccentricity)
        circ_data = self.stats_comparator.get_parameter_comparison('circularity')
        aspect_data = self.stats_comparator.get_parameter_comparison('aspect_ratio')
        ecc_data = self.stats_comparator.get_parameter_comparison('eccentricity')
        
        has_data = any(circ_data.get(m, []) and aspect_data.get(m, []) and ecc_data.get(m, []) for m in methods)
        
        if has_data:
            for method in methods:
                if method in circ_data and method in aspect_data and method in ecc_data:
                    if circ_data[method] and aspect_data[method] and ecc_data[method]:
                        # Limit to 100 points per method for clarity
                        circs = circ_data[method][:100]
                        aspects = aspect_data[method][:100]
                        eccs = ecc_data[method][:100]
                        
                        ax4.scatter(circs, aspects, eccs, 
                                   label=method.replace(' Analysis', '').replace(' Benchmark', ''),
                                   alpha=0.6, s=20, color=color_map.get(method, '#757575'))
            
            ax4.set_xlabel('Circularity', fontsize=9)
            ax4.set_ylabel('Aspect Ratio', fontsize=9)
            ax4.set_zlabel('Eccentricity', fontsize=9)
            ax4.set_title('3D Morphometric Parameters', fontsize=11, fontweight='bold')
            ax4.legend(fontsize=8, loc='upper left')
            ax4.view_init(elev=20, azim=45)
        else:
            # If no 3D data, show text message
            ax4.text2D(0.5, 0.5, 'Morphometric data\nnot available', 
                      ha='center', va='center', fontsize=10, transform=ax4.transAxes)
            ax4.set_title('3D Morphometric Parameters', fontsize=11, fontweight='bold')
        
        self.figure.tight_layout()
        self.canvas.draw()
        
        # Switch to Statistics tab
        self.tabs.setCurrentIndex(3)
        
    def run_benchmark(self):
        """Run comprehensive benchmarking with batch image processing"""
        # Ask user: single image or batch processing
        from PyQt5.QtWidgets import QInputDialog
        
        items = ["Single Image (Current)", "Batch Processing (Multiple Images)"]
        item, ok = QInputDialog.getItem(self, "Benchmark Mode", 
                                       "Select benchmark mode:", items, 0, False)
        
        if not ok:
            return
        
        if "Single" in item:
            self._run_single_benchmark()
        else:
            self._run_batch_benchmark()
    
    def _run_single_benchmark(self):
        """Run benchmark on current single image"""
        if self.current_image is None:
            QMessageBox.warning(self, "Warning", "Please load or generate an image first")
            return
            
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.statusBar().showMessage('Running benchmark...')
        self.results_text.clear()
        self.results_text.append("=" * 60)
        self.results_text.append("BENCHMARK TEST - SINGLE IMAGE")
        self.results_text.append("=" * 60)
        self.results_text.append("\nComparing Quick vs Deep analysis methods")
        if self.ground_truth_count > 0:
            self.results_text.append(f"Ground Truth: {self.ground_truth_count} particles\n")
        
        import time
        import numpy as np
        from datetime import datetime
        from collections import Counter
        from src.analysis.report_generator import ReportGenerator
        
        # Store comprehensive benchmark results
        benchmark_data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'image_info': {
                'width': self.current_image.shape[1],
                'height': self.current_image.shape[0],
                'has_ground_truth': self.ground_truth_count > 0
            },
            'ground_truth': self.ground_truth_count,
            'quick_analysis': {},
            'deep_analysis': {},
            'watershed_analysis': {}
        }
        
        # Run Quick Analysis
        self.results_text.append("\n[1/3] Running Quick Analysis...")
        QApplication.processEvents()
        
        quick_params = self.get_analysis_params('quick')
        start_time = time.time()
        quick_analyzer = QuickAnalyzer()
        quick_result = quick_analyzer.analyze(self.current_image, quick_params)
        quick_time = time.time() - start_time
        
        # Calculate metrics for Quick
        quick_metrics = self._calculate_metrics(quick_result, 'Quick')
        benchmark_data['quick_analysis'] = quick_metrics
        
        self.results_text.append(f"  Detected: {quick_result.num_detections} particles in {quick_time:.3f}s")
        
        # Run Deep Analysis
        self.results_text.append("\n[2/3] Running Deep Analysis...")
        QApplication.processEvents()
        
        deep_params = self.get_analysis_params('deep')
        start_time = time.time()
        deep_analyzer = DeepAnalyzer()
        deep_result = deep_analyzer.analyze(self.current_image, deep_params)
        deep_time = time.time() - start_time
        
        # Calculate metrics for Deep
        deep_metrics = self._calculate_metrics(deep_result, 'Deep')
        benchmark_data['deep_analysis'] = deep_metrics
        
        self.results_text.append(f"  Detected: {deep_result.num_detections} particles in {deep_time:.3f}s")
        
        # Run Watershed (if different from Deep)
        self.results_text.append("\n[3/3] Running Watershed Analysis...")
        QApplication.processEvents()
        
        watershed_params = PreprocessingParams(
            method='watershed',
            min_area=deep_params.min_area,
            max_area=deep_params.max_area,
            blur=deep_params.blur,
            distance_thresh=deep_params.distance_thresh,
            marker_size=deep_params.marker_size
        )
        start_time = time.time()
        watershed_result = deep_analyzer.analyze(self.current_image, watershed_params)
        watershed_time = time.time() - start_time
        
        # Calculate metrics for Watershed
        watershed_metrics = self._calculate_metrics(watershed_result, 'Watershed')
        benchmark_data['watershed_analysis'] = watershed_metrics
        
        self.results_text.append(f"  Detected: {watershed_result.num_detections} particles in {watershed_time:.3f}s")
        
        # Display summary
        self.results_text.append("\n" + "=" * 60)
        self.results_text.append("BENCHMARK SUMMARY")
        self.results_text.append("=" * 60)
        
        for method_name, metrics in [('Quick', quick_metrics), ('Deep', deep_metrics), ('Watershed', watershed_metrics)]:
            self.results_text.append(f"\n{method_name} Analysis:")
            self.results_text.append(f"  Detected: {metrics['detected']}")
            self.results_text.append(f"  Precision: {metrics['precision']:.3f}")
            self.results_text.append(f"  Recall: {metrics['recall']:.3f}")
            self.results_text.append(f"  F1-Score: {metrics['f1_score']:.3f}")
            self.results_text.append(f"  Time: {metrics['processing_time']:.3f}s")
        
        # Generate HTML report
        self.results_text.append("\n" + "=" * 60)
        self.results_text.append("GENERATING HTML REPORT...")
        QApplication.processEvents()
        
        try:
            # Extract ground truth distributions if available
            # Filter to only fluorescent colors and grouped shapes
            fluorescent_colors = {'Red', 'Green', 'Blue', 'Yellow'}
            gt_shapes = Counter()
            gt_colors = Counter()
            gt_areas = []
            if self.ground_truth and len(self.ground_truth) > 0:
                for particle in self.ground_truth:
                    # Map to grouped shape categories
                    shape = map_to_grouped_shape(particle.get('shape', 'Unknown'))
                    gt_shapes[shape] += 1
                    # Filter to only fluorescent colors
                    color = particle.get('color_label', 'Unknown')
                    if color in fluorescent_colors:
                        gt_colors[color] += 1
                    if 'area' in particle:
                        gt_areas.append(particle['area'])
            
            # Add ground truth distributions to benchmark data
            if gt_shapes or gt_colors or gt_areas:
                benchmark_data['ground_truth_distributions'] = {
                    'shape_distribution': dict(gt_shapes),
                    'color_distribution': dict(gt_colors),
                    'area_distribution': gt_areas
                }
            
            report_gen = ReportGenerator()
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = Path('benchmark_results')
            output_dir.mkdir(exist_ok=True)
            
            html_path = output_dir / f'benchmark_report_{timestamp}.html'
            report_gen.generate_benchmark_report(benchmark_data, str(html_path))
            
            self.results_text.append(f"\n✓ HTML Report saved: {html_path}")
            self.results_text.append("\nOpening report in browser...")
            
            # Open in browser
            import webbrowser
            webbrowser.open(f'file:///{html_path.absolute()}')
            
        except Exception as e:
            self.results_text.append(f"\n✗ Failed to generate report: {e}")
            import traceback
            traceback.print_exc()
        
        self.progress_bar.setVisible(False)
        self.statusBar().showMessage('Benchmark complete - Report generated')
        self.results_text.append("\n" + "=" * 60)
        self.results_text.append("BENCHMARK COMPLETE")
        self.results_text.append("=" * 60)
    
    def _find_ground_truth_file(self, img_path):
        """
        Find ground truth file for an image, checking multiple locations.
        
        Checks:
        1. Same directory as image: <imagename>_groundtruth.txt
        2. Parallel ground_truth directory (for YOLO datasets): ground_truth/<split>/<imagename>_groundtruth.txt
        
        Args:
            img_path: Path object for the image file
            
        Returns:
            Path object for ground truth file, or None if not found
        """
        from pathlib import Path
        
        # Try same directory first
        gt_path = img_path.parent / f"{img_path.stem}_groundtruth.txt"
        
        if gt_path.exists():
            return gt_path
        
        # Try parallel ground_truth directory structure
        # e.g., yolo_microplastic_dataset/images/train/img.png -> yolo_microplastic_dataset/ground_truth/train/img_groundtruth.txt
        parent = img_path.parent  # e.g., images/train/
        
        # Check if parent directory is named like a split (train/val/test) and grandparent is 'images'
        if parent.parent and parent.parent.name == 'images':
            # Found: .../images/train/ structure
            split_name = parent.name  # 'train', 'val', or 'test'
            dataset_root = parent.parent.parent  # Go up to dataset root
            gt_dir = dataset_root / 'ground_truth' / split_name
            gt_path_alt = gt_dir / f"{img_path.stem}_groundtruth.txt"
            
            if gt_path_alt.exists():
                print(f"DEBUG: Found ground truth in parallel structure: {gt_path_alt}")
                return gt_path_alt
        
        # Alternative: check if parent's parent contains 'dataset' in name (legacy support)
        parent_name = parent.parent.name if parent.parent else ""
        if parent_name and ('dataset' in parent_name.lower() or parent_name in ['yolo_microplastic_dataset']):
            split_name = parent.name
            gt_dir = parent.parent / 'ground_truth' / split_name
            gt_path_alt = gt_dir / f"{img_path.stem}_groundtruth.txt"
            
            if gt_path_alt.exists():
                print(f"DEBUG: Found ground truth in legacy structure: {gt_path_alt}")
                return gt_path_alt
        
        return None
    
    def _run_batch_benchmark(self):
        """Run benchmark on multiple images (batch processing)"""
        # Ask for image folder or generate synthetic images
        from PyQt5.QtWidgets import QInputDialog
        
        items = ["Load Image Folder", "Generate 200 Synthetic Images"]
        item, ok = QInputDialog.getItem(self, "Batch Source", 
                                       "Select image source:", items, 0, False)
        
        if not ok:
            return
        
        images_data = []
        
        if "Generate" in item:
            # Generate synthetic images
            num_images, ok = QInputDialog.getInt(self, "Number of Images",
                                                 "How many synthetic images?", 200, 1, 1000, 10)
            if not ok:
                return
            
            # Ask if user wants to save images
            from PyQt5.QtWidgets import QMessageBox
            from datetime import datetime
            from pathlib import Path
            
            save_reply = QMessageBox.question(self, 'Save Images?', 
                                             'Do you want to save generated images to disk?\n\n'
                                             'Yes: Save to benchmark_results/images/\n'
                                             'No: Process in memory only (faster)',
                                             QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            save_images = (save_reply == QMessageBox.Yes)
            
            # Create output directory if saving
            image_output_dir = None
            if save_images:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                image_output_dir = Path('benchmark_results') / 'images' / f'batch_{timestamp}'
                image_output_dir.mkdir(parents=True, exist_ok=True)
                self.results_text.append(f"Images will be saved to: {image_output_dir}")
            
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, num_images)
            self.statusBar().showMessage('Generating synthetic images...')
            
            from src.data_generation import SyntheticImageGenerator
            from config.settings import SyntheticImageParams
            generator = SyntheticImageGenerator()
            
            for i in range(num_images):
                self.progress_bar.setValue(i)
                QApplication.processEvents()
                
                # Generate with parameters from GUI interface
                import random
                params = SyntheticImageParams(
                    num_particles=self.synth_num_particles.value(),
                    shape_type=self.synth_shape_type.currentText(),
                    color_type=self.synth_color_type.currentText(),
                    enable_blur=self.synth_enable_blur.isChecked(),
                    blur_kernel=self.synth_blur_kernel.value(),
                    enable_glow=self.synth_enable_glow.isChecked(),
                    glow_intensity=self.synth_glow_intensity.value(),
                    image_width=self.synth_width.value(),
                    image_height=self.synth_height.value(),
                    background_brightness=self.synth_brightness.value(),
                    particle_brightness=self.synth_particle_brightness.value()
                )
                
                image, ground_truth = generator.generate(params)
                
                # Save image to disk if requested
                if save_images and image_output_dir:
                    img_filename = f'synthetic_{i+1:03d}.png'
                    img_path = image_output_dir / img_filename
                    # Convert RGB to BGR for OpenCV
                    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(img_path), image_bgr)
                    
                    # Save ground truth metadata with FULL details (shape, color, area, etc.)
                    gt_filename = f'synthetic_{i+1:03d}_groundtruth.txt'
                    gt_path = image_output_dir / gt_filename
                    with open(gt_path, 'w') as f:
                        f.write(f"Total Particles: {len(ground_truth)}\n\n")
                        for idx, particle in enumerate(ground_truth, 1):
                            f.write(f"Particle {idx}:\n")
                            f.write(f"  Shape: {particle.get('shape', 'Unknown')}\n")
                            f.write(f"  Color: {particle.get('color_label', 'Unknown')}\n")
                            if 'wavelength' in particle:
                                f.write(f"  Wavelength: {particle['wavelength']}nm\n")
                            if 'position' in particle:
                                pos = particle['position']
                                f.write(f"  Position: ({pos[0]}, {pos[1]})\n")
                            if 'area' in particle:
                                f.write(f"  Area: {particle['area']:.2f} px²\n")
                            if 'size' in particle:
                                f.write(f"  Size: {particle['size']:.2f} px\n")
                            f.write("\n")
                
                images_data.append({
                    'image': image,
                    'ground_truth': len(ground_truth),
                    'ground_truth_data': ground_truth,  # Store full ground truth data
                    'name': f'synthetic_{i+1:03d}'
                })
            
            if save_images:
                self.statusBar().showMessage(f'Generated and saved {num_images} images to {image_output_dir}')
                self.results_text.append(f"\n✓ Saved {num_images} images to: {image_output_dir}")
            else:
                self.statusBar().showMessage(f'Generated {num_images} synthetic images (in memory)')
            
        else:
            # Load images from folder
            folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
            if not folder:
                return
            
            from pathlib import Path
            image_files = []
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.tiff', '*.bmp']:
                image_files.extend(Path(folder).glob(ext))
            
            if not image_files:
                QMessageBox.warning(self, "Warning", "No images found in folder")
                return
            
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, len(image_files))
            
            for i, img_path in enumerate(image_files):
                self.progress_bar.setValue(i)
                QApplication.processEvents()
                
                image = cv2.imread(str(img_path))
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Try to load ground truth with full particle details
                    ground_truth_count = 0
                    ground_truth_data = []
                    gt_path = self._find_ground_truth_file(img_path)
                    
                    if gt_path and gt_path.exists():
                        try:
                            with open(gt_path, 'r') as f:
                                content = f.read().strip()
                                
                                # Parse ground truth file
                                # Support two formats:
                                # Format 1 (simple): "Particles: N\nShapes:\n  1. Fragment\n  2. Fiber..."
                                # Format 2 (detailed): "Total Particles: N\n\nParticle 1:\n  Shape: ...\n  Color: ..."
                                lines = content.split('\n')
                                
                                # Get total count (try both formats)
                                if lines and 'Particles:' in lines[0]:
                                    ground_truth_count = int(lines[0].split(':')[1].strip())
                                
                                # Check format type
                                is_simple_format = any('Shapes:' in line for line in lines)
                                
                                if is_simple_format:
                                    # Parse simple format: just numbered shapes
                                    in_shapes_section = False
                                    for line in lines:
                                        line = line.strip()
                                        if line == 'Shapes:':
                                            in_shapes_section = True
                                            continue
                                        if in_shapes_section and line:
                                            # Line format: "  1. Fragment" or "1. Fragment"
                                            if '.' in line:
                                                shape = line.split('.', 1)[1].strip()
                                                ground_truth_data.append({
                                                    'shape': shape
                                                })
                                else:
                                    # Parse detailed format with full particle info
                                    current_particle = {}
                                    for line in lines[1:]:
                                        line = line.strip()
                                        if not line:
                                            if current_particle:
                                                ground_truth_data.append(current_particle)
                                                current_particle = {}
                                            continue
                                        
                                        if line.startswith('Particle '):
                                            if current_particle:
                                                ground_truth_data.append(current_particle)
                                            current_particle = {}
                                        elif ':' in line:
                                            key, value = line.split(':', 1)
                                            key = key.strip()
                                            value = value.strip()
                                            
                                            if key == 'Shape':
                                                current_particle['shape'] = value
                                            elif key == 'Color':
                                                current_particle['color_label'] = value
                                            elif key == 'Wavelength':
                                                current_particle['wavelength'] = int(value.replace('nm', ''))
                                            elif key == 'Position':
                                                # Parse (x, y) format
                                                pos_str = value.strip('()')
                                                x, y = pos_str.split(',')
                                                current_particle['position'] = (float(x.strip()), float(y.strip()))
                                            elif key == 'Area':
                                                current_particle['area'] = float(value.split()[0])
                                            elif key == 'Size':
                                                current_particle['size'] = float(value.split()[0])
                                    
                                    # Add last particle
                                    if current_particle:
                                        ground_truth_data.append(current_particle)
                                    
                        except Exception as e:
                            print(f"Warning: Could not parse ground truth from {gt_path}: {e}")
                            ground_truth_data = []
                    
                    images_data.append({
                        'image': image,
                        'ground_truth': ground_truth_count,
                        'ground_truth_data': ground_truth_data,
                        'name': img_path.stem
                    })
        
        if not images_data:
            QMessageBox.warning(self, "Warning", "No images to process")
            self.progress_bar.setVisible(False)
            return
        
        # Count how many images have ground truth
        images_with_gt = sum(1 for img in images_data if img['ground_truth'] > 0)
        
        # Run benchmark on all images
        self.results_text.clear()
        self.results_text.append("=" * 60)
        self.results_text.append(f"BATCH BENCHMARK - {len(images_data)} IMAGES")
        self.results_text.append("=" * 60)
        if images_with_gt > 0:
            self.results_text.append(f"✓ Ground truth loaded for {images_with_gt}/{len(images_data)} images")
        else:
            self.results_text.append(f"⚠ No ground truth files found")
        
        quick_results_list = []
        deep_results_list = []
        
        self.progress_bar.setRange(0, len(images_data) * 2)  # Quick + Deep
        
        import time
        from collections import Counter
        
        # Process all images with Quick Analysis
        self.results_text.append(f"\n[1/2] Running Quick Analysis on {len(images_data)} images...")
        QApplication.processEvents()
        
        quick_params = self.get_analysis_params('quick')
        quick_analyzer = QuickAnalyzer()
        
        for i, img_data in enumerate(images_data):
            self.progress_bar.setValue(i)
            self.statusBar().showMessage(f'Quick Analysis: {i+1}/{len(images_data)}')
            QApplication.processEvents()
            
            result = quick_analyzer.analyze(img_data['image'], quick_params)
            
            # Debug: Log detection counts for first 5 images and any large discrepancies
            if i < 5 or (img_data['ground_truth'] > 0 and abs(result.num_detections - img_data['ground_truth']) > 5):
                print(f"DEBUG Quick [{i+1}] {img_data['name']}: Detected={result.num_detections}, GT={img_data['ground_truth']}")
            
            quick_results_list.append({
                'name': img_data['name'],
                'detected': result.num_detections,
                'ground_truth': img_data['ground_truth'],
                'time': result.processing_time,
                'result': result
            })
        
        # Process all images with Deep Analysis
        self.results_text.append(f"\n[2/2] Running Deep Analysis on {len(images_data)} images...")
        QApplication.processEvents()
        
        deep_params = self.get_analysis_params('deep')
        deep_analyzer = DeepAnalyzer()
        
        for i, img_data in enumerate(images_data):
            self.progress_bar.setValue(len(images_data) + i)
            self.statusBar().showMessage(f'Deep Analysis: {i+1}/{len(images_data)}')
            QApplication.processEvents()
            
            result = deep_analyzer.analyze(img_data['image'], deep_params)
            deep_results_list.append({
                'name': img_data['name'],
                'detected': result.num_detections,
                'ground_truth': img_data['ground_truth'],
                'time': result.processing_time,
                'result': result
            })
        
        # Calculate aggregate statistics
        import numpy as np
        
        quick_detections = [r['detected'] for r in quick_results_list]
        deep_detections = [r['detected'] for r in deep_results_list]
        quick_times = [r['time'] for r in quick_results_list]
        deep_times = [r['time'] for r in deep_results_list]
        
        ground_truths = [r['ground_truth'] for r in quick_results_list if r['ground_truth'] > 0]
        
        # Display summary
        self.results_text.append("\n" + "=" * 60)
        self.results_text.append("BATCH BENCHMARK SUMMARY")
        self.results_text.append("=" * 60)
        
        self.results_text.append(f"\nProcessed: {len(images_data)} images")
        
        # Display results in comparison table format
        if ground_truths:
            self.results_text.append("\n" + "-" * 60)
            self.results_text.append(f"{'Method':<20} {'Avg Detections':<20} {'Avg Time (s)':<20}")
            self.results_text.append("-" * 60)
            self.results_text.append(f"{'Ground Truth':<20} {np.mean(ground_truths):<20.1f} {'N/A':<20}")
            self.results_text.append(f"{'Quick Analysis':<20} {np.mean(quick_detections):<20.1f} {np.mean(quick_times):<20.3f}")
            self.results_text.append(f"{'Deep Analysis':<20} {np.mean(deep_detections):<20.1f} {np.mean(deep_times):<20.3f}")
            self.results_text.append("-" * 60)
            
            self.results_text.append(f"\nGround Truth: {len(ground_truths)} images with GT (out of {len(images_data)} total)")
            self.results_text.append(f"  Total particles: {np.sum(ground_truths):.0f}")
            self.results_text.append(f"  Std deviation: ± {np.std(ground_truths):.1f}")
        else:
            self.results_text.append("\n" + "-" * 60)
            self.results_text.append(f"{'Method':<20} {'Avg Detections':<20} {'Avg Time (s)':<20}")
            self.results_text.append("-" * 60)
            self.results_text.append(f"{'Quick Analysis':<20} {np.mean(quick_detections):<20.1f} {np.mean(quick_times):<20.3f}")
            self.results_text.append(f"{'Deep Analysis':<20} {np.mean(deep_detections):<20.1f} {np.mean(deep_times):<20.3f}")
            self.results_text.append("-" * 60)
        
        self.results_text.append(f"\nQuick Analysis Details:")
        self.results_text.append(f"  Total detections: {np.sum(quick_detections):.0f}")
        self.results_text.append(f"  Std deviation: ± {np.std(quick_detections):.1f}")
        self.results_text.append(f"  Total time: {np.sum(quick_times):.1f}s")
        
        self.results_text.append(f"\nDeep Analysis Details:")
        self.results_text.append(f"  Total detections: {np.sum(deep_detections):.0f}")
        self.results_text.append(f"  Std deviation: ± {np.std(deep_detections):.1f}")
        self.results_text.append(f"  Total time: {np.sum(deep_times):.1f}s")
        
        if ground_truths:
            # Calculate and display metrics
            def calc_metrics_batch(detections, gts):
                """
                Calculate aggregate precision, recall, F1 across all images.
                Uses per-image TP/FP/FN and sums them.
                """
                total_tp = total_fp = total_fn = 0
                for det, gt in zip(detections, gts):
                    # Calculate per-image metrics
                    tp = min(det, gt)  # True positives
                    fp = max(0, det - gt)  # False positives (over-detection)
                    fn = max(0, gt - det)  # False negatives (missed particles)
                    
                    total_tp += tp
                    total_fp += fp
                    total_fn += fn
                
                # Calculate overall metrics from totals
                precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
                recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                
                return precision, recall, f1
            
            quick_p, quick_r, quick_f = calc_metrics_batch(
                [r['detected'] for r in quick_results_list if r['ground_truth'] > 0],
                [r['ground_truth'] for r in quick_results_list if r['ground_truth'] > 0]
            )
            deep_p, deep_r, deep_f = calc_metrics_batch(
                [r['detected'] for r in deep_results_list if r['ground_truth'] > 0],
                [r['ground_truth'] for r in deep_results_list if r['ground_truth'] > 0]
            )
            
            # Debug information
            quick_det_with_gt = [r['detected'] for r in quick_results_list if r['ground_truth'] > 0]
            quick_gt_with_gt = [r['ground_truth'] for r in quick_results_list if r['ground_truth'] > 0]
            print(f"DEBUG Benchmark: {len(quick_gt_with_gt)} images with GT")
            print(f"DEBUG Quick: Total detected={sum(quick_det_with_gt)}, Total GT={sum(quick_gt_with_gt)}")
            print(f"DEBUG Quick: Precision={quick_p:.3f}, Recall={quick_r:.3f}, F1={quick_f:.3f}")
            
            self.results_text.append("\n" + "=" * 60)
            self.results_text.append("ACCURACY METRICS (vs Ground Truth)")
            self.results_text.append("=" * 60)
            self.results_text.append(f"\n{'Method':<20} {'Precision':<15} {'Recall':<15} {'F1-Score':<15}")
            self.results_text.append("-" * 60)
            self.results_text.append(f"{'Quick Analysis':<20} {quick_p:<15.3f} {quick_r:<15.3f} {quick_f:<15.3f}")
            self.results_text.append(f"{'Deep Analysis':<20} {deep_p:<15.3f} {deep_r:<15.3f} {deep_f:<15.3f}")
            self.results_text.append("-" * 60)
        else:
            self.results_text.append(f"\n⚠ No ground truth data found")
            self.results_text.append(f"  To calculate metrics, ensure ground truth files exist:")
            self.results_text.append(f"  Format: <imagename>_groundtruth.txt")
            self.results_text.append(f"  Content: 'Particles: <count>'")
        
        # Display aggregated distributions across all images
        self.results_text.append(f"\n" + "=" * 60)
        self.results_text.append("AGGREGATED DISTRIBUTIONS (ALL IMAGES)")
        self.results_text.append("=" * 60)
        
        # Count total particles across all images
        from collections import Counter
        
        # Quick Analysis aggregate distributions
        quick_all_shapes = Counter()
        quick_all_colors = Counter()
        quick_all_areas = []
        for r in quick_results_list:
            if r['result'].features:
                for f in r['result'].features:
                    quick_all_shapes[f.get('shape', 'Unknown')] += 1
                    quick_all_colors[f.get('color', 'Unknown')] += 1
                    quick_all_areas.append(f.get('area', 0))
        
        self.results_text.append(f"\nQuick Analysis - Total Particles: {sum(quick_detections)}")
        if quick_all_shapes:
            self.results_text.append(f"  Shapes: {dict(quick_all_shapes)}")
        if quick_all_colors:
            self.results_text.append(f"  Colors: {dict(quick_all_colors)}")
        if quick_all_areas:
            self.results_text.append(f"  Area range: {min(quick_all_areas):.0f} - {max(quick_all_areas):.0f} pixels")
        
        # Deep Analysis aggregate distributions
        deep_all_shapes = Counter()
        deep_all_colors = Counter()
        deep_all_areas = []
        for r in deep_results_list:
            if r['result'].features:
                for f in r['result'].features:
                    deep_all_shapes[f.get('shape', 'Unknown')] += 1
                    deep_all_colors[f.get('color', 'Unknown')] += 1
                    deep_all_areas.append(f.get('area', 0))
        
        self.results_text.append(f"\nDeep Analysis - Total Particles: {sum(deep_detections)}")
        if deep_all_shapes:
            self.results_text.append(f"  Shapes: {dict(deep_all_shapes)}")
        if deep_all_colors:
            self.results_text.append(f"  Colors: {dict(deep_all_colors)}")
        if deep_all_areas:
            self.results_text.append(f"  Area range: {min(deep_all_areas):.0f} - {max(deep_all_areas):.0f} pixels")
        
        # Ground Truth aggregate distributions (if available)
        gt_all_shapes = Counter()
        gt_all_colors = Counter()
        gt_all_areas = []
        has_gt_data = False
        for img_data in images_data:
            if 'ground_truth_data' in img_data and img_data['ground_truth_data']:
                has_gt_data = True
                for particle in img_data['ground_truth_data']:
                    gt_all_shapes[particle.get('shape', 'Unknown')] += 1
                    gt_all_colors[particle.get('color_label', 'Unknown')] += 1
                    if 'area' in particle:
                        gt_all_areas.append(particle['area'])
        
        if has_gt_data:
            self.results_text.append(f"\nGround Truth - Total Particles: {sum(ground_truths) if ground_truths else 0}")
            if gt_all_shapes:
                self.results_text.append(f"  Shapes: {dict(gt_all_shapes)}")
            if gt_all_colors:
                self.results_text.append(f"  Colors: {dict(gt_all_colors)}")
            if gt_all_areas:
                self.results_text.append(f"  Area range: {min(gt_all_areas):.0f} - {max(gt_all_areas):.0f} pixels")
        
        # Diagnostic: Compare distributions
        self.results_text.append(f"\n" + "=" * 60)
        self.results_text.append("SHAPE DISTRIBUTION COMPARISON")
        self.results_text.append("=" * 60)
        all_shape_names = sorted(set(list(quick_all_shapes.keys()) + list(deep_all_shapes.keys()) + list(gt_all_shapes.keys())))
        self.results_text.append(f"\n{'Shape':<20} {'Quick':<10} {'Deep':<10} {'GT':<10} {'Diff':<15}")
        self.results_text.append("-" * 60)
        for shape in all_shape_names:
            q = quick_all_shapes.get(shape, 0)
            d = deep_all_shapes.get(shape, 0)
            g = gt_all_shapes.get(shape, 0)
            diff = f"{(d-g)/g*100:+.1f}%" if g > 0 else "N/A"
            self.results_text.append(f"{shape:<20} {q:<10} {d:<10} {g:<10} {diff:<15}")
        
        self.results_text.append(f"\n" + "=" * 60)
        self.results_text.append("COLOR DISTRIBUTION COMPARISON")
        self.results_text.append("=" * 60)
        all_color_names = sorted(set(list(quick_all_colors.keys()) + list(deep_all_colors.keys()) + list(gt_all_colors.keys())))
        self.results_text.append(f"\n{'Color':<20} {'Quick':<10} {'Deep':<10} {'GT':<10} {'Diff':<15}")
        self.results_text.append("-" * 60)
        for color in all_color_names:
            q = quick_all_colors.get(color, 0)
            d = deep_all_colors.get(color, 0)
            g = gt_all_colors.get(color, 0)
            diff = f"{(d-g)/g*100:+.1f}%" if g > 0 else "N/A"
            self.results_text.append(f"{color:<20} {q:<10} {d:<10} {g:<10} {diff:<15}")
        
        self.results_text.append(f"\n⚠️  WHY GROUND TRUTH ≠ DETECTED:")
        self.results_text.append(f"  1️⃣ Fragment OVER-detected ({deep_all_shapes.get('Fragment', 0)} vs GT {gt_all_shapes.get('Fragment', 0)}):")
        self.results_text.append(f"     → Watershed splits overlapping particles into multiple fragments")
        self.results_text.append(f"     → Edge detection creates extra small fragments")
        self.results_text.append(f"  2️⃣ Film/Irregular UNDER-detected (missing from Quick/Deep):")
        self.results_text.append(f"     → Don't meet shape classification thresholds (AR, C, E)")
        self.results_text.append(f"     → May be classified as Fragment instead")
        self.results_text.append(f"  3️⃣ Fiber/Filament under-detected:")
        self.results_text.append(f"     → Elongated shapes may break apart during preprocessing")
        self.results_text.append(f"  💡 SOLUTIONS:")
        self.results_text.append(f"     • Adjust min_area threshold to capture smaller Film/Irregular")
        self.results_text.append(f"     • Tune watershed distance_thresh to reduce Fragment splitting")
        self.results_text.append(f"     • Review shape classification thresholds in constants.py")
        
        # Create comparison charts
        self._create_batch_comparison_charts(quick_results_list, deep_results_list, images_data)
        
        # Generate HTML report
        self.results_text.append("\n" + "=" * 60)
        self.results_text.append("GENERATING HTML REPORT...")
        QApplication.processEvents()
        
        try:
            from datetime import datetime
            from pathlib import Path
            from src.analysis.report_generator import ReportGenerator
            
            # Calculate metrics for batch
            def calculate_batch_metrics(detections_list, ground_truths_list):
                """Calculate aggregate precision, recall, F1 for batch"""
                if not ground_truths_list or len(ground_truths_list) == 0:
                    return 0.0, 0.0, 0.0
                
                total_tp = total_fp = total_fn = 0
                for detected, gt in zip(detections_list, ground_truths_list):
                    if gt > 0:
                        tp = min(detected, gt)
                        fp = max(0, detected - gt)
                        fn = max(0, gt - detected)
                        total_tp += tp
                        total_fp += fp
                        total_fn += fn
                
                precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
                recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                
                return precision, recall, f1
            
            # Get ground truth values for images that have them
            gt_values = [r['ground_truth'] for r in quick_results_list if r['ground_truth'] > 0]
            quick_det_with_gt = [r['detected'] for r in quick_results_list if r['ground_truth'] > 0]
            deep_det_with_gt = [r['detected'] for r in deep_results_list if r['ground_truth'] > 0]
            
            quick_precision, quick_recall, quick_f1 = calculate_batch_metrics(quick_det_with_gt, gt_values)
            deep_precision, deep_recall, deep_f1 = calculate_batch_metrics(deep_det_with_gt, gt_values)
            
            # Debug output
            self.results_text.append(f"\n[DEBUG] Metrics for HTML Report:")
            self.results_text.append(f"  Quick: P={quick_precision:.3f}, R={quick_recall:.3f}, F1={quick_f1:.3f}")
            self.results_text.append(f"  Deep: P={deep_precision:.3f}, R={deep_recall:.3f}, F1={deep_f1:.3f}")
            self.results_text.append(f"  GT values count: {len(gt_values)}, Quick det count: {len(quick_det_with_gt)}")
            
            # Aggregate shape, color, and area distributions from all images
            from collections import Counter
            
            # Filter to only fluorescent colors and grouped shapes
            fluorescent_colors = {'Red', 'Green', 'Blue', 'Yellow'}
            
            # Quick Analysis distributions
            quick_shapes = Counter()
            quick_colors = Counter()
            quick_areas = []
            for r in quick_results_list:
                if r['result'].features:
                    for f in r['result'].features:
                        shape = map_to_grouped_shape(f.get('shape', 'Unknown'))
                        quick_shapes[shape] += 1
                        color = f.get('color', 'Unknown')
                        if color in fluorescent_colors:
                            quick_colors[color] += 1
                        quick_areas.append(f.get('area', 0))
            
            # Deep Analysis distributions
            deep_shapes = Counter()
            deep_colors = Counter()
            deep_areas = []
            for r in deep_results_list:
                if r['result'].features:
                    for f in r['result'].features:
                        shape = map_to_grouped_shape(f.get('shape', 'Unknown'))
                        deep_shapes[shape] += 1
                        color = f.get('color', 'Unknown')
                        if color in fluorescent_colors:
                            deep_colors[color] += 1
                        deep_areas.append(f.get('area', 0))
            
            # Ground Truth distributions (aggregate from all images)
            # Filter to only fluorescent colors and grouped shapes
            fluorescent_colors = {'Red', 'Green', 'Blue', 'Yellow'}
            gt_shapes = Counter()
            gt_colors = Counter()
            gt_areas = []
            for img_data in images_data:
                if 'ground_truth_data' in img_data and img_data['ground_truth_data']:
                    for particle in img_data['ground_truth_data']:
                        # Map to grouped shape categories
                        shape = map_to_grouped_shape(particle.get('shape', 'Unknown'))
                        gt_shapes[shape] += 1
                        # Filter to only fluorescent colors
                        color = particle.get('color_label', 'Unknown')
                        if color in fluorescent_colors:
                            gt_colors[color] += 1
                        if 'area' in particle:
                            gt_areas.append(particle['area'])
            
            # Aggregate data for report
            batch_data = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'num_images': len(images_data),
                'quick_analysis': {
                    'detected': int(np.mean(quick_detections)),
                    'ground_truth': int(np.mean(gt_values)) if gt_values else 0,
                    'precision': quick_precision,
                    'recall': quick_recall,
                    'f1_score': quick_f1,
                    'processing_time': np.mean(quick_times),
                    'shape_distribution': dict(quick_shapes),
                    'color_distribution': dict(quick_colors),
                    'area_distribution': quick_areas
                },
                'deep_analysis': {
                    'detected': int(np.mean(deep_detections)),
                    'ground_truth': int(np.mean(gt_values)) if gt_values else 0,
                    'precision': deep_precision,
                    'recall': deep_recall,
                    'f1_score': deep_f1,
                    'processing_time': np.mean(deep_times),
                    'shape_distribution': dict(deep_shapes),
                    'color_distribution': dict(deep_colors),
                    'area_distribution': deep_areas
                },
                'ground_truth': int(np.mean(gt_values)) if gt_values else 0,
                'ground_truth_distributions': {
                    'shape_distribution': dict(gt_shapes),
                    'color_distribution': dict(gt_colors),
                    'area_distribution': gt_areas
                } if gt_shapes or gt_colors or gt_areas else None
            }
            
            report_gen = ReportGenerator()
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = Path('benchmark_results')
            output_dir.mkdir(exist_ok=True)
            
            html_path = output_dir / f'batch_benchmark_{len(images_data)}images_{timestamp}.html'
            report_gen.generate_benchmark_report(batch_data, str(html_path))
            
            self.results_text.append(f"\n✓ HTML Report saved: {html_path}")
            
            # Open in browser
            import webbrowser
            webbrowser.open(f'file:///{html_path.absolute()}')
            
        except Exception as e:
            self.results_text.append(f"\n✗ Failed to generate report: {e}")
        
        self.progress_bar.setVisible(False)
        self.statusBar().showMessage(f'Batch benchmark complete - {len(images_data)} images processed')
        self.results_text.append("\n" + "=" * 60)
        self.results_text.append("BATCH BENCHMARK COMPLETE")
        self.results_text.append("=" * 60)
    
    def _create_batch_comparison_charts(self, quick_results, deep_results, images_data):
        """Create comparison charts for batch benchmark with ground truth"""
        self.figure.clear()
        
        import numpy as np
        
        # Create 2x2 layout
        ax1 = self.figure.add_subplot(2, 2, 1)
        ax2 = self.figure.add_subplot(2, 2, 2)
        ax3 = self.figure.add_subplot(2, 2, 3)
        ax4 = self.figure.add_subplot(2, 2, 4)
        
        # 1. Detection count distribution with ground truth
        quick_detections = [r['detected'] for r in quick_results]
        deep_detections = [r['detected'] for r in deep_results]
        ground_truths = [img['ground_truth'] for img in images_data if img['ground_truth'] > 0]
        
        max_val = max(max(quick_detections), max(deep_detections))
        if ground_truths:
            max_val = max(max_val, max(ground_truths))
        
        bins = np.linspace(0, max_val, 30)
        ax1.hist(quick_detections, bins=bins, alpha=0.5, label='Quick', color='#2196F3', edgecolor='black')
        ax1.hist(deep_detections, bins=bins, alpha=0.5, label='Deep', color='#4CAF50', edgecolor='black')
        if ground_truths:
            ax1.hist(ground_truths, bins=bins, alpha=0.5, label='Ground Truth', color='#FF5722', edgecolor='black')
        ax1.set_xlabel('Particle Count', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax1.set_title('Detection Count Distribution', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Scatter plot: Quick vs Deep vs Ground Truth
        if ground_truths and len(ground_truths) == len(quick_detections):
            # If we have ground truth for all images, show GT comparison
            gt_full = [img['ground_truth'] for img in images_data]
            ax2.scatter(gt_full, quick_detections, alpha=0.6, s=50, c='#2196F3', 
                       edgecolors='black', label='Quick vs GT')
            ax2.scatter(gt_full, deep_detections, alpha=0.6, s=50, c='#4CAF50', 
                       edgecolors='black', label='Deep vs GT')
            max_val = max(max(gt_full), max(quick_detections), max(deep_detections))
            ax2.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Match')
            ax2.set_xlabel('Ground Truth Count', fontsize=11, fontweight='bold')
            ax2.set_ylabel('Detected Count', fontsize=11, fontweight='bold')
            ax2.set_title('Detection vs Ground Truth', fontsize=12, fontweight='bold')
        else:
            # No complete GT, show Quick vs Deep
            ax2.scatter(quick_detections, deep_detections, alpha=0.5, s=50, c='purple', edgecolors='black')
            max_val = max(max(quick_detections), max(deep_detections))
            ax2.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Agreement')
            ax2.set_xlabel('Quick Analysis Count', fontsize=11, fontweight='bold')
            ax2.set_ylabel('Deep Analysis Count', fontsize=11, fontweight='bold')
            ax2.set_title('Quick vs Deep Detection Counts', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Processing time comparison
        quick_times = [r['time'] for r in quick_results]
        deep_times = [r['time'] for r in deep_results]
        
        methods = ['Quick', 'Deep']
        avg_times = [np.mean(quick_times), np.mean(deep_times)]
        std_times = [np.std(quick_times), np.std(deep_times)]
        
        bars = ax3.bar(methods, avg_times, yerr=std_times, capsize=10, 
                      color=['#2196F3', '#4CAF50'], alpha=0.8, edgecolor='black', linewidth=2)
        ax3.set_ylabel('Time (seconds)', fontsize=11, fontweight='bold')
        ax3.set_title('Average Processing Time', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        for bar, time_val in zip(bars, avg_times):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + max(avg_times) * 0.02,
                    f'{time_val:.3f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 4. Statistics summary with ground truth
        stats_text = f"Batch Statistics ({len(quick_results)} images)\n\n"
        
        if ground_truths:
            stats_text += f"Ground Truth:\n"
            stats_text += f"  Avg: {np.mean(ground_truths):.1f} ± {np.std(ground_truths):.1f}\n"
            stats_text += f"  Min: {np.min(ground_truths)}, Max: {np.max(ground_truths)}\n\n"
        
        stats_text += f"Quick Analysis:\n"
        stats_text += f"  Avg: {np.mean(quick_detections):.1f} ± {np.std(quick_detections):.1f}\n"
        stats_text += f"  Min: {np.min(quick_detections)}, Max: {np.max(quick_detections)}\n"
        stats_text += f"  Time: {np.mean(quick_times):.3f}s avg\n\n"
        
        stats_text += f"Deep Analysis:\n"
        stats_text += f"  Avg: {np.mean(deep_detections):.1f} ± {np.std(deep_detections):.1f}\n"
        stats_text += f"  Min: {np.min(deep_detections)}, Max: {np.max(deep_detections)}\n"
        stats_text += f"  Time: {np.mean(deep_times):.3f}s avg\n\n"
        
        correlation = np.corrcoef(quick_detections, deep_detections)[0, 1]
        stats_text += f"Correlation: {correlation:.3f}\n"
        
        difference = np.array(deep_detections) - np.array(quick_detections)
        stats_text += f"Avg Difference: {np.mean(difference):+.1f}"
        
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, 
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax4.set_title('Statistical Summary', fontsize=12, fontweight='bold')
        ax4.axis('off')
        self.figure.clear()
        
        import numpy as np
        
        # Create 2x2 layout
        ax1 = self.figure.add_subplot(2, 2, 1)
        ax2 = self.figure.add_subplot(2, 2, 2)
        ax3 = self.figure.add_subplot(2, 2, 3)
        ax4 = self.figure.add_subplot(2, 2, 4)
        
        # 1. Detection count distribution
        quick_detections = [r['detected'] for r in quick_results]
        deep_detections = [r['detected'] for r in deep_results]
        
        bins = np.linspace(0, max(max(quick_detections), max(deep_detections)), 30)
        ax1.hist(quick_detections, bins=bins, alpha=0.6, label='Quick', color='#2196F3', edgecolor='black')
        ax1.hist(deep_detections, bins=bins, alpha=0.6, label='Deep', color='#4CAF50', edgecolor='black')
        ax1.set_xlabel('Particle Count', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax1.set_title('Detection Count Distribution', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Scatter plot: Quick vs Deep
        ax2.scatter(quick_detections, deep_detections, alpha=0.5, s=50, c='purple', edgecolors='black')
        max_val = max(max(quick_detections), max(deep_detections))
        ax2.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Agreement')
        ax2.set_xlabel('Quick Analysis Count', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Deep Analysis Count', fontsize=11, fontweight='bold')
        ax2.set_title('Quick vs Deep Detection Counts', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Processing time comparison
        quick_times = [r['time'] for r in quick_results]
        deep_times = [r['time'] for r in deep_results]
        
        methods = ['Quick', 'Deep']
        avg_times = [np.mean(quick_times), np.mean(deep_times)]
        std_times = [np.std(quick_times), np.std(deep_times)]
        
        bars = ax3.bar(methods, avg_times, yerr=std_times, capsize=10, 
                      color=['#2196F3', '#4CAF50'], alpha=0.8, edgecolor='black', linewidth=2)
        ax3.set_ylabel('Time (seconds)', fontsize=11, fontweight='bold')
        ax3.set_title('Average Processing Time', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        for bar, time_val in zip(bars, avg_times):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + max(avg_times) * 0.02,
                    f'{time_val:.3f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 4. Statistics summary
        stats_text = f"Batch Statistics ({len(quick_results)} images)\n\n"
        stats_text += f"Quick Analysis:\n"
        stats_text += f"  Avg: {np.mean(quick_detections):.1f} ± {np.std(quick_detections):.1f}\n"
        stats_text += f"  Min: {np.min(quick_detections)}, Max: {np.max(quick_detections)}\n"
        stats_text += f"  Time: {np.mean(quick_times):.3f}s avg\n\n"
        
        stats_text += f"Deep Analysis:\n"
        stats_text += f"  Avg: {np.mean(deep_detections):.1f} ± {np.std(deep_detections):.1f}\n"
        stats_text += f"  Min: {np.min(deep_detections)}, Max: {np.max(deep_detections)}\n"
        stats_text += f"  Time: {np.mean(deep_times):.3f}s avg\n\n"
        
        correlation = np.corrcoef(quick_detections, deep_detections)[0, 1]
        stats_text += f"Correlation: {correlation:.3f}\n"
        
        difference = np.array(deep_detections) - np.array(quick_detections)
        stats_text += f"Avg Difference: {np.mean(difference):+.1f}"
        
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, 
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax4.set_title('Statistical Summary', fontsize=12, fontweight='bold')
        ax4.axis('off')
        
        self.figure.tight_layout()
        self.canvas.draw()
        
        # Switch to Statistics tab
        self.tabs.setCurrentIndex(3)
    
    def _calculate_metrics(self, result, method_name):
        """Calculate precision, recall, F1-score for analysis result"""
        from collections import Counter
        
        detected = result.num_detections
        true_count = self.ground_truth_count if self.ground_truth_count > 0 else detected
        
        # Calculate metrics (simple approximation)
        if true_count > 0:
            # True Positives: minimum of detected and ground truth (conservative)
            tp = min(detected, true_count)
            # False Positives: extra detections
            fp = max(0, detected - true_count)
            # False Negatives: missed particles
            fn = max(0, true_count - detected)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        else:
            precision = recall = f1_score = 0
        
        # Extract distributions
        shape_dist = Counter([f['shape'] for f in result.features]) if result.features else {}
        color_dist = Counter([f['color'] for f in result.features]) if result.features else {}
        areas = [f['area'] for f in result.features] if result.features else []
        
        return {
            'method': method_name,
            'detected': detected,
            'ground_truth': true_count,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'processing_time': result.processing_time,
            'shape_distribution': dict(shape_dist),
            'color_distribution': dict(color_dist),
            'area_distribution': areas
        }
        
    def on_analysis_complete(self, result):
        """Handle analysis completion"""
        self.progress_bar.setVisible(False)
        
        if result is None:
            QMessageBox.warning(self, "Error", "Analysis failed")
            return
            
        self.current_result = result
        
        # Store result based on analysis type
        if result.analysis_type == 'quick':
            self.quick_result = result
        elif result.analysis_type == 'deep':
            self.deep_result = result
        
        # Display mask
        if result.mask is not None:
            mask_rgb = cv2.cvtColor(result.mask, cv2.COLOR_GRAY2RGB)
            self.mask_label.set_image(mask_rgb)
        
        # Create annotated image with ground truth comparison if available
        annotated = self.create_annotated_image(result)
        self.annotated_label.set_image(annotated)
        
        # Update results summary with ground truth comparison
        self.update_results_summary(result)
        
        # Update table
        self.update_results_table(result)
        
        # Update charts with full visualization based on dropdown selection
        self.update_charts_full()
        
        # Build status message with ground truth comparison
        status_msg = f'Analysis complete: {result.num_detections} particles in {result.processing_time:.2f}s'
        if self.ground_truth_count > 0:
            accuracy = (result.num_detections / self.ground_truth_count) * 100
            status_msg += f' | Ground Truth: {self.ground_truth_count} | Accuracy: {accuracy:.1f}%'
        
        self.statusBar().showMessage(status_msg)
        
    def create_annotated_image(self, result):
        """Create annotated image with detections"""
        annotated = self.current_image.copy()
        
        # Color map for shapes (grouped categories)
        color_map = {
            'Microbead/Pellet': (255, 0, 0),
            'Fiber/Filament': (0, 0, 255),

            'Fragment': (0, 255, 255),
            'Irregular': (128, 128, 128)
        }
        
        for feature in result.features:
            # Get bounding box
            x, y, w, h = feature['bounding_box']
            
            # Draw rectangle
            color = color_map.get(feature['shape'], (255, 255, 255))
            cv2.rectangle(annotated, (x, y), (x+w, y+h), color, 2)
            
            # Add label
            label = f"{feature['shape']}"
            cv2.putText(annotated, label, (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return annotated
        
    def update_results_summary(self, result):
        """Update results text summary"""
        text = f"Analysis Type: {result.analysis_type.upper()}\n"
        text += f"Processing Time: {result.processing_time:.3f} seconds\n"
        text += f"Detected Particles: {result.num_detections}\n"
        
        # Ground truth comparison
        if self.ground_truth_count > 0:
            text += f"\n{'='*40}\n"
            text += f"GROUND TRUTH COMPARISON\n"
            text += f"{'='*40}\n"
            text += f"Ground Truth: {self.ground_truth_count} particles\n"
            text += f"Detected: {result.num_detections} particles\n"
            
            # Calculate metrics
            detected = result.num_detections
            true_count = self.ground_truth_count
            
            # Simple accuracy (assumes 1:1 matching)
            accuracy = min(detected, true_count) / max(detected, true_count) * 100 if max(detected, true_count) > 0 else 0
            
            # Precision and Recall estimates
            # Precision: of detected particles, how many are real
            precision = min(detected, true_count) / detected if detected > 0 else 0
            # Recall: of real particles, how many were detected
            recall = min(detected, true_count) / true_count if true_count > 0 else 0
            # F1 Score
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            text += f"\nPrecision: {precision:.3f} ({precision*100:.1f}%)\n"
            text += f"Recall: {recall:.3f} ({recall*100:.1f}%)\n"
            text += f"F1-Score: {f1_score:.3f} ({f1_score*100:.1f}%)\n"
            text += f"Detection Rate: {accuracy:.1f}%\n"
            
            if detected > true_count:
                text += f"\n⚠ Over-detection: {detected - true_count} extra particles\n"
            elif detected < true_count:
                text += f"\n⚠ Under-detection: {true_count - detected} particles missed\n"
            else:
                text += f"\n✓ Perfect count match!\n"
            
            text += f"{'='*40}\n\n"
        
        text += f"Background: {result.background_color}\n"
        
        if hasattr(result, 'fiber_count'):
            text += f"Fiber Count: {result.fiber_count}\n"
        
        # Shape distribution
        from collections import Counter
        shapes = Counter([f['shape'] for f in result.features])
        text += "\nShape Distribution:\n"
        for shape, count in shapes.most_common():
            text += f"  {shape}: {count}\n"
        
        self.results_text.setText(text)
        
    def update_results_table(self, result):
        """Update results table"""
        features = result.features
        
        if not features:
            return
            
        # Get columns
        columns = ['ID', 'Shape', 'Color', 'Area', 'Circularity']
        if 'aspect_ratio' in features[0]:
            columns.extend(['Eccentricity', 'Aspect Ratio'])
        
        self.results_table.setColumnCount(len(columns))
        self.results_table.setHorizontalHeaderLabels(columns)
        self.results_table.setRowCount(len(features))
        
        for i, feature in enumerate(features):
            self.results_table.setItem(i, 0, QTableWidgetItem(str(feature['id'])))
            self.results_table.setItem(i, 1, QTableWidgetItem(feature['shape']))
            self.results_table.setItem(i, 2, QTableWidgetItem(feature['color']))
            self.results_table.setItem(i, 3, QTableWidgetItem(f"{feature['area']:.1f}"))
            self.results_table.setItem(i, 4, QTableWidgetItem(f"{feature['circularity']:.3f}"))
            
            if 'aspect_ratio' in feature:
                self.results_table.setItem(i, 5, QTableWidgetItem(f"{feature['eccentricity']:.3f}"))
                self.results_table.setItem(i, 6, QTableWidgetItem(f"{feature['aspect_ratio']:.2f}"))
        
        self.results_table.resizeColumnsToContents()
        
    def update_charts(self, result):
        """Update statistics charts (basic version)"""
        self.figure.clear()
        
        if not result.features:
            return
        
        # Shape distribution pie chart
        from collections import Counter
        shapes = Counter([f['shape'] for f in result.features])
        
        ax = self.figure.add_subplot(111)
        ax.pie(shapes.values(), labels=shapes.keys(), autopct='%1.1f%%')
        ax.set_title('Shape Distribution')
        
        self.canvas.draw()
        
    def update_charts_full(self):
        """Update statistics charts with comprehensive visualizations like original code"""
        if self.current_result is None or not self.current_result.features:
            return
            
        features = self.current_result.features
        chart_type = self.chart_type_combo.currentText()
        
        self.figure.clear()
        
        from collections import Counter
        import numpy as np
        
        if chart_type == 'All Charts':
            # Create 2x3 subplot layout (added 3D morphometric plot)
            
            # 1. Shape Distribution (Pie Chart)
            ax1 = self.figure.add_subplot(2, 3, 1)
            shapes = Counter([f['shape'] for f in features])
            
            # If ground truth data exists, create comparison bar chart instead of pie
            if self.ground_truth and len(self.ground_truth) > 0:
                # Map ground truth shapes to grouped names
                gt_shapes = Counter([map_to_grouped_shape(p['shape']) for p in self.ground_truth])
                all_shapes = sorted(set(list(shapes.keys()) + list(gt_shapes.keys())))
                
                x = np.arange(len(all_shapes))
                width = 0.35
                
                detected_counts = [shapes.get(shape, 0) for shape in all_shapes]
                gt_counts = [gt_shapes.get(shape, 0) for shape in all_shapes]
                
                ax1.bar(x - width/2, detected_counts, width, label=f'Detected (n={len(features)})', 
                       color='#4CAF50', alpha=0.8)
                ax1.bar(x + width/2, gt_counts, width, label=f'Ground Truth (n={len(self.ground_truth)})', 
                       color='#FF5722', alpha=0.8)
                
                ax1.set_ylabel('Count', fontsize=10, fontweight='bold')
                ax1.set_xticks(x)
                ax1.set_xticklabels(all_shapes, rotation=45, ha='right', fontsize=8)
                ax1.legend(fontsize=8)
                ax1.grid(True, alpha=0.3, axis='y')
                title_text = 'Shape Distribution Comparison'
            else:
                ax1.pie(shapes.values(), labels=shapes.keys(), autopct='%1.1f%%', startangle=90)
                title_text = f'Shape Distribution (n={len(features)})'
                if self.ground_truth_count > 0:
                    title_text += f'\nGround Truth: {self.ground_truth_count}'
            ax1.set_title(title_text)
            
            # 2. Size Distribution (Histogram)
            ax2 = self.figure.add_subplot(2, 3, 2)
            areas = [f['area'] for f in features]
            
            # Add ground truth areas if available
            if self.ground_truth and len(self.ground_truth) > 0:
                gt_areas = [p['area'] for p in self.ground_truth]
                all_areas = areas + gt_areas
                
                if all_areas:
                    bins = np.linspace(min(all_areas), max(all_areas), 15)
                    ax2.hist(areas, bins=bins, alpha=0.6, label=f'Detected (n={len(areas)})', 
                            color='#4CAF50', edgecolor='black')
                    ax2.hist(gt_areas, bins=bins, alpha=0.6, label=f'Ground Truth (n={len(gt_areas)})', 
                            color='#FF5722', edgecolor='black')
                    ax2.legend(fontsize=8)
                    title_text = 'Area Distribution Comparison'
            else:
                ax2.hist(areas, bins=20, color='skyblue', edgecolor='black')
                title_text = 'Size Distribution'
                if self.ground_truth_count > 0:
                    title_text += f' (GT: {self.ground_truth_count})'
            
            ax2.set_xlabel('Area (pixels)')
            ax2.set_ylabel('Frequency')
            ax2.set_title(title_text)
            ax2.grid(True, alpha=0.3)
            
            # 3. Color Distribution (4 fluorescent colors with ground truth)
            ax3 = self.figure.add_subplot(2, 3, 3)
            from collections import Counter
            
            # Get colors and map to 4 main colors
            colors_list = [f.get('color', 'Unknown') for f in features]
            main_colors = ['Red', 'Green', 'Blue', 'Yellow']
            color_counts = {c: 0 for c in main_colors}
            
            for color_name in colors_list:
                if 'Red' in color_name or 'Pink' in color_name or 'Orange' in color_name:
                    color_counts['Red'] += 1
                elif 'Green' in color_name or 'Cyan' in color_name:
                    color_counts['Green'] += 1
                elif 'Blue' in color_name or 'Purple' in color_name or 'Violet' in color_name:
                    color_counts['Blue'] += 1
                elif 'Yellow' in color_name:
                    color_counts['Yellow'] += 1
            
            # Visual colors
            visual_colors = {'Red': '#FF0000', 'Green': '#00FF00', 'Blue': '#0000FF', 'Yellow': '#FFFF00'}
            
            # Check if ground truth exists
            if self.ground_truth and len(self.ground_truth) > 0:
                # Get ground truth colors
                gt_colors_list = [p.get('color_label', 'Unknown') for p in self.ground_truth if 'color_label' in p]
                gt_color_counts = {c: 0 for c in main_colors}
                
                for color_name in gt_colors_list:
                    if 'Red' in color_name or 'Pink' in color_name or 'Orange' in color_name:
                        gt_color_counts['Red'] += 1
                    elif 'Green' in color_name or 'Cyan' in color_name:
                        gt_color_counts['Green'] += 1
                    elif 'Blue' in color_name or 'Purple' in color_name or 'Violet' in color_name:
                        gt_color_counts['Blue'] += 1
                    elif 'Yellow' in color_name:
                        gt_color_counts['Yellow'] += 1
                
                # Create grouped bar chart
                x = np.arange(len(main_colors))
                width = 0.35
                
                bars1 = ax3.bar(x - width/2, [color_counts[c] for c in main_colors], width,
                               label=f'Detected (n={len(features)})',
                               color=[visual_colors[c] for c in main_colors],
                               alpha=0.7, edgecolor='black', linewidth=1.5)
                bars2 = ax3.bar(x + width/2, [gt_color_counts[c] for c in main_colors], width,
                               label=f'Ground Truth (n={len(self.ground_truth)})',
                               color=[visual_colors[c] for c in main_colors],
                               alpha=0.4, edgecolor='black', linewidth=1.5, hatch='//')
                
                # Add count labels
                for bar in bars1:
                    height = bar.get_height()
                    if height > 0:
                        ax3.text(bar.get_x() + bar.get_width()/2., height,
                                f'{int(height)}', ha='center', va='bottom', fontsize=9, fontweight='bold')
                for bar in bars2:
                    height = bar.get_height()
                    if height > 0:
                        ax3.text(bar.get_x() + bar.get_width()/2., height,
                                f'{int(height)}', ha='center', va='bottom', fontsize=9, fontweight='bold')
                
                ax3.set_xticks(x)
                ax3.set_xticklabels(main_colors)
                ax3.legend(fontsize=8)
                title_text = 'Color Distribution Comparison'
            else:
                # No ground truth - single bar chart
                bars = ax3.bar(main_colors, [color_counts[c] for c in main_colors],
                              color=[visual_colors[c] for c in main_colors],
                              alpha=0.8, edgecolor='black', linewidth=2)
                
                # Add count labels on bars
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax3.text(bar.get_x() + bar.get_width()/2., height,
                                f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
                
                title_text = f'Color Distribution (n={len(features)})'
            
            ax3.set_ylabel('Count', fontsize=10, fontweight='bold')
            ax3.set_title(title_text, fontsize=10)
            ax3.grid(True, alpha=0.3, axis='y')
            
            # 4. 3D Morphometric Parameters (Circularity, Aspect Ratio, Eccentricity)
            ax4 = self.figure.add_subplot(2, 2, 4, projection='3d')
            colors_map = {'Microbead/Pellet': 'red', 'Fiber/Filament': 'blue',
                         'Fragment': 'cyan',
                         'Irregular': 'gray'}
            
            # Extract morphometric parameters
            circularities = [f.get('circularity', 0) for f in features]
            aspect_ratios = [f.get('aspect_ratio', 1) for f in features]
            eccentricities = [f.get('eccentricity', 0) for f in features]
            
            # Color by shape type
            for shape in set([f['shape'] for f in features]):
                shape_features = [f for f in features if f['shape'] == shape]
                x = [f.get('circularity', 0) for f in shape_features]
                y = [f.get('aspect_ratio', 1) for f in shape_features]
                z = [f.get('eccentricity', 0) for f in shape_features]
                ax4.scatter(x, y, z, label=shape, alpha=0.6,
                           c=colors_map.get(shape, 'black'), s=30)
            
            ax4.set_xlabel('Circularity', fontsize=9)
            ax4.set_ylabel('Aspect Ratio', fontsize=9)
            ax4.set_zlabel('Eccentricity', fontsize=9)
            title_text = '3D Morphometric Parameters'
            if self.ground_truth_count > 0:
                title_text += f' (n={len(features)})'
            ax4.set_title(title_text, fontsize=10)
            ax4.legend(loc='upper left', fontsize=7)
            ax4.view_init(elev=20, azim=45)
            
        elif chart_type == 'Shape Distribution':
            ax = self.figure.add_subplot(111)
            shapes = Counter([f['shape'] for f in features])
            
            # If ground truth exists, show comparison bar chart
            if self.ground_truth and len(self.ground_truth) > 0:
                # Map ground truth shapes to grouped names
                gt_shapes = Counter([map_to_grouped_shape(p['shape']) for p in self.ground_truth])
                all_shapes = sorted(set(list(shapes.keys()) + list(gt_shapes.keys())))
                
                x = np.arange(len(all_shapes))
                width = 0.35
                
                detected_counts = [shapes.get(shape, 0) for shape in all_shapes]
                gt_counts = [gt_shapes.get(shape, 0) for shape in all_shapes]
                
                bars1 = ax.bar(x - width/2, detected_counts, width, 
                              label=f'Detected ({len(features)} particles)', 
                              color='#4CAF50', alpha=0.8, edgecolor='black', linewidth=1.5)
                bars2 = ax.bar(x + width/2, gt_counts, width, 
                              label=f'Ground Truth ({len(self.ground_truth)} particles)', 
                              color='#FF5722', alpha=0.8, edgecolor='black', linewidth=1.5)
                
                ax.set_ylabel('Count', fontsize=12, fontweight='bold')
                ax.set_xlabel('Shape Type', fontsize=12, fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels(all_shapes, rotation=45, ha='right', fontsize=11)
                ax.legend(fontsize=11)
                ax.grid(True, alpha=0.3, axis='y')
                
                # Add value labels on bars
                for bars in [bars1, bars2]:
                    for bar in bars:
                        height = bar.get_height()
                        if height > 0:
                            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                                   f'{int(height)}', ha='center', va='bottom', fontsize=9)
                
                title_text = 'Shape Distribution Comparison'
            else:
                ax.pie(shapes.values(), labels=shapes.keys(), autopct='%1.1f%%', startangle=90)
                title_text = f'Shape Distribution\n(Detected: {len(features)} particles)'
                if self.ground_truth_count > 0:
                    title_text += f'\nGround Truth: {self.ground_truth_count}'
            ax.set_title(title_text)
            
        elif chart_type == 'Size Distribution':
            ax = self.figure.add_subplot(111)
            areas = [f['area'] for f in features]
            
            # Add ground truth if available
            if self.ground_truth and len(self.ground_truth) > 0:
                gt_areas = [p['area'] for p in self.ground_truth]
                all_areas = areas + gt_areas
                
                if all_areas:
                    bins = np.linspace(min(all_areas), max(all_areas), 25)
                    ax.hist(areas, bins=bins, alpha=0.6, label=f'Detected (n={len(areas)})', 
                           color='#4CAF50', edgecolor='black', linewidth=1.2)
                    ax.hist(gt_areas, bins=bins, alpha=0.6, label=f'Ground Truth (n={len(gt_areas)})', 
                           color='#FF5722', edgecolor='black', linewidth=1.2)
                    
                    # Add mean lines
                    mean_detected = np.mean(areas)
                    mean_gt = np.mean(gt_areas)
                    ax.axvline(mean_detected, color='#2E7D32', linestyle='--', linewidth=2, 
                              label=f'Detected Mean: {mean_detected:.1f}')
                    ax.axvline(mean_gt, color='#D32F2F', linestyle='--', linewidth=2, 
                              label=f'GT Mean: {mean_gt:.1f}')
                    
                    title_text = 'Particle Area Distribution Comparison'
            else:
                ax.hist(areas, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
                mean_area = np.mean(areas)
                median_area = np.median(areas)
                ax.axvline(mean_area, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_area:.1f}')
                ax.axvline(median_area, color='green', linestyle='--', linewidth=2, label=f'Median: {median_area:.1f}')
                title_text = f'Particle Size Distribution (n={len(features)})'
                if self.ground_truth_count > 0:
                    title_text += f'\nGround Truth: {self.ground_truth_count}'
            
            ax.set_xlabel('Area (pixels)', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title(title_text, fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)
            
        elif chart_type == 'Circularity Distribution':
            ax = self.figure.add_subplot(111)
            circularities = [f['circularity'] for f in features]
            
            # Add ground truth if available
            if self.ground_truth and len(self.ground_truth) > 0:
                gt_circs = [p.get('circularity', 0) for p in self.ground_truth if 'circularity' in p]
                if gt_circs:
                    bins = np.linspace(0, 1, 25)
                    ax.hist(circularities, bins=bins, alpha=0.6, label=f'Detected (n={len(circularities)})', 
                           color='#4CAF50', edgecolor='black', linewidth=1.2)
                    ax.hist(gt_circs, bins=bins, alpha=0.6, label=f'Ground Truth (n={len(gt_circs)})', 
                           color='#FF5722', edgecolor='black', linewidth=1.2)
                    
                    # Add mean lines
                    mean_detected = np.mean(circularities)
                    mean_gt = np.mean(gt_circs)
                    ax.axvline(mean_detected, color='#2E7D32', linestyle='--', linewidth=2, 
                              label=f'Detected Mean: {mean_detected:.3f}')
                    ax.axvline(mean_gt, color='#D32F2F', linestyle='--', linewidth=2, 
                              label=f'GT Mean: {mean_gt:.3f}')
                    
                    title_text = 'Circularity Distribution Comparison'
                else:
                    ax.hist(circularities, bins=30, color='lightcoral', edgecolor='black', alpha=0.7)
                    mean_circ = np.mean(circularities)
                    ax.axvline(mean_circ, color='red', linestyle='--', linewidth=2, 
                              label=f'Mean: {mean_circ:.3f}')
                    title_text = f'Circularity Distribution (n={len(features)})'
            else:
                ax.hist(circularities, bins=30, color='lightcoral', edgecolor='black', alpha=0.7)
                mean_circ = np.mean(circularities)
                ax.axvline(mean_circ, color='red', linestyle='--', linewidth=2, 
                          label=f'Mean: {mean_circ:.3f}')
                title_text = f'Circularity Distribution (n={len(features)})'
                if self.ground_truth_count > 0:
                    title_text += f'\nGround Truth: {self.ground_truth_count}'
            
            ax.set_xlabel('Circularity', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title(title_text, fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)
            
        elif chart_type == 'Color Distribution':
            # Single comprehensive color distribution chart
            from collections import Counter
            
            # Get colors and map to 4 main colors
            colors_list = [f.get('color', 'Unknown') for f in features]
            main_colors = ['Red', 'Green', 'Blue', 'Yellow']
            color_counts = {c: 0 for c in main_colors}
            
            for color_name in colors_list:
                if 'Red' in color_name or 'Pink' in color_name or 'Orange' in color_name:
                    color_counts['Red'] += 1
                elif 'Green' in color_name or 'Cyan' in color_name:
                    color_counts['Green'] += 1
                elif 'Blue' in color_name or 'Purple' in color_name or 'Violet' in color_name:
                    color_counts['Blue'] += 1
                elif 'Yellow' in color_name:
                    color_counts['Yellow'] += 1
            
            # Visual colors
            visual_colors = {'Red': '#FF0000', 'Green': '#00FF00', 'Blue': '#0000FF', 'Yellow': '#FFFF00'}
            
            ax = self.figure.add_subplot(111)
            
            # Check if ground truth exists
            if self.ground_truth and len(self.ground_truth) > 0:
                # Get ground truth colors
                gt_colors_list = [p.get('color_label', 'Unknown') for p in self.ground_truth if 'color_label' in p]
                gt_color_counts = {c: 0 for c in main_colors}
                
                for color_name in gt_colors_list:
                    if 'Red' in color_name or 'Pink' in color_name or 'Orange' in color_name:
                        gt_color_counts['Red'] += 1
                    elif 'Green' in color_name or 'Cyan' in color_name:
                        gt_color_counts['Green'] += 1
                    elif 'Blue' in color_name or 'Purple' in color_name or 'Violet' in color_name:
                        gt_color_counts['Blue'] += 1
                    elif 'Yellow' in color_name:
                        gt_color_counts['Yellow'] += 1
                
                # Create grouped bar chart with ground truth comparison
                x = np.arange(len(main_colors))
                width = 0.35
                
                bars1 = ax.bar(x - width/2, [color_counts[c] for c in main_colors], width,
                              label=f'Detected (n={len(features)})',
                              color=[visual_colors[c] for c in main_colors],
                              alpha=0.8, edgecolor='black', linewidth=2)
                bars2 = ax.bar(x + width/2, [gt_color_counts[c] for c in main_colors], width,
                              label=f'Ground Truth (n={len(self.ground_truth)})',
                              color=[visual_colors[c] for c in main_colors],
                              alpha=0.5, edgecolor='black', linewidth=2, hatch='//')
                
                # Add count labels on bars
                for bar in bars1:
                    height = bar.get_height()
                    if height > 0:
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{int(height)}', ha='center', va='bottom', 
                               fontsize=13, fontweight='bold')
                for bar in bars2:
                    height = bar.get_height()
                    if height > 0:
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{int(height)}', ha='center', va='bottom', 
                               fontsize=13, fontweight='bold')
                
                ax.set_xticks(x)
                ax.set_xticklabels(main_colors, fontsize=14, fontweight='bold')
                ax.legend(fontsize=12, loc='upper right')
                title_text = 'Color Distribution Comparison\n(4 Fluorescent Colors)'
            else:
                # No ground truth - single bar chart
                bars = ax.bar(main_colors, [color_counts[c] for c in main_colors],
                             color=[visual_colors[c] for c in main_colors],
                             alpha=0.8, edgecolor='black', linewidth=2.5)
                
                # Add count labels and percentage
                total = sum(color_counts.values())
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        percentage = (height / total * 100) if total > 0 else 0
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{int(height)}\n({percentage:.1f}%)', 
                               ha='center', va='bottom', 
                               fontsize=14, fontweight='bold')
                
                title_text = f'Color Distribution (4 Fluorescent Colors)\nTotal Particles: {len(features)}'
            
            ax.set_ylabel('Count', fontsize=14, fontweight='bold')
            ax.set_xlabel('Fluorescent Color', fontsize=14, fontweight='bold')
            ax.set_title(title_text, fontsize=16, fontweight='bold', pad=20)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Set y-axis limit with some headroom
            max_count = max(max(color_counts.values()), 
                           max([gt_color_counts[c] for c in main_colors]) if self.ground_truth else 0)
            ax.set_ylim(0, max_count * 1.2 if max_count > 0 else 1)
            
        elif chart_type == 'Scatter Plots':
            # Create 2x2 scatter plot matrix
            ax1 = self.figure.add_subplot(2, 2, 1)
            ax2 = self.figure.add_subplot(2, 2, 2)
            ax3 = self.figure.add_subplot(2, 2, 3)
            ax4 = self.figure.add_subplot(2, 2, 4)
            
            colors_map = {'Microbead/Pellet': 'red', 'Fiber/Filament': 'blue',
                         'Fragment': 'cyan',
                         'Irregular': 'gray'}
            
            for shape in set([f['shape'] for f in features]):
                shape_features = [f for f in features if f['shape'] == shape]
                areas = [f['area'] for f in shape_features]
                circs = [f['circularity'] for f in shape_features]
                
                color = colors_map.get(shape, 'black')
                
                # Area vs Circularity
                ax1.scatter(areas, circs, label=shape, alpha=0.6, c=color, s=50)
                
                # If we have eccentricity and aspect_ratio
                if 'eccentricity' in shape_features[0]:
                    eccs = [f['eccentricity'] for f in shape_features]
                    ax2.scatter(areas, eccs, alpha=0.6, c=color, s=50)
                    ax3.scatter(circs, eccs, alpha=0.6, c=color, s=50)
                
                if 'aspect_ratio' in shape_features[0]:
                    aspects = [f['aspect_ratio'] for f in shape_features]
                    ax4.scatter(areas, aspects, alpha=0.6, c=color, s=50)
            
            ax1.set_xlabel('Area')
            ax1.set_ylabel('Circularity')
            title_text = f'Area vs Circularity (n={len(features)})'
            if self.ground_truth_count > 0:
                title_text += f'\nGT: {self.ground_truth_count}'
            ax1.set_title(title_text)
            ax1.legend(fontsize=8)
            ax1.grid(True, alpha=0.3)
            
            ax2.set_xlabel('Area')
            ax2.set_ylabel('Eccentricity')
            ax2.set_title('Area vs Eccentricity')
            ax2.grid(True, alpha=0.3)
            
            ax3.set_xlabel('Circularity')
            ax3.set_ylabel('Eccentricity')
            ax3.set_title('Circularity vs Eccentricity')
            ax3.grid(True, alpha=0.3)
            
            ax4.set_xlabel('Area')
            ax4.set_ylabel('Aspect Ratio')
            ax4.set_title('Area vs Aspect Ratio')
            ax4.grid(True, alpha=0.3)
        
        self.figure.tight_layout()
        self.canvas.draw()
        
    def zoom_in_image(self):
        """Zoom in the original image"""
        self.image_label.zoom_in()
        self.update_zoom_label()
        
    def zoom_out_image(self):
        """Zoom out the original image"""
        self.image_label.zoom_out()
        self.update_zoom_label()
        
    def reset_zoom_image(self):
        """Reset zoom to 100%"""
        self.image_label.reset_zoom()
        self.update_zoom_label()
        
    def toggle_zoom_box(self, checked):
        """Toggle box zoom mode"""
        self.image_label.enable_zoom_box(checked)
        if checked:
            self.statusBar().showMessage("Box Zoom: Left-click and drag to zoom into area, Right-click to zoom out")
        else:
            self.statusBar().showMessage("Box Zoom disabled")
            
    def update_zoom_label(self):
        """Update zoom percentage label"""
        zoom_percent = int(self.image_label.zoom_factor * 100)
        self.zoom_level_label.setText(f"Zoom: {zoom_percent}%")
    
    # Zoom functions for Annotated Results view
    def zoom_in_annotated(self):
        """Zoom in the annotated image"""
        self.annotated_label.zoom_in()
        self.update_annotated_zoom_label()
        
    def zoom_out_annotated(self):
        """Zoom out the annotated image"""
        self.annotated_label.zoom_out()
        self.update_annotated_zoom_label()
        
    def reset_zoom_annotated(self):
        """Reset annotated image zoom to 100%"""
        self.annotated_label.reset_zoom()
        self.update_annotated_zoom_label()
        
    def toggle_zoom_box_annotated(self, checked):
        """Toggle box zoom mode for annotated view"""
        self.annotated_label.enable_zoom_box(checked)
        if checked:
            self.statusBar().showMessage("Annotated Box Zoom: Left-click and drag to zoom into area, Right-click to zoom out, Ctrl+Wheel to zoom")
        else:
            self.statusBar().showMessage("Box Zoom disabled")
            
    def update_annotated_zoom_label(self):
        """Update annotated zoom percentage label"""
        zoom_percent = int(self.annotated_label.zoom_factor * 100)
        self.annotated_zoom_level_label.setText(f"Zoom: {zoom_percent}%")
        
    def save_results(self):
        """Save analysis results"""
        if self.current_result is None:
            QMessageBox.warning(self, "Warning", "No results to save")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Results", "", 
            "JSON Files (*.json);;CSV Files (*.csv);;All Files (*)"
        )
        
        if file_path:
            self.export_results_to_file(file_path)
            
    def export_results(self, format_type):
        """Export results in specified format"""
        if self.current_result is None:
            QMessageBox.warning(self, "Warning", "No results to export")
            return
            
        if format_type == 'json':
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Export JSON", "", "JSON Files (*.json)"
            )
        else:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Export CSV", "", "CSV Files (*.csv)"
            )
            
        if file_path:
            self.export_results_to_file(file_path)
            
    def export_results_to_file(self, file_path):
        """Export results to file"""
        try:
            if file_path.endswith('.json'):
                # Export as JSON
                data = {
                    'num_detections': self.current_result.num_detections,
                    'processing_time': self.current_result.processing_time,
                    'background_color': self.current_result.background_color,
                    'features': [
                        {
                            'id': f['id'],
                            'shape': f['shape'],
                            'color': f['color'],
                            'area': f['area'],
                            'circularity': f['circularity'],
                            'centroid': f['centroid']
                        }
                        for f in self.current_result.features
                    ]
                }
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2)
                    
            elif file_path.endswith('.csv'):
                # Export as CSV
                import pandas as pd
                df = pd.DataFrame(self.current_result.features)
                df.to_csv(file_path, index=False)
                
            QMessageBox.information(self, "Success", f"Results exported to {file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export: {str(e)}")
            
    def generate_synthetic(self):
        """Generate synthetic data (batch)"""
        dialog = SyntheticDataDialog(self)
        dialog.exec_()
    
    def generate_synthetic_single(self):
        """Generate a single synthetic image and load it"""
        try:
            self.statusBar().showMessage('Generating synthetic image...')
            
            # Get synthetic parameters from UI
            params = SyntheticImageParams(
                num_particles=self.synth_num_particles.value(),
                shape_type=self.synth_shape_type.currentText(),
                color_type=self.synth_color_type.currentText(),
                enable_blur=self.synth_enable_blur.isChecked(),
                blur_kernel=self.synth_blur_kernel.value(),
                enable_glow=self.synth_enable_glow.isChecked(),
                glow_intensity=self.synth_glow_intensity.value(),
                glow_sigma_min=self.synth_glow_sigma_min.value(),
                glow_sigma_max=self.synth_glow_sigma_max.value(),
                background_noise_min=self.synth_bg_noise_min.value(),
                background_noise_max=self.synth_bg_noise_max.value(),
                image_width=self.synth_width.value(),
                image_height=self.synth_height.value(),
                background_brightness=self.synth_brightness.value(),
                particle_brightness=self.synth_particle_brightness.value()
            )
            
            # Generate image
            generator = SyntheticImageGenerator()
            image, ground_truth = generator.generate(params)
            
            # Load the generated image
            self.current_image = image
            self.image_label.set_image(image)
            
            # Store ground truth
            self.ground_truth = ground_truth
            self.ground_truth_count = len(ground_truth)
            
            # Enable analysis buttons
            self.quick_btn.setEnabled(True)
            self.deep_btn.setEnabled(True)
            self.both_btn.setEnabled(True)
            
            self.statusBar().showMessage(f'Generated synthetic image with {len(ground_truth)} particles (Ground Truth)')
            
            # Show ground truth info
            gt_text = f"Generated {len(ground_truth)} particles:\n"
            from collections import Counter
            # Map ground truth shapes to grouped names for display
            shapes = Counter([map_to_grouped_shape(p['shape']) for p in ground_truth])
            for shape, count in shapes.items():
                gt_text += f"  {shape}: {count}\n"
            
            QMessageBox.information(self, "Synthetic Image Generated", gt_text)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate: {str(e)}")
    
    def export_yolo_dataset(self):
        """Export YOLO training dataset with synthetic images and annotations"""
        try:
            from pathlib import Path
            from src.data_generation.yolo_exporter import YOLODatasetExporter, print_dataset_stats
            
            # Ask user for dataset configuration
            from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QSpinBox, QDialogButtonBox, QComboBox, QCheckBox
            
            dialog = QDialog(self)
            dialog.setWindowTitle("Export YOLO Training Dataset")
            dialog.setMinimumWidth(450)
            
            layout = QVBoxLayout()
            
            # Description
            desc = QLabel("Generate synthetic microplastic images with YOLOv8 format annotations.\n\n"
                         "This will create a dataset structure with:\n"
                         "  • images/train/, images/val/, images/test/\n"
                         "  • labels/train/, labels/val/, labels/test/\n"
                         "  • dataset.yaml configuration file")
            desc.setWordWrap(True)
            layout.addWidget(desc)
            
            # Number of training images
            train_label = QLabel("Training Images:")
            layout.addWidget(train_label)
            train_spin = QSpinBox()
            train_spin.setRange(10, 10000)
            train_spin.setValue(500)
            train_spin.setToolTip("Number of synthetic images to generate for training")
            layout.addWidget(train_spin)
            
            # Number of validation images
            val_label = QLabel("Validation Images:")
            layout.addWidget(val_label)
            val_spin = QSpinBox()
            val_spin.setRange(10, 5000)
            val_spin.setValue(100)
            val_spin.setToolTip("Number of synthetic images to generate for validation")
            layout.addWidget(val_spin)
            
            # Number of test images
            test_label = QLabel("Test Images:")
            layout.addWidget(test_label)
            test_spin = QSpinBox()
            test_spin.setRange(0, 5000)
            test_spin.setValue(100)
            test_spin.setToolTip("Number of synthetic images to generate for testing (final evaluation)")
            layout.addWidget(test_spin)
            
            # Use current synthetic parameters checkbox
            use_current_params = QCheckBox("Use Current Synthetic Parameters")
            use_current_params.setChecked(True)
            use_current_params.setToolTip("Use the parameters currently set in the Synthetic Image Parameters section")
            layout.addWidget(use_current_params)
            
            # Append mode checkbox
            layout.addWidget(QLabel(""))  # Spacing
            append_mode_checkbox = QCheckBox("Append to Existing Dataset (Don't Overwrite)")
            append_mode_checkbox.setChecked(False)
            append_mode_checkbox.setToolTip(
                "If checked, new images will be added to the existing dataset\n"
                "with sequential numbering (e.g., if you have train_00500,\n"
                "new images will start from train_00501).\n\n"
                "If unchecked, numbering starts from 0 (may overwrite existing files)."
            )
            append_mode_checkbox.setStyleSheet("QCheckBox { color: #0066cc; font-weight: bold; }")
            layout.addWidget(append_mode_checkbox)
            
            # Buttons
            buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            buttons.accepted.connect(dialog.accept)
            buttons.rejected.connect(dialog.reject)
            layout.addWidget(buttons)
            
            dialog.setLayout(layout)
            
            if dialog.exec_() != QDialog.Accepted:
                return
            
            num_train = train_spin.value()
            num_val = val_spin.value()
            num_test = test_spin.value()
            append_mode = append_mode_checkbox.isChecked()
            
            # Ask for output directory
            output_dir = QFileDialog.getExistingDirectory(self, "Select Output Directory for YOLO Dataset")
            if not output_dir:
                return
            
            output_path = Path(output_dir) / 'yolo_microplastic_dataset'
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Check for existing files if append mode is enabled
            train_start_idx = 0
            val_start_idx = 0
            test_start_idx = 0
            
            if append_mode:
                # Count existing files
                train_images_dir = output_path / 'images' / 'train'
                val_images_dir = output_path / 'images' / 'val'
                test_images_dir = output_path / 'images' / 'test'
                
                if train_images_dir.exists():
                    train_start_idx = len(list(train_images_dir.glob('train_*.png')))
                if val_images_dir.exists():
                    val_start_idx = len(list(val_images_dir.glob('val_*.png')))
                if test_images_dir.exists():
                    test_start_idx = len(list(test_images_dir.glob('test_*.png')))
                
                if train_start_idx > 0 or val_start_idx > 0 or test_start_idx > 0:
                    self.results_text.append("\n" + "="*60)
                    self.results_text.append("APPEND MODE: Adding to existing dataset")
                    self.results_text.append("="*60)
                    if train_start_idx > 0:
                        self.results_text.append(f"  Existing training images: {train_start_idx}")
                        self.results_text.append(f"  New training images will start from train_{train_start_idx:05d}")
                    if val_start_idx > 0:
                        self.results_text.append(f"  Existing validation images: {val_start_idx}")
                        self.results_text.append(f"  New validation images will start from val_{val_start_idx:05d}")
                    if test_start_idx > 0:
                        self.results_text.append(f"  Existing test images: {test_start_idx}")
                        self.results_text.append(f"  New test images will start from test_{test_start_idx:05d}")
                    self.results_text.append("="*60)
                else:
                    self.results_text.append("\n⚠ Append mode enabled but no existing files found. Starting from 0.")
            
            self.statusBar().showMessage('Exporting YOLO dataset...')
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, num_train + num_val + num_test)
            
            # Get synthetic parameters
            if use_current_params.isChecked():
                params = SyntheticImageParams(
                    num_particles=self.synth_num_particles.value(),
                    shape_type=self.synth_shape_type.currentText(),
                    color_type=self.synth_color_type.currentText(),
                    enable_blur=self.synth_enable_blur.isChecked(),
                    blur_kernel=self.synth_blur_kernel.value(),
                    enable_glow=self.synth_enable_glow.isChecked(),
                    glow_intensity=self.synth_glow_intensity.value(),
                    glow_sigma_min=self.synth_glow_sigma_min.value(),
                    glow_sigma_max=self.synth_glow_sigma_max.value(),
                    background_noise_min=self.synth_bg_noise_min.value(),
                    background_noise_max=self.synth_bg_noise_max.value(),
                    image_width=self.synth_width.value(),
                    image_height=self.synth_height.value(),
                    background_brightness=self.synth_brightness.value(),
                    particle_brightness=self.synth_particle_brightness.value()
                )
            else:
                # Use default parameters
                params = SyntheticImageParams()
            
            generator = SyntheticImageGenerator()
            
            # Create directory structure first
            images_train_dir = output_path / 'images' / 'train'
            labels_train_dir = output_path / 'labels' / 'train'
            ground_truth_train_dir = output_path / 'ground_truth' / 'train'
            images_train_dir.mkdir(parents=True, exist_ok=True)
            labels_train_dir.mkdir(parents=True, exist_ok=True)
            ground_truth_train_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate and save training set (one by one)
            self.results_text.append("\n=== Generating Training Set ===")
            if append_mode and train_start_idx > 0:
                self.results_text.append(f"Starting from index {train_start_idx}")
            train_count = 0
            for i in range(num_train):
                image, ground_truth = generator.generate(params)
                image_idx = train_start_idx + i  # Use offset for append mode
                name = f'train_{image_idx:05d}'
                
                # Save immediately to disk
                YOLODatasetExporter.save_single_image(
                    image, ground_truth, name, 
                    images_train_dir, labels_train_dir, ground_truth_train_dir
                )
                train_count += 1
                
                self.progress_bar.setValue(i + 1)
                self.statusBar().showMessage(f'Generated & saved training image {i+1}/{num_train}')
                if (i + 1) % 10 == 0:  # Update text every 10 images
                    self.results_text.append(f"  Saved {i+1}/{num_train} images...")
                QApplication.processEvents()
            
            self.results_text.append(f"✓ Saved {train_count} training images")
            
            # Create validation directories
            images_val_dir = output_path / 'images' / 'val'
            labels_val_dir = output_path / 'labels' / 'val'
            ground_truth_val_dir = output_path / 'ground_truth' / 'val'
            images_val_dir.mkdir(parents=True, exist_ok=True)
            labels_val_dir.mkdir(parents=True, exist_ok=True)
            ground_truth_val_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate and save validation set (one by one)
            self.results_text.append("\n=== Generating Validation Set ===")
            if append_mode and val_start_idx > 0:
                self.results_text.append(f"Starting from index {val_start_idx}")
            val_count = 0
            for i in range(num_val):
                image, ground_truth = generator.generate(params)
                image_idx = val_start_idx + i  # Use offset for append mode
                name = f'val_{image_idx:05d}'
                
                # Save immediately to disk
                YOLODatasetExporter.save_single_image(
                    image, ground_truth, name, 
                    images_val_dir, labels_val_dir, ground_truth_val_dir
                )
                val_count += 1
                
                self.progress_bar.setValue(num_train + i + 1)
                self.statusBar().showMessage(f'Generated & saved validation image {i+1}/{num_val}')
                if (i + 1) % 10 == 0:  # Update text every 10 images
                    self.results_text.append(f"  Saved {i+1}/{num_val} images...")
                QApplication.processEvents()
            
            self.results_text.append(f"✓ Saved {val_count} validation images")
            
            # Generate and save test set (if requested, one by one)
            test_count = 0
            if num_test > 0:
                # Create test directories
                images_test_dir = output_path / 'images' / 'test'
                labels_test_dir = output_path / 'labels' / 'test'
                ground_truth_test_dir = output_path / 'ground_truth' / 'test'
                images_test_dir.mkdir(parents=True, exist_ok=True)
                labels_test_dir.mkdir(parents=True, exist_ok=True)
                ground_truth_test_dir.mkdir(parents=True, exist_ok=True)
                
                self.results_text.append("\n=== Generating Test Set ===")
                if append_mode and test_start_idx > 0:
                    self.results_text.append(f"Starting from index {test_start_idx}")
                for i in range(num_test):
                    image, ground_truth = generator.generate(params)
                    image_idx = test_start_idx + i  # Use offset for append mode
                    name = f'test_{image_idx:05d}'
                    
                    # Save immediately to disk
                    YOLODatasetExporter.save_single_image(
                        image, ground_truth, name, 
                        images_test_dir, labels_test_dir, ground_truth_test_dir
                    )
                    test_count += 1
                    
                    self.progress_bar.setValue(num_train + num_val + i + 1)
                    self.statusBar().showMessage(f'Generated & saved test image {i+1}/{num_test}')
                    if (i + 1) % 10 == 0:  # Update text every 10 images
                        self.results_text.append(f"  Saved {i+1}/{num_test} images...")
                    QApplication.processEvents()
                
                self.results_text.append(f"✓ Saved {test_count} test images")
            
            # Create dataset.yaml
            yaml_path = YOLODatasetExporter.create_dataset_yaml(output_path)
            self.results_text.append(f"✓ Created dataset configuration: {yaml_path.name}")
            
            # Show total dataset size if in append mode
            if append_mode:
                total_train = train_start_idx + train_count
                total_val = val_start_idx + val_count
                total_test = test_start_idx + test_count
                self.results_text.append("\n" + "="*60)
                self.results_text.append("Dataset Summary (After Appending):")
                self.results_text.append("="*60)
                self.results_text.append(f"  Total training images: {total_train} (Added {train_count} new)")
                self.results_text.append(f"  Total validation images: {total_val} (Added {val_count} new)")
                if test_count > 0:
                    self.results_text.append(f"  Total test images: {total_test} (Added {test_count} new)")
                self.results_text.append("="*60)
            
            # Get and display statistics
            stats = YOLODatasetExporter.get_dataset_stats(output_path)
            self.results_text.append("\n=== Dataset Statistics ===")
            for split in ['train', 'val', 'test']:
                if stats[split]['total'] > 0:
                    self.results_text.append(f"\n{split.upper()}:")
                    self.results_text.append(f"  Images: {stats[split]['total']}")
                    for class_name, count in stats[split]['classes'].items():
                        if count > 0:
                            self.results_text.append(f"    {class_name}: {count}")
            
            self.progress_bar.setVisible(False)
            self.statusBar().showMessage(f'YOLO dataset exported successfully to {output_path}')
            
            # Show success message with next steps
            if append_mode:
                total_train = train_start_idx + train_count
                total_val = val_start_idx + val_count
                total_test = test_start_idx + test_count
                success_msg = (f"✓ YOLO dataset updated successfully!\n\n"
                              f"Location: {output_path}\n\n"
                              f"New Training images: {train_count}\n"
                              f"New Validation images: {val_count}\n"
                              f"New Test images: {test_count}\n\n"
                              f"TOTAL Training images: {total_train}\n"
                              f"TOTAL Validation images: {total_val}\n"
                              f"TOTAL Test images: {total_test}\n\n"
                              f"✓ Old data preserved - files numbered sequentially!\n\n")
            else:
                success_msg = (f"✓ YOLO dataset exported successfully!\n\n"
                              f"Location: {output_path}\n\n"
                              f"Training images: {train_count}\n"
                              f"Validation images: {val_count}\n"
                              f"Test images: {test_count}\n\n")
            
            success_msg += (f"Next steps:\n"
                           f"1. Review the dataset.yaml file\n"
                           f"2. Train your model using YOLOv8:\n"
                           f"   yolo detect train data={yaml_path.name} model=yolov8s.pt epochs=100\n"
                           f"3. Evaluate on test set:\n"
                           f"   yolo detect val data={yaml_path.name} model=runs/detect/train/weights/best.pt split=test\n\n"
                           f"See YOLO_TRAINING_GUIDE.md for detailed instructions.")
            
            title = "YOLO Dataset Updated" if append_mode else "YOLO Dataset Exported"
            QMessageBox.information(self, title, success_msg)
            
        except Exception as e:
            import traceback
            self.results_text.append(f"\n✗ Error exporting YOLO dataset: {str(e)}")
            self.results_text.append(traceback.format_exc())
            QMessageBox.critical(self, "Error", f"Failed to export YOLO dataset:\n{str(e)}")
            self.progress_bar.setVisible(False)
            
    def load_yolo_model(self):
        """Load YOLO model for ML detection"""
        if not YOLO_AVAILABLE:
            self._show_yolo_installation_guide()
            return
            
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load YOLO Model", "", 
            "Model Files (*.pt *.pth);;All Files (*)"
        )
        
        if file_path:
            try:
                self.statusBar().showMessage('Loading YOLO model...')
                self.yolo_model = YOLO(file_path)
                self.yolo_model_path = file_path
                
                model_name = Path(file_path).name
                self.ml_model_label.setText(f"Model: {model_name}")
                self.ml_model_label.setStyleSheet("color: green; font-weight: bold;")
                self.run_ml_btn.setEnabled(True)
                self.run_ml_benchmark_btn.setEnabled(True)
                
                self.statusBar().showMessage(f'Loaded model: {model_name}')
                QMessageBox.information(self, "Success", f"YOLO model loaded:\n{model_name}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load model:\n{str(e)}")
                self.ml_model_label.setText("Model load failed")
                self.ml_model_label.setStyleSheet("color: red;")
    
    def _show_yolo_installation_guide(self):
        """Show detailed installation guide for YOLO"""
        import sys
        py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        
        if sys.version_info >= (3, 13):
            title = f"YOLO Not Available (Python {py_version})"
            message = (
                f"<h3>PyTorch doesn't support Python 3.13 yet</h3>"
                f"<p>YOLO requires PyTorch, which currently supports Python 3.8-3.12.</p>"
                f"<h4>Solutions:</h4>"
                f"<ol>"
                f"<li><b>Use Python 3.11 (Recommended)</b><br/>"
                f"   • Download: <a href='https://www.python.org/downloads/'>python.org/downloads</a><br/>"
                f"   • Install packages: <code>pip install ultralytics torch torchvision</code><br/>"
                f"   • Run app: <code>py -3.11 main.py</code></li>"
                f"<li><b>Create Virtual Environment with Python 3.11</b><br/>"
                f"   • Windows: <code>py -3.11 -m venv venv_ml</code><br/>"
                f"   • Activate: <code>venv_ml\\Scripts\\activate</code><br/>"
                f"   • Install: <code>pip install ultralytics torch torchvision</code></li>"
                f"<li><b>Wait for PyTorch Update</b><br/>"
                f"   Monitor: <a href='https://pytorch.org/'>pytorch.org</a> for Python 3.13 support</li>"
                f"</ol>"
                f"<p><b>Note:</b> Quick and Deep analysis work perfectly without YOLO.</p>"
            )
        else:
            title = "YOLO Not Installed"
            message = (
                f"<h3>YOLO/PyTorch is not installed</h3>"
                f"<h4>Installation Steps:</h4>"
                f"<ol>"
                f"<li><b>Install ultralytics (includes YOLO)</b><br/>"
                f"   <code>pip install ultralytics</code></li>"
                f"<li><b>Install PyTorch</b><br/>"
                f"   Visit <a href='https://pytorch.org/get-started/locally/'>pytorch.org</a><br/>"
                f"   Or use: <code>pip install torch torchvision</code></li>"
                f"<li><b>Restart the application</b></li>"
                f"</ol>"
                f"<p><b>Your Python version ({py_version}) is compatible!</b></p>"
            )
        
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle(title)
        msg_box.setTextFormat(Qt.RichText)
        msg_box.setText(message)
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()
                
    def run_yolo_detection(self):
        """Run YOLO detection on current image"""
        if self.current_image is None:
            QMessageBox.warning(self, "Warning", "Please load an image first")
            return
            
        if self.yolo_model is None:
            QMessageBox.warning(self, "Warning", "Please load a YOLO model first")
            return
            
        try:
            import time
            start_time = time.time()
            
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)
            self.statusBar().showMessage('Running YOLO detection...')
            
            # Run detection
            results = self.yolo_model(self.current_image)
            
            # Extract detections and create features for statistics
            detections = []
            features = []
            annotated_image = self.current_image.copy()
            
            for result in results:
                boxes = result.boxes
                for i, box in enumerate(boxes):
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    
                    # Get class name
                    class_name = result.names[cls] if cls < len(result.names) else f"Class_{cls}"
                    
                    # Calculate bbox dimensions
                    bbox_w = int(x2 - x1)
                    bbox_h = int(y2 - y1)
                    bbox_area = bbox_w * bbox_h
                    
                    # Store detection
                    detections.append({
                        'id': i + 1,
                        'class': class_name,
                        'confidence': float(conf),
                        'bbox': [int(x1), int(y1), bbox_w, bbox_h],
                        'center': [int((x1+x2)/2), int((y1+y2)/2)]
                    })
                    
                    # Create feature for statistics
                    feature = {
                        'id': i + 1,
                        'shape': class_name,  # Use YOLO class as shape
                        'ml_class': class_name,
                        'ml_confidence': float(conf),
                        'area': bbox_area,
                        'circularity': 0.5,  # Default value for YOLO detections
                        'aspect_ratio': bbox_w / bbox_h if bbox_h > 0 else 1.0,
                        'color': 'Unknown',  # YOLO doesn't detect color
                        'centroid': [int((x1+x2)/2), int((y1+y2)/2)],
                        'bounding_box': [int(x1), int(y1), bbox_w, bbox_h],
                        'bbox_global': [int(x1), int(y1), bbox_w, bbox_h]
                    }
                    features.append(feature)
                    
                    # Draw on image
                    color = (0, 255, 0) if conf > 0.5 else (255, 165, 0)
                    cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    label = f"{class_name}: {conf:.2f}"
                    cv2.putText(annotated_image, label, (int(x1), int(y1)-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            processing_time = time.time() - start_time
            
            # Create AnalysisResult object for statistics
            from config.settings import AnalysisResult, PreprocessingParams
            ml_result = AnalysisResult(
                features=features,
                mask=None,
                processing_time=processing_time,
                params=PreprocessingParams(),
                analysis_type='ml_yolo',
                num_detections=len(features),
                image_dimensions=(self.current_image.shape[1], self.current_image.shape[0])
            )
            
            # Store result for statistics
            self.ml_result = ml_result
            self.current_result = ml_result
            
            # Display annotated image
            self.annotated_label.set_image(annotated_image)
            self.tabs.setCurrentIndex(2)  # Switch to annotated tab
            
            # Update results
            result_text = f"YOLO Detection Results\n"
            result_text += f"Model: {Path(self.yolo_model_path).name}\n"
            result_text += f"Detections: {len(detections)}\n"
            result_text += f"Processing Time: {processing_time:.2f}s\n\n"
            
            from collections import Counter
            classes = Counter([d['class'] for d in detections])
            result_text += "Class Distribution:\n"
            for cls, count in classes.most_common():
                result_text += f"  {cls}: {count}\n"
            
            self.results_text.setText(result_text)
            
            # Update table with ML detections
            self.update_ml_results_table(detections)
            
            # Update statistics charts with YOLO results
            self.update_charts_full()
            
            self.progress_bar.setVisible(False)
            self.statusBar().showMessage(f'ML Detection complete: {len(detections)} objects detected')
            
        except Exception as e:
            self.progress_bar.setVisible(False)
            QMessageBox.critical(self, "Error", f"YOLO detection failed:\n{str(e)}")
            
    def run_ml_benchmark(self):
        """Run ML Benchmark analysis with full shape/color metrics"""
        if self.yolo_model is None:
            QMessageBox.warning(self, "Warning", "Please load a YOLO model first")
            return
        
        # Ask user: single image or batch processing
        from PyQt5.QtWidgets import QInputDialog
        
        items = ["Single Image (Current)", "Batch Processing (Folder)"]
        item, ok = QInputDialog.getItem(self, "ML Benchmark Mode", 
                                       "Select benchmark mode:", items, 0, False)
        
        if not ok:
            return
        
        if "Single" in item:
            self._run_single_ml_benchmark()
        else:
            self._run_batch_ml_benchmark()
    
    def _run_single_ml_benchmark(self):
        """Run ML Benchmark on single current image"""
        if self.current_image is None:
            QMessageBox.warning(self, "Warning", "Please load an image first")
            return
            
        try:
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)
            self.statusBar().showMessage('Running ML Benchmark analysis...')
            
            # Run ML Benchmark analysis
            params = self.get_analysis_params('deep')  # Use deep params for consistency
            ml_analyzer = MLBenchmarkAnalyzer(self.yolo_model)
            ml_result = ml_analyzer.analyze(self.current_image, params)
            
            # Store result
            self.ml_result = ml_result
            self.current_result = ml_result
            
            # Create annotated visualization
            annotated_image = self.current_image.copy()
            for feature in ml_result.features:
                bbox = feature.get('bbox_global', feature.get('bounding_box'))
                if bbox:
                    x, y, w, h = bbox
                    # Color code by confidence if available
                    conf = feature.get('ml_confidence', 1.0)
                    color = (0, 255, 0) if conf > 0.5 else (255, 165, 0)
                    cv2.rectangle(annotated_image, (x, y), (x+w, y+h), color, 2)
                    
                    # Add label with shape and ML class
                    shape = feature.get('shape', '')
                    ml_class = feature.get('ml_class', '')
                    label = f"{shape} ({ml_class}:{conf:.2f})"
                    cv2.putText(annotated_image, label, (x, y-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Display annotated image
            self.annotated_label.set_image(annotated_image)
            self.tabs.setCurrentIndex(2)  # Switch to annotated tab
            
            # Update results text
            result_text = "=== ML BENCHMARK ANALYSIS ===\n\n"
            result_text += f"Model: {Path(self.yolo_model_path).name}\n"
            result_text += f"Detections: {ml_result.num_detections}\n"
            result_text += f"Processing Time: {ml_result.processing_time:.2f}s\n\n"
            
            # Shape distribution
            from collections import Counter
            shapes = Counter([f['shape'] for f in ml_result.features])
            result_text += "Shape Distribution:\n"
            for shape, count in shapes.most_common():
                result_text += f"  {shape}: {count}\n"
            
            # Color distribution
            colors = Counter([f['color'] for f in ml_result.features])
            result_text += "\nColor Distribution:\n"
            for color, count in colors.most_common():
                result_text += f"  {color}: {count}\n"
            
            # Statistics summary
            if ml_result.features:
                circs = [f['circularity'] for f in ml_result.features]
                aspects = [f['aspect_ratio'] for f in ml_result.features]
                result_text += f"\nStatistics:\n"
                result_text += f"  Circularity: {np.mean(circs):.3f} ± {np.std(circs):.3f}\n"
                result_text += f"  Aspect Ratio: {np.mean(aspects):.2f} ± {np.std(aspects):.2f}\n"
            
            self.results_text.setText(result_text)
            
            # Update table
            self.update_results_table(ml_result)
            
            # Update statistics charts with ML results
            self.update_charts_full()
            
            self.progress_bar.setVisible(False)
            self.statusBar().showMessage(f'ML Benchmark complete: {ml_result.num_detections} particles detected')
            
        except Exception as e:
            self.progress_bar.setVisible(False)
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"ML Benchmark failed:\n{str(e)}")
    
    def _run_batch_ml_benchmark(self):
        """Run ML Benchmark on batch of images from folder"""
        # Ask for image folder or generate synthetic images
        from PyQt5.QtWidgets import QInputDialog
        
        items = ["Load Image Folder", "Generate Synthetic Images"]
        item, ok = QInputDialog.getItem(self, "Batch Source", 
                                       "Select image source:", items, 0, False)
        
        if not ok:
            return
        
        images_data = []
        
        if "Generate" in item:
            # Generate synthetic images
            num_images, ok = QInputDialog.getInt(self, "Number of Images",
                                                 "How many synthetic images?", 200, 1, 1000, 10)
            if not ok:
                return
            
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, num_images)
            self.statusBar().showMessage('Generating synthetic images...')
            
            from src.data_generation import SyntheticImageGenerator
            from config.settings import SyntheticImageParams
            generator = SyntheticImageGenerator()
            
            for i in range(num_images):
                self.progress_bar.setValue(i)
                QApplication.processEvents()
                
                # Generate with parameters from GUI interface
                import random
                params = SyntheticImageParams(
                    num_particles=self.synth_num_particles.value(),
                    shape_type=self.synth_shape_type.currentText(),
                    color_type=self.synth_color_type.currentText(),
                    enable_blur=self.synth_enable_blur.isChecked(),
                    blur_kernel=self.synth_blur_kernel.value(),
                    enable_glow=self.synth_enable_glow.isChecked(),
                    glow_intensity=self.synth_glow_intensity.value(),
                    image_width=self.synth_width.value(),
                    image_height=self.synth_height.value(),
                    background_brightness=self.synth_brightness.value(),
                    particle_brightness=self.synth_particle_brightness.value()
                )
                
                image, ground_truth = generator.generate(params)
                
                images_data.append({
                    'image': image,
                    'ground_truth': len(ground_truth),
                    'ground_truth_data': ground_truth,
                    'name': f'synthetic_{i+1:03d}'
                })
        else:
            # Load images from folder
            folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
            if not folder:
                return
            
            from pathlib import Path
            image_files = []
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.tiff', '*.bmp']:
                image_files.extend(Path(folder).glob(ext))
            
            if not image_files:
                QMessageBox.warning(self, "Warning", "No images found in folder")
                return
            
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, len(image_files))
            
            for i, img_path in enumerate(image_files):
                self.progress_bar.setValue(i)
                QApplication.processEvents()
                
                image = cv2.imread(str(img_path))
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Try to load ground truth with full particle details
                    ground_truth_count = 0
                    ground_truth_data = []
                    gt_path = self._find_ground_truth_file(img_path)
                    
                    if gt_path and gt_path.exists():
                        try:
                            with open(gt_path, 'r') as f:
                                content = f.read().strip()
                                
                                # Parse ground truth file
                                # Support two formats:
                                # Format 1 (simple): "Particles: N\nShapes:\n  1. Fragment\n  2. Fiber..."
                                # Format 2 (detailed): "Total Particles: N\n\nParticle 1:\n  Shape: ...\n  Color: ..."
                                lines = content.split('\n')
                                
                                # Get total count (try both formats)
                                if lines and 'Particles:' in lines[0]:
                                    ground_truth_count = int(lines[0].split(':')[1].strip())
                                
                                # Check format type
                                is_simple_format = any('Shapes:' in line for line in lines)
                                
                                if is_simple_format:
                                    # Parse simple format: just numbered shapes
                                    in_shapes_section = False
                                    for line in lines:
                                        line = line.strip()
                                        if line == 'Shapes:':
                                            in_shapes_section = True
                                            continue
                                        if in_shapes_section and line:
                                            # Line format: "  1. Fragment" or "1. Fragment"
                                            if '.' in line:
                                                shape = line.split('.', 1)[1].strip()
                                                ground_truth_data.append({
                                                    'shape': shape
                                                })
                                else:
                                    # Parse detailed format with full particle info
                                    current_particle = {}
                                    for line in lines[1:]:
                                        line = line.strip()
                                        if not line:
                                            if current_particle:
                                                ground_truth_data.append(current_particle)
                                                current_particle = {}
                                            continue
                                        
                                        if line.startswith('Particle '):
                                            if current_particle:
                                                ground_truth_data.append(current_particle)
                                            current_particle = {}
                                        elif ':' in line:
                                            key, value = line.split(':', 1)
                                            key = key.strip()
                                            value = value.strip()
                                            
                                            if key == 'Shape':
                                                current_particle['shape'] = value
                                            elif key == 'Color':
                                                current_particle['color_label'] = value
                                            elif key == 'Wavelength':
                                                current_particle['wavelength'] = int(value.replace('nm', ''))
                                            elif key == 'Position':
                                                # Parse (x, y) format
                                                pos_str = value.strip('()')
                                                x, y = pos_str.split(',')
                                                current_particle['position'] = (float(x.strip()), float(y.strip()))
                                            elif key == 'Area':
                                                current_particle['area'] = float(value.split()[0])
                                            elif key == 'Size':
                                                current_particle['size'] = float(value.split()[0])
                                    
                                    # Add last particle
                                    if current_particle:
                                        ground_truth_data.append(current_particle)
                                    
                        except Exception as e:
                            print(f"Warning: Could not parse ground truth from {gt_path}: {e}")
                            ground_truth_data = []
                    
                    images_data.append({
                        'image': image,
                        'ground_truth': ground_truth_count,
                        'ground_truth_data': ground_truth_data,
                        'name': img_path.stem
                    })
        
        if not images_data:
            QMessageBox.warning(self, "Warning", "No images to process")
            self.progress_bar.setVisible(False)
            return
        
        # Count how many images have ground truth
        images_with_gt = sum(1 for img in images_data if img['ground_truth'] > 0)
        
        # Run all three analyses on all images
        self.results_text.clear()
        self.results_text.append("=" * 60)
        self.results_text.append(f"ML BENCHMARK BATCH - {len(images_data)} IMAGES")
        self.results_text.append("=" * 60)
        if images_with_gt > 0:
            self.results_text.append(f"✓ Ground truth loaded for {images_with_gt}/{len(images_data)} images")
        else:
            self.results_text.append(f"⚠ No ground truth files found")
        
        quick_results_list = []
        deep_results_list = []
        ml_results_list = []
        
        self.progress_bar.setRange(0, len(images_data) * 3)  # Quick + Deep + ML
        
        import time
        from collections import Counter
        
        # Process all images with Quick Analysis
        self.results_text.append(f"\n[1/3] Running Quick Analysis on {len(images_data)} images...")
        QApplication.processEvents()
        
        quick_params = self.get_analysis_params('quick')
        quick_analyzer = QuickAnalyzer()
        
        for i, img_data in enumerate(images_data):
            self.progress_bar.setValue(i)
            self.statusBar().showMessage(f'Quick Analysis: {i+1}/{len(images_data)}')
            QApplication.processEvents()
            
            result = quick_analyzer.analyze(img_data['image'], quick_params)
            quick_results_list.append({
                'name': img_data['name'],
                'detected': result.num_detections,
                'ground_truth': img_data['ground_truth'],
                'time': result.processing_time,
                'result': result
            })
        
        # Process all images with Deep Analysis
        self.results_text.append(f"\n[2/3] Running Deep Analysis on {len(images_data)} images...")
        QApplication.processEvents()
        
        deep_params = self.get_analysis_params('deep')
        deep_analyzer = DeepAnalyzer()
        
        for i, img_data in enumerate(images_data):
            self.progress_bar.setValue(len(images_data) + i)
            self.statusBar().showMessage(f'Deep Analysis: {i+1}/{len(images_data)}')
            QApplication.processEvents()
            
            result = deep_analyzer.analyze(img_data['image'], deep_params)
            deep_results_list.append({
                'name': img_data['name'],
                'detected': result.num_detections,
                'ground_truth': img_data['ground_truth'],
                'time': result.processing_time,
                'result': result
            })
        
        # Process all images with ML Benchmark
        self.results_text.append(f"\n[3/3] Running ML Benchmark on {len(images_data)} images...")
        QApplication.processEvents()
        
        ml_analyzer = MLBenchmarkAnalyzer(self.yolo_model)
        
        for i, img_data in enumerate(images_data):
            self.progress_bar.setValue(len(images_data) * 2 + i)
            self.statusBar().showMessage(f'ML Benchmark: {i+1}/{len(images_data)}')
            QApplication.processEvents()
            
            result = ml_analyzer.analyze(img_data['image'], deep_params)
            ml_results_list.append({
                'name': img_data['name'],
                'detected': result.num_detections,
                'ground_truth': img_data['ground_truth'],
                'time': result.processing_time,
                'result': result
            })
        
        # Calculate aggregate statistics
        import numpy as np
        
        quick_detections = [r['detected'] for r in quick_results_list]
        deep_detections = [r['detected'] for r in deep_results_list]
        ml_detections = [r['detected'] for r in ml_results_list]
        quick_times = [r['time'] for r in quick_results_list]
        deep_times = [r['time'] for r in deep_results_list]
        ml_times = [r['time'] for r in ml_results_list]
        
        ground_truths = [r['ground_truth'] for r in quick_results_list if r['ground_truth'] > 0]
        
        # Display summary
        self.results_text.append("\n" + "=" * 60)
        self.results_text.append("ML BENCHMARK SUMMARY")
        self.results_text.append("=" * 60)
        
        self.results_text.append(f"\nProcessed: {len(images_data)} images")
        
        # Display results in comparison table format
        if ground_truths:
            self.results_text.append("\n" + "-" * 60)
            self.results_text.append(f"{'Method':<20} {'Avg Detections':<20} {'Avg Time (s)':<20}")
            self.results_text.append("-" * 60)
            self.results_text.append(f"{'Ground Truth':<20} {np.mean(ground_truths):<20.1f} {'N/A':<20}")
            self.results_text.append(f"{'Quick Analysis':<20} {np.mean(quick_detections):<20.1f} {np.mean(quick_times):<20.3f}")
            self.results_text.append(f"{'Deep Analysis':<20} {np.mean(deep_detections):<20.1f} {np.mean(deep_times):<20.3f}")
            self.results_text.append(f"{'ML Benchmark':<20} {np.mean(ml_detections):<20.1f} {np.mean(ml_times):<20.3f}")
            self.results_text.append("-" * 60)
        else:
            self.results_text.append("\n" + "-" * 60)
            self.results_text.append(f"{'Method':<20} {'Avg Detections':<20} {'Avg Time (s)':<20}")
            self.results_text.append("-" * 60)
            self.results_text.append(f"{'Quick Analysis':<20} {np.mean(quick_detections):<20.1f} {np.mean(quick_times):<20.3f}")
            self.results_text.append(f"{'Deep Analysis':<20} {np.mean(deep_detections):<20.1f} {np.mean(deep_times):<20.3f}")
            self.results_text.append(f"{'ML Benchmark':<20} {np.mean(ml_detections):<20.1f} {np.mean(ml_times):<20.3f}")
            self.results_text.append("-" * 60)
        
        # Generate HTML report with all three methods
        self.results_text.append("\n" + "=" * 60)
        self.results_text.append("GENERATING HTML REPORT...")
        QApplication.processEvents()
        
        try:
            from datetime import datetime
            from pathlib import Path
            from src.analysis.report_generator import ReportGenerator
            
            # Calculate metrics for all three methods
            def calculate_batch_metrics(detections_list, ground_truths_list):
                if not ground_truths_list or len(ground_truths_list) == 0:
                    return 0.0, 0.0, 0.0
                
                total_tp = total_fp = total_fn = 0
                for detected, gt in zip(detections_list, ground_truths_list):
                    if gt > 0:
                        tp = min(detected, gt)
                        fp = max(0, detected - gt)
                        fn = max(0, gt - detected)
                        total_tp += tp
                        total_fp += fp
                        total_fn += fn
                
                precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
                recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                
                return precision, recall, f1
            
            gt_values = [r['ground_truth'] for r in quick_results_list if r['ground_truth'] > 0]
            quick_det_with_gt = [r['detected'] for r in quick_results_list if r['ground_truth'] > 0]
            deep_det_with_gt = [r['detected'] for r in deep_results_list if r['ground_truth'] > 0]
            ml_det_with_gt = [r['detected'] for r in ml_results_list if r['ground_truth'] > 0]
            
            quick_precision, quick_recall, quick_f1 = calculate_batch_metrics(quick_det_with_gt, gt_values)
            deep_precision, deep_recall, deep_f1 = calculate_batch_metrics(deep_det_with_gt, gt_values)
            ml_precision, ml_recall, ml_f1 = calculate_batch_metrics(ml_det_with_gt, gt_values)
            
            # Aggregate distributions for all three methods
            # Use consistent grouping and color filtering for all methods
            fluorescent_colors = {'Red', 'Green', 'Blue', 'Yellow'}
            
            quick_shapes = Counter()
            quick_colors = Counter()
            quick_areas = []
            for r in quick_results_list:
                if r['result'].features:
                    for f in r['result'].features:
                        shape = map_to_grouped_shape(f.get('shape', 'Unknown'))
                        quick_shapes[shape] += 1
                        color = f.get('color', 'Unknown')
                        if color in fluorescent_colors:
                            quick_colors[color] += 1
                        quick_areas.append(f.get('area', 0))
            
            deep_shapes = Counter()
            deep_colors = Counter()
            deep_areas = []
            for r in deep_results_list:
                if r['result'].features:
                    for f in r['result'].features:
                        shape = map_to_grouped_shape(f.get('shape', 'Unknown'))
                        deep_shapes[shape] += 1
                        color = f.get('color', 'Unknown')
                        if color in fluorescent_colors:
                            deep_colors[color] += 1
                        deep_areas.append(f.get('area', 0))
            
            ml_shapes = Counter()
            ml_colors = Counter()
            ml_areas = []
            for r in ml_results_list:
                if r['result'].features:
                    for f in r['result'].features:
                        shape = map_to_grouped_shape(f.get('shape', 'Unknown'))
                        ml_shapes[shape] += 1
                        color = f.get('color', 'Unknown')
                        if color in fluorescent_colors:
                            ml_colors[color] += 1
                        ml_areas.append(f.get('area', 0))
            
            # Collect ground truth distributions if available
            # Map to grouped shapes and filter to fluorescent colors
            gt_shapes = Counter()
            gt_colors = Counter()
            gt_areas = []
            for img_data in images_data:
                if img_data.get('ground_truth_data'):
                    for particle in img_data['ground_truth_data']:
                        if 'shape' in particle:
                            shape = map_to_grouped_shape(particle['shape'])
                            gt_shapes[shape] += 1
                        if 'color_label' in particle:
                            color = particle['color_label']
                            if color in fluorescent_colors:
                                gt_colors[color] += 1
                        if 'area' in particle:
                            gt_areas.append(particle['area'])
            
            # Create batch data with ML Benchmark included
            batch_data = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'num_images': len(images_data),
                'quick_analysis': {
                    'detected': int(np.mean(quick_detections)),
                    'ground_truth': int(np.mean(gt_values)) if gt_values else 0,
                    'precision': quick_precision,
                    'recall': quick_recall,
                    'f1_score': quick_f1,
                    'processing_time': np.mean(quick_times),
                    'shape_distribution': dict(quick_shapes),
                    'color_distribution': dict(quick_colors),
                    'area_distribution': quick_areas
                },
                'deep_analysis': {
                    'detected': int(np.mean(deep_detections)),
                    'ground_truth': int(np.mean(gt_values)) if gt_values else 0,
                    'precision': deep_precision,
                    'recall': deep_recall,
                    'f1_score': deep_f1,
                    'processing_time': np.mean(deep_times),
                    'shape_distribution': dict(deep_shapes),
                    'color_distribution': dict(deep_colors),
                    'area_distribution': deep_areas
                },
                'ml_benchmark': {
                    'detected': int(np.mean(ml_detections)),
                    'ground_truth': int(np.mean(gt_values)) if gt_values else 0,
                    'precision': ml_precision,
                    'recall': ml_recall,
                    'f1_score': ml_f1,
                    'processing_time': np.mean(ml_times),
                    'shape_distribution': dict(ml_shapes),
                    'color_distribution': dict(ml_colors),
                    'area_distribution': ml_areas,
                    'model_name': Path(self.yolo_model_path).name
                },
                'ground_truth': int(np.mean(gt_values)) if gt_values else 0,
                'ground_truth_distributions': {
                    'shape_distribution': dict(gt_shapes),
                    'color_distribution': dict(gt_colors),
                    'area_distribution': gt_areas
                } if gt_shapes or gt_colors or gt_areas else None
            }
            
            report_gen = ReportGenerator()
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = Path('benchmark_results')
            output_dir.mkdir(exist_ok=True)
            
            html_path = output_dir / f'ml_benchmark_{len(images_data)}images_{timestamp}.html'
            report_gen.generate_benchmark_report(batch_data, str(html_path))
            
            self.results_text.append(f"\n✓ HTML Report saved: {html_path}")
            
            # Open in browser
            import webbrowser
            webbrowser.open(f'file:///{html_path.absolute()}')
            
        except Exception as e:
            self.results_text.append(f"\n✗ Failed to generate report: {e}")
            import traceback
            traceback.print_exc()
        
        self.progress_bar.setVisible(False)
        self.statusBar().showMessage(f'ML Benchmark complete - {len(images_data)} images processed')
        self.results_text.append("\n" + "=" * 60)
        self.results_text.append("ML BENCHMARK COMPLETE")
        self.results_text.append("=" * 60)
            
    def update_ml_results_table(self, detections):
        """Update table with ML detection results"""
        columns = ['ID', 'Class', 'Confidence', 'X', 'Y', 'Width', 'Height']
        
        self.results_table.setColumnCount(len(columns))
        self.results_table.setHorizontalHeaderLabels(columns)
        self.results_table.setRowCount(len(detections))
        
        for i, det in enumerate(detections):
            self.results_table.setItem(i, 0, QTableWidgetItem(str(det['id'])))
            self.results_table.setItem(i, 1, QTableWidgetItem(det['class']))
            self.results_table.setItem(i, 2, QTableWidgetItem(f"{det['confidence']:.3f}"))
            self.results_table.setItem(i, 3, QTableWidgetItem(str(det['bbox'][0])))
            self.results_table.setItem(i, 4, QTableWidgetItem(str(det['bbox'][1])))
            self.results_table.setItem(i, 5, QTableWidgetItem(str(det['bbox'][2])))
            self.results_table.setItem(i, 6, QTableWidgetItem(str(det['bbox'][3])))
        
        self.results_table.resizeColumnsToContents()
        
    def set_fluorescent_params(self):
        """Set optimal parameters for fluorescent microscopy (black background)"""
        # Quick Analysis settings
        self.quick_method.setCurrentText('adaptive')
        self.quick_min_area.setValue(10)
        self.quick_blur.setValue(3)
        self.quick_distance_thresh.setValue(0.3)
        
        # Deep Analysis settings
        self.deep_method.setCurrentText('advanced')
        self.deep_min_area.setValue(5)
        self.deep_blur.setValue(3)
        self.deep_distance_thresh.setValue(0.3)
        self.deep_marker_size.setValue(3)
        self.adaptive_c_value.setValue(1)
        
        self.statusBar().showMessage('Parameters optimized for fluorescent microscopy')
        QMessageBox.information(
            self,
            "Parameters Updated",
            "Settings optimized for fluorescent microscopy:\n\n"
            "• Quick: adaptive (faster)\n"
            "• Deep: advanced (more accurate)\n"
            "• Min Area: reduced to 5-10\n"
            "• Blur: reduced to 3\n"
            "• Distance threshold: 0.3\n\n"
            "These settings work best with dark backgrounds and bright particles."
        )
    
    def set_brightfield_params(self):
        """Set optimal parameters for brightfield microscopy (white background)"""
        # Quick Analysis settings
        self.quick_method.setCurrentText('basic')
        self.quick_min_area.setValue(50)
        self.quick_blur.setValue(5)
        self.quick_distance_thresh.setValue(0.5)
        
        # Deep Analysis settings
        self.deep_method.setCurrentText('basic')
        self.deep_min_area.setValue(10)
        self.deep_blur.setValue(5)
        self.deep_distance_thresh.setValue(0.5)
        self.deep_marker_size.setValue(5)
        self.adaptive_c_value.setValue(2)
        
        self.statusBar().showMessage('Parameters optimized for brightfield microscopy')
        QMessageBox.information(
            self,
            "Parameters Updated",
            "Settings optimized for brightfield microscopy:\n\n"
            "• Preprocessing: basic/advanced\n"
            "• Min Area: 10-50\n"
            "• Blur: 5\n"
            "• Distance threshold: 0.5\n\n"
            "These settings work best with bright backgrounds."
        )
        
    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(
            self, 
            "About Microplastic Analyzer",
            "Microplastic Analyzer Pro\n\n"
            "Version 2.0 (Refactored)\n\n"
            "A comprehensive tool for detecting and analyzing\n"
            "microplastic particles in microscopy images.\n\n"
            "Uses advanced computer vision and machine learning.\n\n"
            "Features:\n"
            "• Quick & Deep Analysis modes\n"
            "• Synthetic data generation\n"
            "• YOLO ML model support\n"
            "• Comprehensive parameter controls"
        )


class SyntheticDataDialog(QtWidgets.QDialog):
    """Dialog for generating synthetic data (batch generation)"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Batch Generate Synthetic Data")
        self.setModal(True)
        self.parent_gui = parent
        self.initUI()
        
    def initUI(self):
        layout = QVBoxLayout(self)
        
        # Info label
        info_label = QLabel("Generate multiple synthetic images using current parameters")
        info_label.setStyleSheet("font-weight: bold; padding: 5px;")
        layout.addWidget(info_label)
        
        # Number of images
        num_layout = QHBoxLayout()
        num_layout.addWidget(QLabel("Number of images:"))
        self.num_spin = QSpinBox()
        self.num_spin.setRange(1, 1000)
        self.num_spin.setValue(10)
        num_layout.addWidget(self.num_spin)
        layout.addLayout(num_layout)
        
        # Output directory
        dir_layout = QHBoxLayout()
        dir_layout.addWidget(QLabel("Output directory:"))
        self.dir_edit = QtWidgets.QLineEdit("data/synthetic")
        dir_layout.addWidget(self.dir_edit)
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_directory)
        dir_layout.addWidget(browse_btn)
        layout.addLayout(dir_layout)
        
        # Save ground truth checkbox
        self.save_gt_checkbox = QCheckBox("Save ground truth JSON files")
        self.save_gt_checkbox.setChecked(True)
        layout.addWidget(self.save_gt_checkbox)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Buttons
        button_layout = QHBoxLayout()
        generate_btn = QPushButton("Generate")
        generate_btn.setStyleSheet("padding: 8px; font-weight: bold; background-color: #4CAF50;")
        generate_btn.clicked.connect(self.generate)
        button_layout.addWidget(generate_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
        
    def browse_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if directory:
            self.dir_edit.setText(directory)
            
    def generate(self):
        num_images = self.num_spin.value()
        output_dir = Path(self.dir_edit.text())
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Get parameters from parent GUI
            params = SyntheticImageParams(
                num_particles=self.parent_gui.synth_num_particles.value(),
                shape_type=self.parent_gui.synth_shape_type.currentText(),
                color_type=self.parent_gui.synth_color_type.currentText(),
                enable_blur=self.parent_gui.synth_enable_blur.isChecked(),
                blur_kernel=self.parent_gui.synth_blur_kernel.value(),
                enable_glow=self.parent_gui.synth_enable_glow.isChecked(),
                glow_intensity=self.parent_gui.synth_glow_intensity.value(),
                glow_sigma_min=self.parent_gui.synth_glow_sigma_min.value(),
                glow_sigma_max=self.parent_gui.synth_glow_sigma_max.value(),
                background_noise_min=self.parent_gui.synth_bg_noise_min.value(),
                background_noise_max=self.parent_gui.synth_bg_noise_max.value(),
                image_width=self.parent_gui.synth_width.value(),
                image_height=self.parent_gui.synth_height.value(),
                background_brightness=self.parent_gui.synth_brightness.value(),
                particle_brightness=self.parent_gui.synth_particle_brightness.value()
            )
            
            generator = SyntheticImageGenerator()
            
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, num_images)
            
            for i in range(num_images):
                image, ground_truth = generator.generate(params)
                
                # Save image
                image_file = output_dir / f"synthetic_{i+1:04d}.png"
                cv2.imwrite(str(image_file), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                
                # Save ground truth if requested
                if self.save_gt_checkbox.isChecked():
                    gt_file = output_dir / f"synthetic_{i+1:04d}_gt.json"
                    with open(gt_file, 'w') as f:
                        json.dump(ground_truth, f, indent=2)
                
                self.progress_bar.setValue(i + 1)
            
            self.progress_bar.setVisible(False)
            
            QMessageBox.information(
                self, 
                "Success", 
                f"Generated {num_images} images in:\n{output_dir}"
            )
            self.accept()
            
        except Exception as e:
            self.progress_bar.setVisible(False)
            QMessageBox.critical(self, "Error", f"Generation failed:\n{str(e)}")


def main():
    """Main entry point for GUI"""
    app = QtWidgets.QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show main window
    window = MicroplasticAnalyzerGUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
