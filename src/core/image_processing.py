"""
Image Processing Module
Contains core image processing algorithms for microplastic detection
"""

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage import morphology, segmentation, measure, feature
from skimage.segmentation import clear_border
from typing import Tuple

from config.constants import (
    BG_WHITE_THRESHOLD,
    BG_BLACK_THRESHOLD,
    BG_STD_THRESHOLD
)


class ImageProcessor:
    """
    Core Image Processing Engine for Microplastic Detection.
    
    Provides multiple preprocessing methods optimized for different microscopy types:
    - Brightfield microscopy (white background, dark particles)
    - Fluorescent microscopy (dark background, bright particles)
    
    Methods are categorized as:
    - QUICK: Fast processing for real-time analysis (basic, edge)
    - DEEP: Advanced processing for detailed analysis (advanced, watershed, adaptive)
    """
    
    @staticmethod
    def detect_background_color(img):
        """
        Automatically detect microscopy background type.
        
        Analyzes image corners and histogram to determine if the background
        is white (brightfield) or black (fluorescent).
        
        Args:
            img: RGB image as numpy array
            
        Returns:
            str: 'white' for brightfield, 'black' for fluorescent, 'unknown' if unclear
        """
        if img is None or img.size == 0:
            return 'unknown'
        
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        
        # Sample corners (10x10 pixels each corner)
        corners = [
            gray[:10, :10].flatten(),
            gray[:10, w-10:].flatten(),
            gray[h-10:, :10].flatten(),
            gray[h-10:, w-10:].flatten()
        ]
        corner_values = np.concatenate(corners)
         
        # Analyze corner statistics
        mean_val = np.mean(corner_values)
        std_val = np.std(corner_values)
        
        # Robust detection with multiple criteria
        if mean_val > BG_WHITE_THRESHOLD and std_val < BG_STD_THRESHOLD:
            return 'white'
        elif mean_val < BG_BLACK_THRESHOLD and std_val < 40:
            return 'black'
        else:
            # Check overall image statistics
            image_mean = np.mean(gray)
            image_std = np.std(gray)
            
            # Check histogram peaks
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = hist.flatten()
            
            # Find peaks in histogram
            peaks = []
            for i in range(1, 255):
                if hist[i] > hist[i-1] and hist[i] > hist[i+1]:
                    peaks.append((i, hist[i]))
            
            peaks.sort(key=lambda x: x[1], reverse=True)
            
            if len(peaks) >= 2:
                main_peak1 = peaks[0][0]
                main_peak2 = peaks[1][0] if len(peaks) > 1 else main_peak1
                
                if (main_peak1 > 220 and main_peak2 < 100) or (main_peak2 > 220 and main_peak1 < 100):
                    return 'white'
                elif (main_peak1 < 50 and main_peak2 > 150) or (main_peak2 < 50 and main_peak1 > 150):
                    return 'black'
            
            # Fallback to overall statistics
            if image_mean > 160 and image_std < 100:
                return 'white'
            elif image_mean < 100 and image_std < 80:
                return 'black'
            else:
                return 'unknown'
    
    @staticmethod
    def _remove_small_components(mask_bool, min_size):
        """
        Remove small noise components from binary mask.
        
        Args:
            mask_bool: Boolean mask array
            min_size: Minimum object area in pixels to retain
            
        Returns:
            Cleaned boolean mask with small objects removed
        """
        if min_size is None:
            return mask_bool
        threshold = int(max(0, min_size))
        if threshold <= 0:
            return mask_bool
        try:
            cleaned = morphology.remove_small_objects(mask_bool, max_size=max(0, threshold - 1))
        except TypeError:
            cleaned = morphology.remove_small_objects(mask_bool, min_size=threshold)
        return cleaned
    
    @staticmethod
    def _auto_canny(gray, sigma=0.33):
        """
        Apply Canny edge detection with automatic threshold calculation.
        
        Thresholds are computed from image median for robust edge detection
        across varying contrast levels.
        
        Args:
            gray: Grayscale image
            sigma: Threshold sensitivity factor (default: 0.33)
            
        Returns:
            Binary edge map
        """
        v = np.median(gray)
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        if lower == upper:
            upper = min(255, lower + 1)
        return cv2.Canny(gray, lower, upper)
    
    @staticmethod
    def _quick_edge_mask(img_rgb, min_size=10):
        """
        Generate particle mask using fast edge-based detection.
        
        Combines Canny and Scharr edge detectors with Otsu thresholding
        for rapid particle segmentation.
        
        Args:
            img_rgb: RGB input image
            min_size: Minimum particle area in pixels
            
        Returns:
            Binary mask (uint8)
        """
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        
        # Detect background and adapt
        bg_color = ImageProcessor.detect_background_color(img_rgb)
        if bg_color == 'white':
            gray = cv2.bitwise_not(gray)
        elif bg_color == 'unknown':
            _, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            _, thresh2 = cv2.threshold(255-gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            if cv2.countNonZero(thresh1) > cv2.countNonZero(thresh2):
                gray = gray
            else:
                gray = 255 - gray
        
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        edge_canny = ImageProcessor._auto_canny(blurred, 0.33)
        scharr_x = cv2.Scharr(blurred, cv2.CV_16S, 1, 0)
        scharr_y = cv2.Scharr(blurred, cv2.CV_16S, 0, 1)
        scharr_mag = cv2.magnitude(scharr_x.astype(np.float32), scharr_y.astype(np.float32))
        scharr_mag = cv2.convertScaleAbs(scharr_mag)
        _, edge_scharr = cv2.threshold(scharr_mag, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        edges = cv2.bitwise_or(edge_canny, edge_scharr)
        
        combined = cv2.bitwise_or(thresh, edges)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        filled = ndi.binary_fill_holes(closed > 0).astype(np.uint8) * 255
        
        mask_bool = filled > 0
        cleaned = ImageProcessor._remove_small_components(mask_bool, min_size)
        
        return (cleaned * 255).astype(np.uint8)
    
    @staticmethod
    def preprocess_segment(img, min_size=20, method='advanced', **kwargs):
        """
        Main preprocessing dispatcher for particle segmentation.
        
        Selects and applies the appropriate preprocessing method based on
        analysis requirements and image characteristics.
        
        Available Methods:
            - 'basic': Fast Otsu thresholding (QUICK)
            - 'advanced': Multi-channel RGB processing with CLAHE (DEEP)
            - 'watershed': Marker-based watershed for touching particles (DEEP)
            - 'edge': Edge detection with contour filling (DEEP only)
            - 'adaptive': Background subtraction with adaptive thresholding (DEEP)
            - 'brightfield': Specialized for brightfield microscopy (DEEP)
        
        Args:
            img: RGB input image as numpy array
            min_size: Minimum particle area in pixels (default: 20)
            method: Preprocessing algorithm to use (default: 'advanced')
            **kwargs: Method-specific parameters
            
        Returns:
            tuple: (binary_mask, threshold_value)
        """
        if img is None or img.size == 0:
            return np.zeros((1, 1), dtype=np.uint8), 0
        
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        method = (method or 'advanced').lower()
        
        # Auto-detect brightfield and use specialized method if background is white
        bg_color = ImageProcessor.detect_background_color(img)
        
        try:
            if method == 'advanced':
                # If brightfield detected, use brightfield-optimized processing
                if bg_color == 'white':
                    return ImageProcessor._preprocess_brightfield(img_bgr, min_size, **kwargs)
                return ImageProcessor._preprocess_advanced(img_bgr, min_size, **kwargs)
            if method == 'basic':
                if bg_color == 'white':
                    return ImageProcessor._preprocess_brightfield_basic(img_bgr, min_size, **kwargs)
                return ImageProcessor._preprocess_basic(img_bgr, min_size, **kwargs)
            if method == 'watershed':
                if bg_color == 'white':
                    return ImageProcessor._preprocess_brightfield_watershed(img_bgr, min_size, **kwargs)
                return ImageProcessor._preprocess_watershed(img_bgr, min_size, **kwargs)
            if method == 'edge':
                return ImageProcessor._preprocess_edge(img_bgr, min_size, **kwargs)
            if method == 'adaptive':
                if bg_color == 'white':
                    return ImageProcessor._preprocess_brightfield(img_bgr, min_size, **kwargs)
                return ImageProcessor._preprocess_adaptive(img_bgr, min_size, **kwargs)
            if method == 'brightfield':
                return ImageProcessor._preprocess_brightfield(img_bgr, min_size, **kwargs)
        except Exception as e:
            print(f"ERROR in preprocessing ({method}): {str(e)}")
            return ImageProcessor._preprocess_basic(img_bgr, min_size, **kwargs)
        
        return ImageProcessor._preprocess_advanced(img_bgr, min_size, **kwargs)
    
    @staticmethod
    def _preprocess_basic(img, min_size=20, **kwargs):
        """
        [QUICK] Fast Otsu-based particle detection.
        
        Optimized for speed using classic Otsu thresholding with automatic
        background detection. Ideal for real-time processing and preliminary
        screening of samples.
        
        Processing Pipeline:
            1. Background detection (white/black/unknown)
            2. Adaptive thresholding based on background type
            3. Morphological cleanup (open/close)
            4. Hole filling and noise removal
        
        Supports:
            - Brightfield microscopy (white background)
            - Fluorescent microscopy (dark background)
        
        Args:
            img: BGR input image
            min_size: Minimum particle area in pixels
            **kwargs: Additional parameters (unused)
            
        Returns:
            tuple: (binary_mask, threshold_value)
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        bg_color = ImageProcessor.detect_background_color(img_rgb)
        
        # === FLUORESCENT MICROSCOPY: Dark background, bright particles ===
        if bg_color == 'black':
            # For fluorescent: particles are bright on dark background
            # Use direct thresholding without inversion
            
            # Calculate threshold based on image statistics
            gray_mean = np.mean(gray)
            gray_std = np.std(gray)
            thresh_val = max(gray_mean + gray_std * 1.5, 30)
            
            # Threshold to get bright particles
            _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
            
            # Quick cleanup
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            
        # === BRIGHTFIELD: White background, dark particles ===
        elif bg_color == 'white':
            # Invert so particles are bright
            gray = 255 - gray
            thresh_val, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
        # === UNKNOWN: Decide based on mean ===
        else:
            if np.mean(gray) > 127:
                gray = 255 - gray
            
            thresh_val, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # === Hole filling ===
        mask = ndi.binary_fill_holes(mask > 0).astype(np.uint8) * 255
        
        # === Remove small noise ===
        mask_bool = mask > 0
        if np.sum(mask_bool) > 0:
            cleaned = ImageProcessor._remove_small_components(mask_bool, min_size)
            mask = (cleaned * 255).astype(np.uint8)
        
        return mask, thresh_val if bg_color == 'black' else thresh_val if 'thresh_val' in locals() else 127
    
    @staticmethod
    def _preprocess_advanced(img, min_size=20, **kwargs):
        """
        [DEEP] Advanced multi-channel RGB processing.
        
        Processes each color channel independently with CLAHE enhancement,
        then intelligently combines results for optimal particle detection.
        Best for samples with color variation or mixed particle types.
        
        Processing Pipeline:
            1. Split RGB channels
            2. CLAHE enhancement per channel
            3. Channel-specific thresholding (fluorescent vs brightfield)
            4. Intelligent channel fusion (OR combination)
            5. Morphological refinement
            6. Boundary smoothing
        
        Supports:
            - Brightfield microscopy (white background)
            - Fluorescent microscopy (dark background)
            - Multi-color fluorescent samples
        
        Args:
            img: BGR input image
            min_size: Minimum particle area in pixels
            **kwargs: Additional parameters (unused)
            
        Returns:
            tuple: (binary_mask, threshold_value)
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bg_color = ImageProcessor.detect_background_color(img_rgb)
        
        # === STEP 1: Process each RGB channel independently ===
        # Extract color channels
        r_channel, g_channel, b_channel = cv2.split(img_rgb)
        
        channel_masks = []
        channel_weights = []  # Track channel importance
        
        print(f"DEBUG Advanced: Processing RGB channels independently...")
        
        for idx, (channel, name) in enumerate([(r_channel, 'Red'), (g_channel, 'Green'), (b_channel, 'Blue')]):
            # Check if channel has useful information
            channel_range = np.max(channel) - np.min(channel)
            channel_std = np.std(channel)
            
            if channel_range < 20 or channel_std < 5:
                print(f"DEBUG Advanced: {name} channel has low variation, skipping")
                continue
            
            # CLAHE enhancement per channel for better contrast
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            enhanced = clahe.apply(channel)
            
            # === Different processing for fluorescent vs brightfield ===
            if bg_color == 'black':
                # FLUORESCENT: Detect BRIGHT particles
                mean_val = np.mean(enhanced)
                std_val = np.std(enhanced)
                thresh_val = max(mean_val + std_val * 2.5, 60)
                
                # _ is value of thesh_val
                _, ch_mask = cv2.threshold(enhanced, thresh_val, 255, cv2.THRESH_BINARY)
                
            else:
                # BRIGHTFIELD: Detect DARK particles
                # Invert channel to make dark particles bright
                enhanced_inv = 255 - enhanced
                
                # Use Otsu on each inverted channel
                _, ch_mask = cv2.threshold(enhanced_inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Clean individual channel mask
            kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            ch_mask = cv2.morphologyEx(ch_mask, cv2.MORPH_OPEN, kernel_small)
            
            # Calculate channel weight based on detected content
            detected_pixels = np.sum(ch_mask > 0)
            if detected_pixels > min_size * 2:
                channel_masks.append(ch_mask)
                channel_weights.append(detected_pixels)
                print(f"DEBUG Advanced: {name} detected {detected_pixels} pixels")
        
        # === STEP 2: Intelligent channel combination ===
        if len(channel_masks) == 0:
            print("DEBUG Advanced: No valid channels detected")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            color_mask = np.zeros_like(gray)
        elif len(channel_masks) == 1:
            # Only one channel valid
            color_mask = channel_masks[0]
            print("DEBUG Advanced: Using single channel")
        else:
            # Multiple channels - use weighted combination
            # Strategy: OR operation to capture all particles from any channel
            color_mask = channel_masks[0]
            for mask in channel_masks[1:]:
                color_mask = cv2.bitwise_or(color_mask, mask)
            print(f"DEBUG Advanced: Combined {len(channel_masks)} channels")
        
        # === STEP 3: Morphological refinement to remove noise ===
        # Use morphological operations to clean up the mask
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # Remove small noise
        refined = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel_open, iterations=2)
        # Fill small gaps
        refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel_close)
        
        # === STEP 4: Hole filling and component cleanup ===
        filled = ndi.binary_fill_holes(refined > 0).astype(np.uint8) * 255
        
        # Remove small components with strict threshold
        mask_bool = filled > 0
        if np.sum(mask_bool) > 0:
            cleaned = ImageProcessor._remove_small_components(mask_bool, min_size)
            # NOTE: Don't use clear_border here - let analyzer handle boundary filtering
            # This preserves all detected objects for accurate counting
            mask = (cleaned * 255).astype(np.uint8)
        else:
            mask = filled
        
        # === STEP 5: Final boundary smoothing for accurate measurements ===
        # Slight dilation-erosion for smooth, accurate boundaries
        kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        mask = cv2.dilate(mask, kernel_smooth, iterations=1)
        mask = cv2.erode(mask, kernel_smooth, iterations=1)
        
        print(f"DEBUG Advanced: Final mask has {np.sum(mask > 0)} pixels")
        return mask, 127
    
    @staticmethod
    def _create_gabor_kernel(ksize, sigma, theta, lambd, gamma, psi):
        """
        Generate Gabor filter kernel for texture analysis.
        
        Args:
            ksize: Kernel size (must be odd)
            sigma: Standard deviation of Gaussian envelope
            theta: Orientation angle in radians
            lambd: Wavelength of sinusoidal factor
            gamma: Spatial aspect ratio
            psi: Phase offset
            
        Returns:
            Gabor kernel as float32 array
        """
        return cv2.getGaborKernel(
            (ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F
        )
    
    @staticmethod
    def _preprocess_watershed(img, min_size=20, **kwargs):
        """
        [DEEP] Watershed segmentation for touching particles.
        
        Advanced watershed algorithm with background subtraction and
        h-maxima marker detection. Optimized for separating overlapping
        or touching particles in dense samples.
        
        Processing Pipeline:
            1. Auto-select best color channel
            2. CLAHE enhancement and denoising
            3. Background subtraction (tophat transform)
            4. Distance transform computation
            5. H-maxima marker detection
            6. Watershed segmentation
            7. Region property filtering
        
        Supports:
            - Dense particle samples
            - Touching/overlapping particles
            - Both microscopy types
        
        Args:
            img: BGR input image
            min_size: Minimum particle area in pixels
            **kwargs:
                - distance_thresh: Distance transform threshold (default: 0.5)
                - marker_size: Marker dilation size (default: 3)
                - expected_max_diameter: Expected particle diameter (default: 50)
                - h_max_factor: H-maxima suppression factor (default: 0.35)
            
        Returns:
            tuple: (binary_mask, threshold_value)
        """
        # Get parameters
        distance_thresh = kwargs.get('distance_thresh', 0.5)
        marker_size = kwargs.get('marker_size', 3)
        expected_diameter = kwargs.get('expected_max_diameter', 200)  # Increased for fiber support
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bg_color = ImageProcessor.detect_background_color(img_rgb)
        
        print(f"DEBUG Watershed: Background={bg_color}, expected_diameter={expected_diameter}")
        
        # === STEP 1: Channel selection (auto-select best channel) ===
        r_channel, g_channel, b_channel = cv2.split(img_rgb)
        # Heuristic: choose channel with highest contrast
        if np.mean(r_channel) > np.mean(g_channel) and np.mean(r_channel) > np.mean(b_channel):
            working_channel = r_channel
            print("DEBUG Watershed: Using RED channel")
        elif np.std(g_channel) > np.std(r_channel):
            working_channel = g_channel
            print("DEBUG Watershed: Using GREEN channel")
        else:
            working_channel = gray
            print("DEBUG Watershed: Using GRAY channel")
        
        # === STEP 2: CLAHE enhancement ===
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(working_channel)
        
        # === STEP 3: Denoise with median filter ===
        denoised = cv2.medianBlur(enhanced, 3)
        
        # === STEP 4: Background subtraction (TOPHAT transform) ===
        # This is KEY improvement - removes uneven illumination
        bg_kernel_size = max(3, int(2.5 * expected_diameter))
        if bg_kernel_size % 2 == 0:
            bg_kernel_size += 1
        bg_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bg_kernel_size, bg_kernel_size))
        background = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, bg_kernel)
        foreground = cv2.subtract(denoised, background)
        
        print(f"DEBUG Watershed: Background subtraction with kernel size {bg_kernel_size}")
        
        # === STEP 5: Adaptive thresholding ===
        # Calculate block size from expected diameter
        block_size = max(15, int(1.5 * expected_diameter))
        if block_size % 2 == 0:
            block_size += 1
        
        if bg_color == 'black':
            # For fluorescent: use global threshold on foreground
            mean_val = np.mean(foreground)
            std_val = np.std(foreground)
            thresh_val = max(mean_val + std_val * 1.5, 20)
            _, basic_mask = cv2.threshold(foreground, thresh_val, 255, cv2.THRESH_BINARY)
        else:
            # For brightfield: use adaptive threshold
            basic_mask = cv2.adaptiveThreshold(
                foreground, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                block_size, -2
            )
        
        # === STEP 6: Clean mask ===
        basic_mask = ndi.binary_fill_holes(basic_mask > 0)
        # Use max_size parameter (new API) with value = min_size - 1 to match old behavior
        try:
            basic_mask = morphology.remove_small_objects(basic_mask, max_size=max(0, min_size - 1))
        except TypeError:
            # Fallback for older skimage versions
            basic_mask = morphology.remove_small_objects(basic_mask, min_size=min_size)
        basic_mask = (basic_mask * 255).astype(np.uint8)
        
        if np.sum(basic_mask > 0) == 0:
            print("DEBUG Watershed: No objects detected after thresholding")
            return basic_mask, 127
        
        # === STEP 7: Distance transform ===
        dist_transform = ndi.distance_transform_edt(basic_mask > 0)
        max_dist = np.max(dist_transform)
        
        if max_dist == 0:
            print("DEBUG Watershed: Distance transform is zero")
            return basic_mask, 127
        
        # === STEP 8: H-MAXIMA for marker detection (KEY IMPROVEMENT) ===
        # Suppress small peaks to reduce false markers
        h_factor = kwargs.get('h_max_factor', 0.35)
        h_value = max(1, int(h_factor * max_dist))
        
        try:
            from skimage import morphology as sk_morph
            # H-maxima suppresses peaks smaller than h_value
            h_maxima_result = sk_morph.h_maxima(dist_transform, h=h_value)
            markers, num_markers = ndi.label(h_maxima_result)
            print(f"DEBUG Watershed: H-maxima (h={h_value}) found {num_markers} markers")
        except Exception as e:
            print(f"DEBUG Watershed: H-maxima failed ({e}), using threshold fallback")
            # Fallback to threshold-based markers
            thresh_value = int(max_dist * distance_thresh)
            markers = ndi.label(dist_transform > thresh_value)[0]
            num_markers = markers.max()
        
        # === STEP 9: Fallback if too few markers ===
        if num_markers < 2:
            print(f"DEBUG Watershed: Too few markers ({num_markers}), using peak_local_max")
            try:
                from skimage import feature
                min_distance = max(1, int(0.5 * expected_diameter))
                coords = feature.peak_local_max(dist_transform, min_distance=min_distance, labels=basic_mask>0)
                markers = np.zeros_like(dist_transform, dtype=int)
                for i, coord in enumerate(coords, start=1):
                    markers[tuple(coord)] = i
                markers = ndi.label(markers > 0)[0]
                num_markers = markers.max()
                print(f"DEBUG Watershed: peak_local_max found {num_markers} markers")
            except Exception as e:
                print(f"DEBUG Watershed: peak_local_max failed ({e})")
                return basic_mask, 127
        
        # === STEP 10: Apply watershed on negative distance ===
        labels = segmentation.watershed(-dist_transform, markers, mask=basic_mask > 0)
        
        print(f"DEBUG Watershed: Watershed segmentation produced {labels.max()} regions")
        
        # === STEP 11: Post-filter by properties (CRITICAL IMPROVEMENT) ===
        props = measure.regionprops(labels, intensity_image=foreground)
        
        # Calculate intensity threshold (20th percentile of foreground)
        fg_pixels = foreground[basic_mask > 0]
        intensity_threshold = np.percentile(fg_pixels, 20) if len(fg_pixels) > 0 else 0
        
        valid_regions = 0
        mask = np.zeros_like(gray, dtype=np.uint8)
        
        for prop in props:
            # Filter by area
            if prop.area < min_size:
                continue
            
            # Calculate aspect ratio for adaptive filtering
            minor_axis = prop.minor_axis_length if prop.minor_axis_length > 0 else 1
            major_axis = prop.major_axis_length if prop.major_axis_length > 0 else 1
            aspect_ratio = major_axis / minor_axis
            
            # Adaptive solidity threshold based on shape
            # Fibers (high aspect ratio) have low solidity, fragments are more compact
            if aspect_ratio > 3.0:
                # Elongated object (fiber) - use lower solidity threshold
                solidity_threshold = 0.15
            elif aspect_ratio > 2.0:
                # Moderately elongated - medium threshold
                solidity_threshold = 0.25
            else:
                # Compact object (fragment/particle) - stricter threshold
                solidity_threshold = 0.4
            
            # Filter by solidity (remove debris/noise)
            if prop.solidity < solidity_threshold:
                continue
            
            # Filter by mean intensity (remove dark artifacts)
            if prop.mean_intensity < intensity_threshold:
                continue
            
            # Accept this region
            region_mask = (labels == prop.label)
            mask[region_mask] = 255
            valid_regions += 1
        
        print(f"DEBUG Watershed: Kept {valid_regions} valid regions after filtering")
        
        # === STEP 12: Fallback if no valid regions ===
        if np.sum(mask > 0) == 0:
            print("DEBUG Watershed: No valid regions, using basic mask")
            mask = basic_mask
        
        return mask, 127
    
    @staticmethod
    def _preprocess_edge(img, min_size=20, **kwargs):
        """
        [QUICK] Edge-based particle detection.
        
        Fast detection using Canny edge detection combined with
        threshold-based segmentation. Good for particles with
        well-defined boundaries.
        
        Processing Pipeline:
            1. Background detection and adaptation
            2. Histogram equalization (fluorescent) or inversion (brightfield)
            3. Canny edge detection
            4. Edge dilation and contour filling
            5. Mask combination and cleanup
        
        Supports:
            - Brightfield microscopy (white background)
            - Fluorescent microscopy (dark background)
        
        Args:
            img: BGR input image
            min_size: Minimum particle area in pixels
            **kwargs: Additional parameters (unused)
            
        Returns:
            tuple: (binary_mask, threshold_value)
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bg_color = ImageProcessor.detect_background_color(img_rgb)
        
        # === FLUORESCENT MICROSCOPY: Enhance bright particles ===
        if bg_color == 'black':
            # For fluorescent: particles are bright, enhance them
            # Don't invert - work directly with bright particles
            enhanced = cv2.equalizeHist(gray)
            blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
            
            # Calculate threshold for bright particles
            gray_mean = np.mean(gray)
            gray_std = np.std(gray)
            thresh_val = max(gray_mean + gray_std * 1.5, 30)
            _, basic_mask = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
            
        # === BRIGHTFIELD: Process dark particles on bright background ===
        elif bg_color == 'white':
            # For brightfield: particles are dark, invert for processing
            gray_inverted = 255 - gray
            blurred = cv2.GaussianBlur(gray_inverted, (5, 5), 0)
            
            # Use Otsu thresholding on inverted image to get dark particles
            _, basic_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
        # === UNKNOWN: Decide based on brightness ===
        else:
            if np.mean(gray) > 127:
                # Bright background - treat like brightfield
                gray_inverted = 255 - gray
                blurred = cv2.GaussianBlur(gray_inverted, (5, 5), 0)
                _, basic_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            else:
                # Dark background - treat like fluorescent
                enhanced = cv2.equalizeHist(gray)
                blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
                gray_mean = np.mean(gray)
                gray_std = np.std(gray)
                thresh_val = max(gray_mean + gray_std * 1.5, 30)
                _, basic_mask = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
        
        # === Edge detection on blurred image ===
        edges = cv2.Canny(blurred, 30, 120)
        
        # === Dilate edges to close gaps ===
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges_dilated = cv2.dilate(edges, kernel_dilate, iterations=2)
        
        # === Find and fill contours ===
        contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        edge_mask = np.zeros_like(gray)
        for cnt in contours:
            if cv2.contourArea(cnt) >= min_size * 0.8:  # Slightly less strict than min_size
                cv2.drawContours(edge_mask, [cnt], -1, 255, -1)
        
        # === Combine edge and threshold masks ===
        combined = cv2.bitwise_or(edge_mask, basic_mask)
        
        # === If detection found nothing, fallback to basic ===
        if np.sum(combined > 0) < (combined.size * 0.001):
            return ImageProcessor._preprocess_basic(img, min_size, **kwargs)
        
        # === Morphology cleanup ===
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        # === Fill holes ===
        filled = ndi.binary_fill_holes(cleaned > 0).astype(np.uint8) * 255
        
        # === Remove small components ===
        mask_bool = filled > 0
        cleaned = ImageProcessor._remove_small_components(mask_bool, min_size)
        mask = (cleaned * 255).astype(np.uint8)
        
        return mask, 127
    
    @staticmethod
    def _preprocess_adaptive(img, min_size=20, **kwargs):
        """
        [DEEP] Adaptive thresholding with background subtraction.
        
        Advanced per-channel processing with morphological background
        removal and adaptive thresholding. Best for samples with uneven
        illumination or complex backgrounds.
        
        Processing Pipeline:
            1. CLAHE enhancement per RGB channel
            2. Background subtraction (tophat transform per channel)
            3. Adaptive thresholding (Gaussian method)
            4. Conservative morphological refinement
            5. Hole filling with boundary preservation
        
        Supports:
            - Uneven illumination conditions
            - Complex/noisy backgrounds
            - Color-diverse samples
            - Both microscopy types
        
        Args:
            img: BGR input image
            min_size: Minimum particle area in pixels
            **kwargs:
                - adaptive_c_value: Threshold constant (default: 2)
                - expected_max_diameter: Max particle diameter for kernel sizing (default: 200)
            
        Returns:
            tuple: (binary_mask, threshold_value)
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bg_color = ImageProcessor.detect_background_color(img_rgb)
        
        adaptive_c = kwargs.get('adaptive_c_value', 2)
        expected_max_diam = kwargs.get('expected_max_diameter', 200)  # particle size estimate
        
        print(f"DEBUG Adaptive: Processing RGB channels with background subtraction...")
        
        # CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        
        # === STEP 1: Process each RGB channel with background subtraction ===
        r_channel, g_channel, b_channel = cv2.split(img_rgb)
        channel_masks = []
        
        # Calculate adaptive block size based on image dimensions
        h, w = r_channel.shape
        block_size = max(15, int(min(h, w) / 50))
        if block_size % 2 == 0:
            block_size += 1
        
        # Background removal kernel (larger than particles)
        bg_kernel_size = max(int(expected_max_diam * 1.2), 15)
        if bg_kernel_size % 2 == 0:
            bg_kernel_size += 1
        bg_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bg_kernel_size, bg_kernel_size))
        
        for idx, (channel, name) in enumerate([(r_channel, 'Red'), (g_channel, 'Green'), (b_channel, 'Blue')]):
            # Check channel validity
            if np.std(channel) < 5:
                print(f"DEBUG Adaptive: {name} channel low variation, skipping")
                continue
            
            # === STEP 1.1: Contrast enhancement ===
            ch_enhanced = clahe.apply(channel)
            
            # === STEP 1.2: Background subtraction (tophat transform) ===
            # Estimate background by morphological opening (removes small bright objects)
            ch_background = cv2.morphologyEx(ch_enhanced, cv2.MORPH_OPEN, bg_kernel)
            # Subtract background to get foreground (particles)
            ch_foreground = cv2.subtract(ch_enhanced, ch_background)
            
            # === STEP 1.3: Noise reduction ===
            ch_denoised = cv2.GaussianBlur(ch_foreground, (3, 3), 0)
            
            # === STEP 1.4: Adaptive thresholding ===
            if bg_color == 'black':
                # FLUORESCENT: Use STRICTER global threshold on background-subtracted image
                mean_val = np.mean(ch_denoised)
                std_val = np.std(ch_denoised)
                # Increased multiplier to reduce over-segmentation
                thresh_val = max(mean_val + std_val * 2.5, 30)  # Stricter than 1.5
                
                _, ch_mask = cv2.threshold(ch_denoised, thresh_val, 255, cv2.THRESH_BINARY)
            else:
                # BRIGHTFIELD: Use STRICTER adaptive threshold
                if np.max(ch_denoised) < 10:
                    print(f"DEBUG Adaptive: {name} channel too dark after bg subtraction")
                    continue
                
                # Increase C parameter for stricter thresholding
                ch_mask = cv2.adaptiveThreshold(
                    ch_denoised, 255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,
                    block_size, adaptive_c + 3  # Stricter: was adaptive_c, now +3
                )
            
            # === STEP 1.5: Morphological cleanup - EROSION FIRST to reduce area ===
            kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            # Erode first to shrink over-detected regions
            ch_mask = cv2.erode(ch_mask, kernel_small, iterations=1)
            # Then open to remove small noise
            ch_mask = cv2.morphologyEx(ch_mask, cv2.MORPH_OPEN, kernel_small)
            # Minimal closing (only 1 iteration) to avoid expansion
            ch_mask = cv2.morphologyEx(ch_mask, cv2.MORPH_CLOSE, kernel_small, iterations=1)
            
            # === STEP 1.6: Filter by detected content ===
            detected_pixels = np.sum(ch_mask > 0)
            if detected_pixels > min_size * 2:
                channel_masks.append(ch_mask)
                print(f"DEBUG Adaptive: {name} detected {detected_pixels} pixels")
        
        # === STEP 2: Combine RGB channel results ===
        if len(channel_masks) == 0:
            print("DEBUG Adaptive: No valid channels, using grayscale fallback with bg subtraction")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_enhanced = clahe.apply(gray)
            
            # Background subtraction on grayscale
            gray_bg = cv2.morphologyEx(gray_enhanced, cv2.MORPH_OPEN, bg_kernel)
            gray_fg = cv2.subtract(gray_enhanced, gray_bg)
            gray_denoised = cv2.GaussianBlur(gray_fg, (3, 3), 0)
            
            if bg_color == 'black':
                mean_val = np.mean(gray_denoised)
                std_val = np.std(gray_denoised)
                # Stricter threshold for grayscale fallback
                thresh_val = max(mean_val + std_val * 2.5, 30)
                _, mask = cv2.threshold(gray_denoised, thresh_val, 255, cv2.THRESH_BINARY)
            else:
                mask = cv2.adaptiveThreshold(
                    gray_denoised, 255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,
                    block_size, adaptive_c + 3  # Stricter
                )
        else:
            # Combine all valid channel masks with OR operation
            mask = channel_masks[0]
            for ch_mask in channel_masks[1:]:
                mask = cv2.bitwise_or(mask, ch_mask)
            print(f"DEBUG Adaptive: Combined {len(channel_masks)} channel masks")
        
        # === STEP 3: Conservative morphological refinement - AVOID EXPANSION ===
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))  # Smaller kernel (was 3x3)
        
        # Remove small noise with opening (doesn't expand)
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        # Minimal closing to avoid area expansion (was 2 iterations, now 1)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # === STEP 4: Fill holes (but apply erosion after to compensate) ===
        filled = ndi.binary_fill_holes(cleaned > 0).astype(np.uint8) * 255
        
        # Additional erosion to shrink boundaries back to true size
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        filled = cv2.erode(filled, kernel_erode, iterations=1)
        
        # === STEP 5: Remove small components ===
        mask_bool = filled > 0
        if np.sum(mask_bool) > 0:
            cleaned = ImageProcessor._remove_small_components(mask_bool, min_size)
            final_mask = (cleaned * 255).astype(np.uint8)
        else:
            final_mask = filled
        
        # === STEP 6: NO FINAL DILATION - Keep tight boundaries ===
        # Removed final dilation to prevent area expansion
        # Original boundaries are now preserved for accurate area measurements
        
        print(f"DEBUG Adaptive: Final mask has {np.sum(final_mask > 0)} pixels")
        return final_mask, 127

    # =========================================================================
    # BRIGHTFIELD-SPECIFIC METHODS
    # =========================================================================
    
    @staticmethod
    def _preprocess_brightfield_basic(img, min_size=20, **kwargs):
        """
        [QUICK] Fast particle detection for brightfield microscopy.
        
        Optimized for white/light backgrounds with colored or dark particles.
        Uses saturation-based detection to find colored particles and
        intensity-based detection for dark particles.
        
        Processing Pipeline:
            1. Convert to HSV color space
            2. Detect colored particles via saturation channel
            3. Detect dark particles via inverted value channel
            4. Combine masks and apply morphology
        
        Args:
            img: BGR input image
            min_size: Minimum particle area in pixels
            **kwargs: Additional parameters
            
        Returns:
            tuple: (binary_mask, threshold_value)
        """
        print("DEBUG Brightfield Basic: Starting brightfield-optimized detection...")
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Convert to HSV for color-based detection
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h_channel, s_channel, v_channel = cv2.split(hsv)
        
        # === Strategy 1: Saturation-based detection (colored particles) ===
        # Colored particles have high saturation, white/gray background has low saturation
        # Use CLAHE to enhance saturation contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        s_enhanced = clahe.apply(s_channel)
        
        # Very low threshold - white background has S near 0
        s_thresh = 12
        _, sat_mask = cv2.threshold(s_enhanced, s_thresh, 255, cv2.THRESH_BINARY)
        print(f"DEBUG Brightfield Basic: Saturation threshold={s_thresh}, detected={np.sum(sat_mask>0)} pixels")
        
        # === Strategy 2: RGB channel difference (specific colors) ===
        r, g, b = cv2.split(img_rgb)
        
        # Red particles: R much higher than G and B
        red_diff = r.astype(np.float32) - np.maximum(g, b).astype(np.float32)
        red_mask = (red_diff > 15).astype(np.uint8) * 255
        
        # Pink/Magenta detection: R high, R and B similar, G lower
        pink_cond = (r > 150) & (r > g) & (np.abs(r.astype(np.int16) - b.astype(np.int16)) < 60)
        pink_mask = (pink_cond).astype(np.uint8) * 255
        
        # Blue particles: B much higher than R and G  
        blue_diff = b.astype(np.float32) - np.maximum(r, g).astype(np.float32)
        blue_mask = (blue_diff > 15).astype(np.uint8) * 255
        
        # Green particles: G much higher than R and B
        green_diff = g.astype(np.float32) - np.maximum(r, b).astype(np.float32)
        green_mask = (green_diff > 15).astype(np.uint8) * 255
        
        # Dark/colored detection: pixels not white (min channel < 220)
        darkness = np.minimum(np.minimum(r, g), b)
        dark_mask = (darkness < 220).astype(np.uint8) * 255
        dark_colored = cv2.bitwise_and(dark_mask, sat_mask)
        
        color_mask = cv2.bitwise_or(red_mask, pink_mask)
        color_mask = cv2.bitwise_or(color_mask, blue_mask)
        color_mask = cv2.bitwise_or(color_mask, green_mask)
        color_mask = cv2.bitwise_or(color_mask, dark_colored)
        print(f"DEBUG Brightfield Basic: Color channel mask detected={np.sum(color_mask>0)} pixels")
        
        # === Strategy 3: Value-based detection (dark particles) ===
        # Invert value channel - dark particles become bright
        v_inverted = 255 - v_channel
        v_blurred = cv2.GaussianBlur(v_inverted, (5, 5), 0)
        
        # Otsu threshold on inverted value
        _, val_mask = cv2.threshold(v_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        print(f"DEBUG Brightfield Basic: Value mask detected={np.sum(val_mask>0)} pixels")
        
        # === Combine all strategies ===
        combined = cv2.bitwise_or(sat_mask, color_mask)
        combined = cv2.bitwise_or(combined, val_mask)
        
        # === Morphological cleanup ===
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # === Fill holes ===
        filled = ndi.binary_fill_holes(cleaned > 0).astype(np.uint8) * 255
        
        # === Remove small components ===
        mask_bool = filled > 0
        if np.sum(mask_bool) > 0:
            cleaned = ImageProcessor._remove_small_components(mask_bool, min_size)
            mask = (cleaned * 255).astype(np.uint8)
        else:
            mask = filled
        
        print(f"DEBUG Brightfield Basic: Final mask has {np.sum(mask > 0)} pixels")
        return mask, int(s_thresh)
    
    @staticmethod
    def _preprocess_brightfield(img, min_size=20, **kwargs):
        """
        [DEEP] Advanced particle detection for brightfield microscopy.
        
        Comprehensive multi-strategy detection for white/light backgrounds.
        Combines saturation, color difference, and grayscale analysis
        for robust detection of colored and dark particles.
        
        Processing Pipeline:
            1. HSV saturation-based detection (colored particles)
            2. Color difference from white (deviation detection)
            3. LAB color space analysis (perceptual color difference)
            4. Grayscale intensity analysis (dark particles)
            5. Intelligent combination and refinement
        
        Args:
            img: BGR input image
            min_size: Minimum particle area in pixels
            **kwargs: Additional parameters
            
        Returns:
            tuple: (binary_mask, threshold_value)
        """
        print("DEBUG Brightfield Advanced: Starting multi-strategy detection...")
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        masks = []
        
        # === Strategy 1: HSV Saturation (colored particles) ===
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h_channel, s_channel, v_channel = cv2.split(hsv)
        
        # CLAHE on saturation for better contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        s_enhanced = clahe.apply(s_channel)
        
        # Use very low saturation threshold - any color stands out against white
        # White background has S close to 0, colored particles have S > 10-20
        s_thresh = 15  # Fixed low threshold for colored particles on white
        
        _, sat_mask = cv2.threshold(s_enhanced, s_thresh, 255, cv2.THRESH_BINARY)
        
        if np.sum(sat_mask > 0) > min_size * 2:
            masks.append(sat_mask)
            print(f"DEBUG Brightfield: Saturation detected {np.sum(sat_mask>0)} pixels (thresh={s_thresh})")
        
        # === Strategy 2: Color difference from background ===
        # Sample background color from corners (should be white/light gray)
        h, w = img_rgb.shape[:2]
        corner_size = min(50, h//10, w//10)
        corners = [
            img_rgb[:corner_size, :corner_size],
            img_rgb[:corner_size, w-corner_size:],
            img_rgb[h-corner_size:, :corner_size],
            img_rgb[h-corner_size:, w-corner_size:]
        ]
        bg_color = np.mean([np.mean(c, axis=(0,1)) for c in corners], axis=0).astype(np.float32)
        print(f"DEBUG Brightfield: Detected background color RGB={bg_color}")
        
        img_float = img_rgb.astype(np.float32)
        
        # Calculate color distance from detected background
        color_diff = np.sqrt(np.sum((img_float - bg_color) ** 2, axis=2))
        
        # Normalize and threshold
        if color_diff.max() > 0:
            color_diff_norm = (color_diff / color_diff.max() * 255).astype(np.uint8)
            
            # Use Otsu to find optimal threshold
            _, diff_mask = cv2.threshold(color_diff_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            if np.sum(diff_mask > 0) > min_size * 2:
                masks.append(diff_mask)
                print(f"DEBUG Brightfield: Color diff detected {np.sum(diff_mask>0)} pixels")
        
        # === Strategy 3: LAB color space (perceptual difference) ===
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # A and B channels contain color information
        # For neutral (white/gray) pixels, A and B are around 128
        a_diff = np.abs(a_channel.astype(np.float32) - 128)
        b_diff = np.abs(b_channel.astype(np.float32) - 128)
        ab_diff = np.sqrt(a_diff**2 + b_diff**2).astype(np.uint8)
        
        # Lower threshold for LAB - even small color deviation is significant
        ab_thresh = 8  # Low fixed threshold
        
        _, ab_mask = cv2.threshold(ab_diff, ab_thresh, 255, cv2.THRESH_BINARY)
        
        if np.sum(ab_mask > 0) > min_size * 2:
            masks.append(ab_mask)
            print(f"DEBUG Brightfield: LAB detected {np.sum(ab_mask>0)} pixels")
        
        # === Strategy 4: Specific color channel detection ===
        # For red particles: high R, low G and B
        r, g, b = cv2.split(img_rgb)
        
        # Red detection: R much higher than G and B
        red_diff = r.astype(np.float32) - np.maximum(g, b).astype(np.float32)
        red_mask = (red_diff > 15).astype(np.uint8) * 255
        
        # Pink/Magenta detection: R high, B moderate, G low
        # Pink = R high, R and B similar, G lower than both
        pink_cond = (r > 150) & (r > g) & (np.abs(r.astype(np.int16) - b.astype(np.int16)) < 60)
        pink_mask = (pink_cond).astype(np.uint8) * 255
        
        # Blue detection: B much higher than R and G
        blue_diff = b.astype(np.float32) - np.maximum(r, g).astype(np.float32)
        blue_mask = (blue_diff > 15).astype(np.uint8) * 255
        
        # Green detection: G much higher than R and B
        green_diff = g.astype(np.float32) - np.maximum(r, b).astype(np.float32)
        green_mask = (green_diff > 15).astype(np.uint8) * 255
        
        # Dark/non-white detection: pixels significantly darker than white
        # White background has all channels > 240, particles are darker
        darkness = np.minimum(np.minimum(r, g), b)
        dark_mask = (darkness < 220).astype(np.uint8) * 255
        # Only keep if saturation also indicates color
        dark_colored = cv2.bitwise_and(dark_mask, sat_mask)
        
        color_channel_mask = cv2.bitwise_or(red_mask, pink_mask)
        color_channel_mask = cv2.bitwise_or(color_channel_mask, blue_mask)
        color_channel_mask = cv2.bitwise_or(color_channel_mask, green_mask)
        color_channel_mask = cv2.bitwise_or(color_channel_mask, dark_colored)
        
        if np.sum(color_channel_mask > 0) > min_size * 2:
            masks.append(color_channel_mask)
            print(f"DEBUG Brightfield: RGB channel detected {np.sum(color_channel_mask>0)} pixels")
        
        # === Strategy 5: Grayscale intensity (dark particles) ===
        gray_inverted = 255 - gray
        gray_enhanced = clahe.apply(gray_inverted)
        
        _, gray_mask = cv2.threshold(gray_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Only use if not too much detected (avoid false positives from texture)
        if np.sum(gray_mask > 0) > min_size * 2 and np.sum(gray_mask > 0) < gray_mask.size * 0.3:
            masks.append(gray_mask)
            print(f"DEBUG Brightfield: Grayscale detected {np.sum(gray_mask>0)} pixels")
        
        # === Combine all strategies ===
        if len(masks) == 0:
            print("DEBUG Brightfield: No strategy worked, using fallback")
            # Fallback: simple saturation threshold
            _, combined = cv2.threshold(s_channel, 10, 255, cv2.THRESH_BINARY)
        elif len(masks) == 1:
            combined = masks[0]
        else:
            # Use OR to catch all particles from any strategy
            combined = masks[0]
            for m in masks[1:]:
                combined = cv2.bitwise_or(combined, m)
            print(f"DEBUG Brightfield: Combined {len(masks)} strategies with OR")
            
            # If result is too large (>30% of image), use intersection for refinement
            if np.sum(combined > 0) > combined.size * 0.3:
                print("DEBUG Brightfield: Too many detections, using AND intersection")
                combined = masks[0]
                for m in masks[1:]:
                    combined = cv2.bitwise_and(combined, m)
        
        # === Morphological refinement ===
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # Remove noise
        refined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_open, iterations=1)
        # Close gaps
        refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel_close, iterations=2)
        
        # === Fill holes ===
        filled = ndi.binary_fill_holes(refined > 0).astype(np.uint8) * 255
        
        # === Remove small components ===
        mask_bool = filled > 0
        if np.sum(mask_bool) > 0:
            cleaned = ImageProcessor._remove_small_components(mask_bool, min_size)
            final_mask = (cleaned * 255).astype(np.uint8)
        else:
            final_mask = filled
        
        print(f"DEBUG Brightfield Advanced: Final mask has {np.sum(final_mask > 0)} pixels")
        return final_mask, 127
    
    @staticmethod
    def _preprocess_brightfield_watershed(img, min_size=20, **kwargs):
        """
        [DEEP] Watershed segmentation for brightfield microscopy.
        
        Combines brightfield-optimized detection with watershed segmentation
        for separating touching particles on white/light backgrounds.
        
        Processing Pipeline:
            1. Brightfield detection (saturation + color difference)
            2. Distance transform
            3. Marker-based watershed segmentation
            4. Region filtering
        
        Args:
            img: BGR input image
            min_size: Minimum particle area in pixels
            **kwargs:
                - distance_thresh: Distance transform threshold (default: 0.5)
            
        Returns:
            tuple: (binary_mask, threshold_value)
        """
        print("DEBUG Brightfield Watershed: Starting watershed segmentation...")
        
        distance_thresh = kwargs.get('distance_thresh', 0.5)
        
        # First get basic brightfield mask
        basic_mask, _ = ImageProcessor._preprocess_brightfield(img, min_size, **kwargs)
        
        if np.sum(basic_mask > 0) == 0:
            print("DEBUG Brightfield Watershed: No objects detected in basic mask")
            return basic_mask, 127
        
        # === Distance transform ===
        dist_transform = ndi.distance_transform_edt(basic_mask > 0)
        max_dist = np.max(dist_transform)
        
        if max_dist == 0:
            print("DEBUG Brightfield Watershed: Distance transform is zero")
            return basic_mask, 127
        
        # === Find markers using h-maxima ===
        h_factor = kwargs.get('h_max_factor', 0.35)
        h_value = max(1, int(h_factor * max_dist))
        
        try:
            from skimage import morphology as sk_morph
            h_maxima_result = sk_morph.h_maxima(dist_transform, h=h_value)
            markers, num_markers = ndi.label(h_maxima_result)
            print(f"DEBUG Brightfield Watershed: H-maxima found {num_markers} markers")
        except Exception as e:
            print(f"DEBUG Brightfield Watershed: H-maxima failed ({e}), using threshold")
            thresh_value = int(max_dist * distance_thresh)
            markers = ndi.label(dist_transform > thresh_value)[0]
            num_markers = markers.max()
        
        if num_markers < 2:
            print("DEBUG Brightfield Watershed: Too few markers, returning basic mask")
            return basic_mask, 127
        
        # === Apply watershed ===
        labels = segmentation.watershed(-dist_transform, markers, mask=basic_mask > 0)
        
        print(f"DEBUG Brightfield Watershed: Watershed produced {labels.max()} regions")
        
        # === Filter regions by properties ===
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        props = measure.regionprops(labels)
        
        mask = np.zeros_like(gray, dtype=np.uint8)
        valid_count = 0
        
        for prop in props:
            # Filter by area
            if prop.area < min_size:
                continue
            
            # Filter by solidity (remove debris)
            if prop.solidity < 0.3:
                continue
            
            # Accept region
            region_mask = (labels == prop.label)
            mask[region_mask] = 255
            valid_count += 1
        
        print(f"DEBUG Brightfield Watershed: Kept {valid_count} valid regions")
        
        if np.sum(mask > 0) == 0:
            return basic_mask, 127
        
        return mask, 127