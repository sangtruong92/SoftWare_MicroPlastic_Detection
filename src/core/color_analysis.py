"""
Color Analysis Module
Contains algorithms for color classification
"""

import cv2
import numpy as np
from typing import Tuple

from config.constants import COLOR_RANGES


class ColorAnalyzer:
    """Color classification algorithms"""
    
    @staticmethod
    def classify_color(h, s, v=None):
        """
        Classify color based on hue, saturation, and value/brightness.
        
        Args:
            h: Hue value (0-1 normalized)
            s: Saturation value (0-1 normalized)
            v: Value/brightness (0-1 normalized, optional)
            
        Returns:
            str: Color name
        """
        # Lower threshold (0.08) for brightfield where particles have subtle color
        if s < 0.08:
            if v is not None:
                if v < 0.2:
                    return "Black"
                if v > 0.8:
                    return "White"
            return "Gray"
        
        # Classify into 4 main fluorescent colors: Red, Green, Blue, Yellow
        # Red includes orange-red, pink, and magenta ranges
        if h < 0.08 or h > 0.92:
            return "Red"
        if h < 0.20:
            return "Yellow"  # Pure yellow range
        if h < 0.48:
            return "Green"   # Includes cyan-green range
        if h < 0.70:
            return "Blue"    # Includes blue-purple range
        # Pink/Magenta range (0.70 - 0.92) -> classify as Red
        return "Red"
    
    @staticmethod
    def wavelength_to_rgb(wavelength_nm, gamma=0.8):
        """
        Approximate conversion from wavelength (nm) to RGB tuple.
        
        Args:
            wavelength_nm: Wavelength in nanometers (380-780)
            gamma: Gamma correction factor
            
        Returns:
            tuple: (R, G, B) values (0-255)
        """
        w = float(wavelength_nm)
        if w < 380 or w > 780:
            return (0, 0, 0)
        
        if w < 440:
            attenuation = 0.3 + 0.7 * (w - 380) / 60
            r = ((-(w - 440) / 60) * attenuation) ** gamma
            g = 0.0
            b = (1.0 * attenuation) ** gamma
        elif w < 490:
            r = 0.0
            g = ((w - 440) / 50) ** gamma
            b = 1.0 ** gamma
        elif w < 510:
            r = 0.0
            g = 1.0 ** gamma
            b = (-(w - 510) / 20) ** gamma
        elif w < 580:
            r = ((w - 510) / 70) ** gamma
            g = 1.0 ** gamma
            b = 0.0
        elif w < 645:
            r = 1.0 ** gamma
            g = (-(w - 645) / 65) ** gamma
            b = 0.0
        elif w < 700:
            # Pure red range - reduce green contamination
            r = 1.0 ** gamma
            g = 0.0
            b = 0.0
        else:
            # Deep red with attenuation
            attenuation = 0.3 + 0.7 * (780 - w) / 80
            r = (1.0 * attenuation) ** gamma
            g = 0.0
            b = 0.0
        
        r = int(np.clip(r, 0.0, 1.0) * 255)
        g = int(np.clip(g, 0.0, 1.0) * 255)
        b = int(np.clip(b, 0.0, 1.0) * 255)
        return (r, g, b)
    
    @staticmethod
    def extract_color_from_region(image, mask):
        """
        Extract dominant color from a masked region.
        
        Uses robust percentile-based extraction to handle small particles
        on brightfield (white backgrounds) where edge pixels may be contaminated.
        
        Args:
            image: RGB image
            mask: Binary mask of the region
            
        Returns:
            tuple: (color_name, h, s, v)
        """
        if image is None or mask is None:
            return "Unknown", 0, 0, 0
        
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Extract pixels in the mask
        masked_pixels = hsv[mask > 0]
        
        if len(masked_pixels) == 0:
            return "Unknown", 0, 0, 0
        
        n_pixels = len(masked_pixels)
        
        # For very small particles (< 50 pixels), use more robust extraction
        # to avoid background contamination at edges
        if n_pixels < 50:
            # Use top 50% most saturated pixels to avoid background contamination
            saturations = masked_pixels[:, 1]
            if np.max(saturations) > 10:  # Has some color
                # Get indices of top saturated pixels
                n_top = max(1, n_pixels // 2)
                top_indices = np.argsort(saturations)[-n_top:]
                top_pixels = masked_pixels[top_indices]
                
                h_val = np.median(top_pixels[:, 0]) / 179.0
                s_val = np.median(top_pixels[:, 1]) / 255.0
                v_val = np.median(top_pixels[:, 2]) / 255.0
            else:
                # Truly gray/white particle
                h_val = np.mean(masked_pixels[:, 0]) / 179.0
                s_val = np.mean(masked_pixels[:, 1]) / 255.0
                v_val = np.mean(masked_pixels[:, 2]) / 255.0
        else:
            # For larger particles, use percentile-based approach
            # This is more robust than mean for particles with edge contamination
            
            # Check if this might be brightfield (high V pixels present)
            v_max = np.max(masked_pixels[:, 2])
            s_max = np.max(masked_pixels[:, 1])
            
            if v_max > 200 and s_max > 20:
                # Brightfield with colored particle - focus on saturated pixels
                saturations = masked_pixels[:, 1]
                # Use top 70% most saturated to exclude white edge contamination
                threshold = np.percentile(saturations, 30)
                color_mask = saturations >= threshold
                
                if np.sum(color_mask) > 0:
                    color_pixels = masked_pixels[color_mask]
                    h_val = np.median(color_pixels[:, 0]) / 179.0
                    s_val = np.median(color_pixels[:, 1]) / 255.0
                    v_val = np.median(color_pixels[:, 2]) / 255.0
                else:
                    h_val = np.median(masked_pixels[:, 0]) / 179.0
                    s_val = np.median(masked_pixels[:, 1]) / 255.0
                    v_val = np.median(masked_pixels[:, 2]) / 255.0
            else:
                # Fluorescent/dark background - use median for robustness
                h_val = np.median(masked_pixels[:, 0]) / 179.0
                s_val = np.median(masked_pixels[:, 1]) / 255.0
                v_val = np.median(masked_pixels[:, 2]) / 255.0
        
        # Classify color
        color_name = ColorAnalyzer.classify_color(h_val, s_val, v_val)
        
        return color_name, h_val, s_val, v_val
    
    @staticmethod
    def get_fluorescent_color(wavelength_range: Tuple[int, int] = (400, 700)):
        """
        Generate a random fluorescent color within wavelength range.
        
        Args:
            wavelength_range: Tuple of (min_nm, max_nm)
            
        Returns:
            tuple: (R, G, B) values (0-255)
        """
        wavelength = np.random.randint(wavelength_range[0], wavelength_range[1])
        return ColorAnalyzer.wavelength_to_rgb(wavelength)
    
    @staticmethod
    def apply_glow_effect(image, intensity=0.4, sigma_range=(2.5, 6.0)):
        """
        Apply glow effect to an image.
        
        Args:
            image: RGB image
            intensity: Glow intensity (0-1)
            sigma_range: Tuple of (min_sigma, max_sigma) for Gaussian blur
            
        Returns:
            np.ndarray: Image with glow effect
        """
        if image is None or intensity <= 0:
            return image
        
        # Random sigma for blur
        sigma = np.random.uniform(sigma_range[0], sigma_range[1])
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(image, (0, 0), sigma)
        
        # Blend original with blurred
        glowed = cv2.addWeighted(image, 1.0, blurred, intensity, 0)
        
        return np.clip(glowed, 0, 255).astype(np.uint8)
