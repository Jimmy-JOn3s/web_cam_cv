"""
Image Processor - Main class for processing images with various effects
"""
import cv2
import numpy as np
from enum import Enum


class ColorSpace(Enum):
    """Enumeration for different color spaces."""
    BGR = "BGR"
    RGB = "RGB"
    GRAY = "GRAY"
    HSV = "HSV"


class ImageProcessor:
    """Main image processing class that handles various image transformations."""
    
    def __init__(self):
        """Initialize the image processor."""
        self.current_color_space = ColorSpace.BGR
        
    def adjust_brightness_contrast(self, image, brightness=0, contrast=1.0):
        """
        Adjust brightness and contrast of an image.
        
        Args:
            image (numpy.ndarray): Input image
            brightness (int): Brightness adjustment (-100 to 100)
            contrast (float): Contrast multiplier (0.1 to 3.0)
            
        Returns:
            numpy.ndarray: Adjusted image
        """
        return cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
    
    def convert_color_space(self, image, target_space):
        """
        Convert image to different color space.
        
        Args:
            image (numpy.ndarray): Input image
            target_space (ColorSpace): Target color space
            
        Returns:
            numpy.ndarray: Converted image
        """
        if target_space == ColorSpace.GRAY:
            if len(image.shape) == 3:
                return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return image
        elif target_space == ColorSpace.HSV:
            if len(image.shape) == 3:
                return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            else:
                # Convert grayscale to BGR first, then to HSV
                bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        elif target_space == ColorSpace.RGB:
            if len(image.shape) == 3:
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:  # BGR
            if len(image.shape) == 2:
                return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            return image
    
    def apply_gaussian_filter(self, image, kernel_size=5, sigma_x=1.0, sigma_y=None):
        """
        Apply Gaussian blur filter.
        
        Args:
            image (numpy.ndarray): Input image
            kernel_size (int): Size of the Gaussian kernel (must be odd)
            sigma_x (float): Standard deviation in X direction
            sigma_y (float): Standard deviation in Y direction (if None, uses sigma_x)
            
        Returns:
            numpy.ndarray: Filtered image
        """
        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        if sigma_y is None:
            sigma_y = sigma_x
            
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma_x, sigmaY=sigma_y)
    
    def apply_bilateral_filter(self, image, d=9, sigma_color=75, sigma_space=75):
        """
        Apply bilateral filter (edge-preserving smoothing).
        
        Args:
            image (numpy.ndarray): Input image
            d (int): Diameter of each pixel neighborhood
            sigma_color (float): Filter sigma in the color space
            sigma_space (float): Filter sigma in the coordinate space
            
        Returns:
            numpy.ndarray: Filtered image
        """
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    
    def process_frame(self, frame, **kwargs):
        """
        Process a frame with multiple effects.
        
        Args:
            frame (numpy.ndarray): Input frame
            **kwargs: Processing parameters
            
        Returns:
            numpy.ndarray: Processed frame
        """
        processed = frame.copy()
        
        # Apply brightness and contrast
        if 'brightness' in kwargs or 'contrast' in kwargs:
            brightness = kwargs.get('brightness', 0)
            contrast = kwargs.get('contrast', 1.0)
            processed = self.adjust_brightness_contrast(processed, brightness, contrast)
        
        # Apply filters
        if kwargs.get('gaussian_blur', False):
            kernel_size = kwargs.get('gaussian_kernel', 5)
            sigma = kwargs.get('gaussian_sigma', 1.0)
            processed = self.apply_gaussian_filter(processed, kernel_size, sigma)
        
        if kwargs.get('bilateral_filter', False):
            d = kwargs.get('bilateral_d', 9)
            sigma_color = kwargs.get('bilateral_sigma_color', 75)
            sigma_space = kwargs.get('bilateral_sigma_space', 75)
            processed = self.apply_bilateral_filter(processed, d, sigma_color, sigma_space)
        
        # Color space conversion
        if 'color_space' in kwargs:
            processed = self.convert_color_space(processed, kwargs['color_space'])
        
        return processed
