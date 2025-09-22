"""
Bilateral Filter - Implements bilateral filtering for edge-preserving smoothing
"""
import cv2
import numpy as np
from .base_filter import BaseFilter


class BilateralFilter(BaseFilter):
    """Bilateral filter implementation for edge-preserving smoothing."""
    
    def __init__(self):
        """Initialize Bilateral filter with default parameters."""
        super().__init__("Bilateral")
        self.parameters = {
            'd': 9,
            'sigma_color': 75,
            'sigma_space': 75
        }
    
    def apply(self, image, **kwargs):
        """
        Apply bilateral filter to the image.
        
        Args:
            image (numpy.ndarray): Input image
            **kwargs: Filter parameters (d, sigma_color, sigma_space)
            
        Returns:
            numpy.ndarray: Filtered image
        """
        # Update parameters with provided values
        d = kwargs.get('d', self.parameters['d'])
        sigma_color = kwargs.get('sigma_color', self.parameters['sigma_color'])
        sigma_space = kwargs.get('sigma_space', self.parameters['sigma_space'])
        
        # Ensure parameters are within valid ranges
        d = max(1, min(50, d))
        sigma_color = max(1, min(200, sigma_color))
        sigma_space = max(1, min(200, sigma_space))
        
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    
    def get_parameter_info(self):
        """
        Get information about bilateral filter parameters.
        
        Returns:
            dict: Parameter information with ranges and descriptions
        """
        return {
            'd': {
                'type': int,
                'range': (1, 50),
                'default': 9,
                'description': 'Diameter of each pixel neighborhood',
                'step': 1
            },
            'sigma_color': {
                'type': int,
                'range': (1, 200),
                'default': 75,
                'description': 'Filter sigma in the color space (larger = more colors mixed)',
                'step': 1
            },
            'sigma_space': {
                'type': int,
                'range': (1, 200),
                'default': 75,
                'description': 'Filter sigma in the coordinate space (larger = more distant pixels)',
                'step': 1
            }
        }
    
    def create_trackbars(self, window_name):
        """
        Create trackbars for real-time parameter adjustment.
        
        Args:
            window_name (str): Name of the OpenCV window
        """
        param_info = self.get_parameter_info()
        
        # d (diameter) trackbar
        d_max = param_info['d']['range'][1]
        d_default = param_info['d']['default']
        cv2.createTrackbar('Bilateral d', window_name, d_default, d_max, lambda x: None)
        
        # Sigma color trackbar
        sigma_color_max = param_info['sigma_color']['range'][1]
        sigma_color_default = param_info['sigma_color']['default']
        cv2.createTrackbar('Bilateral Sigma Color', window_name, sigma_color_default, sigma_color_max, lambda x: None)
        
        # Sigma space trackbar
        sigma_space_max = param_info['sigma_space']['range'][1]
        sigma_space_default = param_info['sigma_space']['default']
        cv2.createTrackbar('Bilateral Sigma Space', window_name, sigma_space_default, sigma_space_max, lambda x: None)
    
    def get_trackbar_values(self, window_name):
        """
        Get current trackbar values and convert to filter parameters.
        
        Args:
            window_name (str): Name of the OpenCV window
            
        Returns:
            dict: Current parameter values
        """
        d = cv2.getTrackbarPos('Bilateral d', window_name)
        sigma_color = cv2.getTrackbarPos('Bilateral Sigma Color', window_name)
        sigma_space = cv2.getTrackbarPos('Bilateral Sigma Space', window_name)
        
        return {
            'd': max(1, d),
            'sigma_color': max(1, sigma_color),
            'sigma_space': max(1, sigma_space)
        }
