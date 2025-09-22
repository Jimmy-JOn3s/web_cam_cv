"""
Gaussian Filter - Implements Gaussian blur filtering
"""
import cv2
import numpy as np
from .base_filter import BaseFilter


class GaussianFilter(BaseFilter):
    """Gaussian blur filter implementation."""
    
    def __init__(self):
        """Initialize Gaussian filter with default parameters."""
        super().__init__("Gaussian Blur")
        self.parameters = {
            'kernel_size': 5,
            'sigma_x': 1.0,
            'sigma_y': 1.0
        }
    
    def apply(self, image, **kwargs):
        """
        Apply Gaussian blur to the image.
        
        Args:
            image (numpy.ndarray): Input image
            **kwargs: Filter parameters (kernel_size, sigma_x, sigma_y)
            
        Returns:
            numpy.ndarray: Blurred image
        """
        # Update parameters with provided values
        kernel_size = kwargs.get('kernel_size', self.parameters['kernel_size'])
        sigma_x = kwargs.get('sigma_x', self.parameters['sigma_x'])
        sigma_y = kwargs.get('sigma_y', self.parameters['sigma_y'])
        
        # Ensure kernel size is odd and positive
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel_size = max(1, kernel_size)
        
        # Ensure sigma values are positive
        sigma_x = max(0.1, sigma_x)
        sigma_y = max(0.1, sigma_y)
        
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma_x, sigmaY=sigma_y)
    
    def get_parameter_info(self):
        """
        Get information about Gaussian filter parameters.
        
        Returns:
            dict: Parameter information with ranges and descriptions
        """
        return {
            'kernel_size': {
                'type': int,
                'range': (1, 99),
                'default': 5,
                'description': 'Size of the Gaussian kernel (must be odd)',
                'step': 2
            },
            'sigma_x': {
                'type': float,
                'range': (0.1, 10.0),
                'default': 1.0,
                'description': 'Standard deviation in X direction',
                'step': 0.1
            },
            'sigma_y': {
                'type': float,
                'range': (0.1, 10.0),
                'default': 1.0,
                'description': 'Standard deviation in Y direction',
                'step': 0.1
            }
        }
    
    def create_trackbars(self, window_name):
        """
        Create trackbars for real-time parameter adjustment.
        
        Args:
            window_name (str): Name of the OpenCV window
        """
        param_info = self.get_parameter_info()
        
        # Kernel size trackbar (scaled for trackbar)
        kernel_max = param_info['kernel_size']['range'][1]
        kernel_default = param_info['kernel_size']['default']
        cv2.createTrackbar('Gaussian Kernel', window_name, kernel_default, kernel_max, lambda x: None)
        
        # Sigma X trackbar (scaled by 10 for precision)
        sigma_x_max = int(param_info['sigma_x']['range'][1] * 10)
        sigma_x_default = int(param_info['sigma_x']['default'] * 10)
        cv2.createTrackbar('Gaussian Sigma X', window_name, sigma_x_default, sigma_x_max, lambda x: None)
        
        # Sigma Y trackbar (scaled by 10 for precision)
        sigma_y_max = int(param_info['sigma_y']['range'][1] * 10)
        sigma_y_default = int(param_info['sigma_y']['default'] * 10)
        cv2.createTrackbar('Gaussian Sigma Y', window_name, sigma_y_default, sigma_y_max, lambda x: None)
    
    def get_trackbar_values(self, window_name):
        """
        Get current trackbar values and convert to filter parameters.
        
        Args:
            window_name (str): Name of the OpenCV window
            
        Returns:
            dict: Current parameter values
        """
        kernel_size = cv2.getTrackbarPos('Gaussian Kernel', window_name)
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel_size = max(1, kernel_size)
        
        sigma_x = cv2.getTrackbarPos('Gaussian Sigma X', window_name) / 10.0
        sigma_y = cv2.getTrackbarPos('Gaussian Sigma Y', window_name) / 10.0
        
        return {
            'kernel_size': kernel_size,
            'sigma_x': max(0.1, sigma_x),
            'sigma_y': max(0.1, sigma_y)
        }
