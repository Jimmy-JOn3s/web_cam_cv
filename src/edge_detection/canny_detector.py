"""
Canny Edge Detection - Custom implementation
"""
import cv2
import numpy as np
from ..filters.base_filter import BaseFilter


class CannyEdgeDetector(BaseFilter):
    """Custom Canny edge detection implementation."""
    
    def __init__(self):
        """Initialize Canny edge detector with default parameters."""
        super().__init__("Canny Edge Detection")
        self.parameters = {
            'gaussian_kernel': 5,
            'gaussian_sigma': 1.0,
            'low_threshold': 50,
            'high_threshold': 150
        }
    
    def gaussian_blur(self, image, kernel_size, sigma):
        """Apply Gaussian blur using OpenCV for preprocessing."""
        if kernel_size % 2 == 0:
            kernel_size += 1
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    def sobel_gradients(self, image):
        """
        Calculate gradients using Sobel operators.
        
        Args:
            image (numpy.ndarray): Input grayscale image
            
        Returns:
            tuple: (magnitude, direction, grad_x, grad_y)
        """
        # Sobel kernels
        sobel_x = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=np.float32)
        
        sobel_y = np.array([[-1, -2, -1],
                           [ 0,  0,  0],
                           [ 1,  2,  1]], dtype=np.float32)
        
        # Apply convolution
        grad_x = cv2.filter2D(image.astype(np.float32), -1, sobel_x)
        grad_y = cv2.filter2D(image.astype(np.float32), -1, sobel_y)
        
        # Calculate magnitude and direction
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        direction = np.arctan2(grad_y, grad_x)
        
        return magnitude, direction, grad_x, grad_y
    
    def non_maximum_suppression(self, magnitude, direction):
        """
        Apply non-maximum suppression to thin edges.
        
        Args:
            magnitude (numpy.ndarray): Gradient magnitude
            direction (numpy.ndarray): Gradient direction
            
        Returns:
            numpy.ndarray: Suppressed magnitude
        """
        height, width = magnitude.shape
        suppressed = np.zeros_like(magnitude)
        
        # Convert angles to degrees and normalize to 0-180
        angle = np.rad2deg(direction) % 180
        
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                try:
                    q = 255
                    r = 255
                    
                    # Angle 0
                    if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                        q = magnitude[i, j + 1]
                        r = magnitude[i, j - 1]
                    # Angle 45
                    elif 22.5 <= angle[i, j] < 67.5:
                        q = magnitude[i + 1, j - 1]
                        r = magnitude[i - 1, j + 1]
                    # Angle 90
                    elif 67.5 <= angle[i, j] < 112.5:
                        q = magnitude[i + 1, j]
                        r = magnitude[i - 1, j]
                    # Angle 135
                    elif 112.5 <= angle[i, j] < 157.5:
                        q = magnitude[i - 1, j - 1]
                        r = magnitude[i + 1, j + 1]
                    
                    if magnitude[i, j] >= q and magnitude[i, j] >= r:
                        suppressed[i, j] = magnitude[i, j]
                    else:
                        suppressed[i, j] = 0
                        
                except IndexError:
                    pass
        
        return suppressed
    
    def double_threshold(self, image, low_threshold, high_threshold):
        """
        Apply double threshold to classify edges.
        
        Args:
            image (numpy.ndarray): Suppressed magnitude image
            low_threshold (float): Low threshold value
            high_threshold (float): High threshold value
            
        Returns:
            numpy.ndarray: Thresholded image
        """
        high_threshold_ratio = high_threshold
        low_threshold_ratio = low_threshold
        
        strong = 255
        weak = 75
        
        result = np.zeros_like(image)
        
        strong_i, strong_j = np.where(image >= high_threshold_ratio)
        weak_i, weak_j = np.where((image <= high_threshold_ratio) & (image >= low_threshold_ratio))
        
        result[strong_i, strong_j] = strong
        result[weak_i, weak_j] = weak
        
        return result
    
    def edge_tracking(self, image):
        """
        Track edges by hysteresis.
        
        Args:
            image (numpy.ndarray): Thresholded image
            
        Returns:
            numpy.ndarray: Final edge image
        """
        height, width = image.shape
        strong = 255
        weak = 75
        
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                if image[i, j] == weak:
                    try:
                        if ((image[i + 1, j - 1] == strong) or (image[i + 1, j] == strong) or 
                            (image[i + 1, j + 1] == strong) or (image[i, j - 1] == strong) or 
                            (image[i, j + 1] == strong) or (image[i - 1, j - 1] == strong) or 
                            (image[i - 1, j] == strong) or (image[i - 1, j + 1] == strong)):
                            image[i, j] = strong
                        else:
                            image[i, j] = 0
                    except IndexError:
                        pass
        
        return image
    
    def apply(self, image, **kwargs):
        """
        Apply Canny edge detection.
        
        Args:
            image (numpy.ndarray): Input image
            **kwargs: Parameters (gaussian_kernel, gaussian_sigma, low_threshold, high_threshold)
            
        Returns:
            numpy.ndarray: Edge detected image
        """
        # Get parameters
        gaussian_kernel = kwargs.get('gaussian_kernel', self.parameters['gaussian_kernel'])
        gaussian_sigma = kwargs.get('gaussian_sigma', self.parameters['gaussian_sigma'])
        low_threshold = kwargs.get('low_threshold', self.parameters['low_threshold'])
        high_threshold = kwargs.get('high_threshold', self.parameters['high_threshold'])
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Step 1: Gaussian blur
        blurred = self.gaussian_blur(gray, gaussian_kernel, gaussian_sigma)
        
        # Step 2: Calculate gradients
        magnitude, direction, _, _ = self.sobel_gradients(blurred)
        
        # Step 3: Non-maximum suppression
        suppressed = self.non_maximum_suppression(magnitude, direction)
        
        # Step 4: Double threshold
        thresholded = self.double_threshold(suppressed, low_threshold, high_threshold)
        
        # Step 5: Edge tracking by hysteresis
        edges = self.edge_tracking(thresholded.copy())
        
        return edges.astype(np.uint8)
    
    def get_parameter_info(self):
        """Get parameter information for Canny edge detection."""
        return {
            'gaussian_kernel': {
                'type': int,
                'range': (3, 15),
                'default': 5,
                'description': 'Gaussian kernel size for noise reduction',
                'step': 2
            },
            'gaussian_sigma': {
                'type': float,
                'range': (0.1, 5.0),
                'default': 1.0,
                'description': 'Gaussian sigma for blur amount',
                'step': 0.1
            },
            'low_threshold': {
                'type': int,
                'range': (0, 255),
                'default': 50,
                'description': 'Low threshold for edge detection',
                'step': 1
            },
            'high_threshold': {
                'type': int,
                'range': (0, 255),
                'default': 150,
                'description': 'High threshold for edge detection',
                'step': 1
            }
        }
    
    def create_trackbars(self, window_name):
        """Create trackbars for Canny edge detection."""
        param_info = self.get_parameter_info()
        
        # Gaussian kernel trackbar
        cv2.createTrackbar('Canny Gaussian Kernel', window_name, 
                          param_info['gaussian_kernel']['default'], 
                          param_info['gaussian_kernel']['range'][1], lambda x: None)
        
        # Gaussian sigma trackbar (scaled by 10)
        cv2.createTrackbar('Canny Gaussian Sigma', window_name, 
                          int(param_info['gaussian_sigma']['default'] * 10), 
                          int(param_info['gaussian_sigma']['range'][1] * 10), lambda x: None)
        
        # Low threshold trackbar
        cv2.createTrackbar('Canny Low Threshold', window_name, 
                          param_info['low_threshold']['default'], 
                          param_info['low_threshold']['range'][1], lambda x: None)
        
        # High threshold trackbar
        cv2.createTrackbar('Canny High Threshold', window_name, 
                          param_info['high_threshold']['default'], 
                          param_info['high_threshold']['range'][1], lambda x: None)
    
    def get_trackbar_values(self, window_name):
        """Get current trackbar values."""
        gaussian_kernel = cv2.getTrackbarPos('Canny Gaussian Kernel', window_name)
        if gaussian_kernel % 2 == 0:
            gaussian_kernel += 1
        gaussian_kernel = max(3, gaussian_kernel)
        
        gaussian_sigma = cv2.getTrackbarPos('Canny Gaussian Sigma', window_name) / 10.0
        low_threshold = cv2.getTrackbarPos('Canny Low Threshold', window_name)
        high_threshold = cv2.getTrackbarPos('Canny High Threshold', window_name)
        
        return {
            'gaussian_kernel': gaussian_kernel,
            'gaussian_sigma': max(0.1, gaussian_sigma),
            'low_threshold': low_threshold,
            'high_threshold': max(high_threshold, low_threshold + 1)  # Ensure high > low
        }
