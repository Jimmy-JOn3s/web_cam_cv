"""
Hough Transform for Line Detection - Adjustable implementation with trackbars
"""
import cv2
import numpy as np
from ..filters.base_filter import BaseFilter


class HoughLineDetector(BaseFilter):
    """Hough Transform line detection with adjustable parameters."""
    
    def __init__(self):
        """Initialize Hough line detector with adjustable parameters."""
        super().__init__("Hough Line Detection")
        # Default parameters - can be adjusted with trackbars
        self.parameters = {
            'threshold': 100,       # Minimum votes for line detection
            'canny_low': 50,        # Canny lower threshold
            'canny_high': 150,      # Canny upper threshold
            'max_lines': 20         # Maximum number of lines to show
        }
        self.rho = 1                # Distance resolution in pixels  
        self.theta = np.pi/180      # Angle resolution in radians
    
    def detect_lines_adjustable(self, edges, threshold, max_lines):
        """
        Adjustable line detection using OpenCV's HoughLines.
        
        Args:
            edges (numpy.ndarray): Edge detected image
            threshold (int): Minimum votes for line detection
            max_lines (int): Maximum number of lines to return
            
        Returns:
            numpy.ndarray: Lines in format [[rho, theta], ...]
        """
        # Use standard Hough transform
        lines = cv2.HoughLines(edges, self.rho, self.theta, threshold=threshold)
        
        if lines is not None:
            # Limit number of lines for performance
            return lines[:max_lines]
        else:
            return np.array([])
    
    def draw_lines_adjustable(self, image, lines):
        """
        Draw detected lines on the image with adjustable visualization.
        
        Args:
            image (numpy.ndarray): Input image
            lines (numpy.ndarray): Array of lines [[rho, theta], ...]
            
        Returns:
            numpy.ndarray: Image with lines drawn
        """
        result = image.copy()
        if len(result.shape) == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        
        if lines is not None and len(lines) > 0:
            for rho, theta in lines[:, 0]:
                # Convert polar coords (rho, theta) to Cartesian direction
                a = np.cos(theta)  # x-component of the unit direction vector
                b = np.sin(theta)  # y-component of the unit direction vector
                
                # (x0, y0) is the point on the line closest to the origin (0,0)
                x0 = a * rho
                y0 = b * rho
                
                # Extend the line far in both directions for drawing:
                # First endpoint (go in the negative perpendicular direction)
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                
                # Second endpoint (go in the positive perpendicular direction)
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                
                # Draw the line on the image in red (BGR: (0, 0, 255)) with thickness=2
                cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Add line count info
            cv2.putText(result, f'Lines: {len(lines)}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return result
    
    def apply(self, image, **kwargs):
        """
        Apply adjustable Hough line detection with trackbar parameters.
        Works directly on original image - applies Canny internally.
        
        Args:
            image (numpy.ndarray): Input image (original BGR or grayscale)
            **kwargs: Parameters from trackbars
            
        Returns:
            numpy.ndarray: Original image with detected lines overlaid
        """
        # Get parameters from trackbars or use defaults
        threshold = kwargs.get('threshold', self.parameters['threshold'])
        canny_low = kwargs.get('canny_low', self.parameters['canny_low'])
        canny_high = kwargs.get('canny_high', self.parameters['canny_high'])
        max_lines = kwargs.get('max_lines', self.parameters['max_lines'])
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply Canny edge detection with adjustable thresholds
        edges = cv2.Canny(gray, canny_low, canny_high, apertureSize=3, L2gradient=True)
        
        # Apply Hough line detection with adjustable parameters
        lines = self.detect_lines_adjustable(edges, threshold, max_lines)
        
        # Draw lines on original image
        result = self.draw_lines_adjustable(image, lines)
        
        return result
    
    def get_parameter_info(self):
        """Get parameter information for adjustable Hough line detection."""
        return {
            'threshold': {
                'type': int,
                'range': (50, 500),
                'default': 100,
                'description': 'Minimum votes for line detection'
            },
            'canny_low': {
                'type': int,
                'range': (10, 200),
                'default': 50,
                'description': 'Canny lower threshold'
            },
            'canny_high': {
                'type': int,
                'range': (50, 300),
                'default': 150,
                'description': 'Canny upper threshold'
            },
            'max_lines': {
                'type': int,
                'range': (5, 50),
                'default': 20,
                'description': 'Maximum lines to display'
            }
        }
    
    def create_trackbars(self, window_name):
        """Create trackbars for adjustable Hough line detection."""
        cv2.createTrackbar('Hough Threshold', window_name, 100, 500, lambda x: None)
        cv2.createTrackbar('Canny Low', window_name, 50, 200, lambda x: None)
        cv2.createTrackbar('Canny High', window_name, 150, 300, lambda x: None)
        cv2.createTrackbar('Max Lines', window_name, 20, 50, lambda x: None)
    
    def get_trackbar_values(self, window_name):
        """Get current trackbar values."""
        threshold = max(50, cv2.getTrackbarPos('Hough Threshold', window_name))
        canny_low = max(10, cv2.getTrackbarPos('Canny Low', window_name))
        canny_high = max(50, cv2.getTrackbarPos('Canny High', window_name))
        max_lines = max(5, cv2.getTrackbarPos('Max Lines', window_name))
        
        # Ensure canny_high > canny_low
        if canny_high <= canny_low:
            canny_high = canny_low + 10
        
        return {
            'threshold': threshold,
            'canny_low': canny_low,
            'canny_high': canny_high,
            'max_lines': max_lines
        }
