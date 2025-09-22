"""
Hough Transform for Line Detection - Custom implementation
"""
import cv2
import numpy as np
from ..filters.base_filter import BaseFilter


class HoughLineDetector(BaseFilter):
    """Custom Hough Transform line detection implementation."""
    
    def __init__(self):
        """Initialize Hough line detector with default parameters."""
        super().__init__("Hough Line Detection")
        self.parameters = {
            'rho_resolution': 1,
            'theta_resolution': 1,
            'threshold': 100,
            'min_line_length': 50,
            'max_line_gap': 10
        }
    
    def hough_transform(self, edges, rho_res=1, theta_res=1):
        """
        Perform Hough transform to detect lines.
        
        Args:
            edges (numpy.ndarray): Edge detected image
            rho_res (int): Rho resolution in pixels
            theta_res (int): Theta resolution in degrees
            
        Returns:
            tuple: (accumulator, rho_values, theta_values)
        """
        height, width = edges.shape
        
        # Maximum possible rho value
        max_rho = int(np.sqrt(height**2 + width**2))
        
        # Create parameter space
        rho_values = np.arange(-max_rho, max_rho + 1, rho_res)
        theta_values = np.deg2rad(np.arange(0, 180, theta_res))
        
        # Initialize accumulator
        accumulator = np.zeros((len(rho_values), len(theta_values)), dtype=np.int32)
        
        # Find edge pixels
        edge_pixels = np.where(edges > 0)
        
        # Vote in Hough space
        for i in range(len(edge_pixels[0])):
            y = edge_pixels[0][i]
            x = edge_pixels[1][i]
            
            for theta_idx, theta in enumerate(theta_values):
                rho = int(x * np.cos(theta) + y * np.sin(theta))
                rho_idx = np.argmin(np.abs(rho_values - rho))
                accumulator[rho_idx, theta_idx] += 1
        
        return accumulator, rho_values, theta_values
    
    def find_peaks(self, accumulator, threshold, min_distance=10):
        """
        Find peaks in the accumulator array.
        
        Args:
            accumulator (numpy.ndarray): Hough accumulator
            threshold (int): Minimum votes for a line
            min_distance (int): Minimum distance between peaks
            
        Returns:
            list: List of (rho_idx, theta_idx, votes) tuples
        """
        peaks = []
        
        # Find all points above threshold
        candidates = np.where(accumulator >= threshold)
        
        for i in range(len(candidates[0])):
            rho_idx = candidates[0][i]
            theta_idx = candidates[1][i]
            votes = accumulator[rho_idx, theta_idx]
            
            # Check if this is a local maximum
            is_peak = True
            for dr in range(-min_distance, min_distance + 1):
                for dt in range(-min_distance, min_distance + 1):
                    r_check = rho_idx + dr
                    t_check = theta_idx + dt
                    
                    if (0 <= r_check < accumulator.shape[0] and 
                        0 <= t_check < accumulator.shape[1]):
                        if accumulator[r_check, t_check] > votes:
                            is_peak = False
                            break
                if not is_peak:
                    break
            
            if is_peak:
                peaks.append((rho_idx, theta_idx, votes))
        
        # Sort by votes (descending)
        peaks.sort(key=lambda x: x[2], reverse=True)
        
        return peaks
    
    def draw_lines(self, image, lines, rho_values, theta_values):
        """
        Draw detected lines on the image.
        
        Args:
            image (numpy.ndarray): Input image
            lines (list): List of detected lines
            rho_values (numpy.ndarray): Rho parameter values
            theta_values (numpy.ndarray): Theta parameter values
            
        Returns:
            numpy.ndarray: Image with lines drawn
        """
        result = image.copy()
        if len(result.shape) == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        
        height, width = image.shape[:2]
        
        for rho_idx, theta_idx, votes in lines:
            rho = rho_values[rho_idx]
            theta = theta_values[theta_idx]
            
            # Convert from polar to Cartesian coordinates
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            
            # Calculate line endpoints
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            
            # Draw the line
            cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Add vote count text
            mid_x = width // 2
            mid_y = 30 + len([l for l in lines if lines.index((rho_idx, theta_idx, votes)) >= lines.index(l)]) * 20
            cv2.putText(result, f'Votes: {votes}', (mid_x, mid_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        return result
    
    def apply(self, image, **kwargs):
        """
        Apply Hough line detection.
        
        Args:
            image (numpy.ndarray): Input image (should be edge detected)
            **kwargs: Parameters
            
        Returns:
            numpy.ndarray: Image with detected lines
        """
        # Get parameters
        rho_res = kwargs.get('rho_resolution', self.parameters['rho_resolution'])
        theta_res = kwargs.get('theta_resolution', self.parameters['theta_resolution'])
        threshold = kwargs.get('threshold', self.parameters['threshold'])
        
        # Ensure we have an edge image
        if len(image.shape) == 3:
            edges = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            edges = image.copy()
        
        # Apply Hough transform
        accumulator, rho_values, theta_values = self.hough_transform(edges, rho_res, theta_res)
        
        # Find line candidates
        lines = self.find_peaks(accumulator, threshold)
        
        # Limit number of lines displayed
        max_lines = 10
        lines = lines[:max_lines]
        
        # Draw lines on original image
        result = self.draw_lines(image, lines, rho_values, theta_values)
        
        return result
    
    def get_parameter_info(self):
        """Get parameter information for Hough line detection."""
        return {
            'rho_resolution': {
                'type': int,
                'range': (1, 5),
                'default': 1,
                'description': 'Rho resolution in pixels',
                'step': 1
            },
            'theta_resolution': {
                'type': int,
                'range': (1, 5),
                'default': 1,
                'description': 'Theta resolution in degrees',
                'step': 1
            },
            'threshold': {
                'type': int,
                'range': (10, 300),
                'default': 100,
                'description': 'Minimum votes for line detection',
                'step': 1
            }
        }
    
    def create_trackbars(self, window_name):
        """Create trackbars for Hough line detection."""
        param_info = self.get_parameter_info()
        
        cv2.createTrackbar('Hough Rho Resolution', window_name, 
                          param_info['rho_resolution']['default'], 
                          param_info['rho_resolution']['range'][1], lambda x: None)
        
        cv2.createTrackbar('Hough Theta Resolution', window_name, 
                          param_info['theta_resolution']['default'], 
                          param_info['theta_resolution']['range'][1], lambda x: None)
        
        cv2.createTrackbar('Hough Threshold', window_name, 
                          param_info['threshold']['default'], 
                          param_info['threshold']['range'][1], lambda x: None)
    
    def get_trackbar_values(self, window_name):
        """Get current trackbar values."""
        rho_res = max(1, cv2.getTrackbarPos('Hough Rho Resolution', window_name))
        theta_res = max(1, cv2.getTrackbarPos('Hough Theta Resolution', window_name))
        threshold = max(10, cv2.getTrackbarPos('Hough Threshold', window_name))
        
        return {
            'rho_resolution': rho_res,
            'theta_resolution': theta_res,
            'threshold': threshold
        }
