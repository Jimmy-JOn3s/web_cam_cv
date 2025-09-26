"""
Trackbar Manager - Manages OpenCV trackbars for real-time parameter adjustment
"""
import cv2


class TrackbarManager:
    """Manages trackbars for real-time parameter adjustment."""
    
    def __init__(self, window_name):
        """
        Initialize trackbar manager.
        
        Args:
            window_name (str): Name of the OpenCV window
        """
        self.window_name = window_name
        self.trackbars = {}
        
    def create_trackbar(self, name, default_value, max_value, callback=None):
        """
        Create a trackbar.
        
        Args:
            name (str): Trackbar name
            default_value (int): Default value
            max_value (int): Maximum value
            callback (function): Callback function for trackbar changes
        """
        if callback is None:
            callback = lambda x: None
            
        cv2.createTrackbar(name, self.window_name, default_value, max_value, callback)
        self.trackbars[name] = {
            'default': default_value,
            'max': max_value,
            'callback': callback
        }
    

    
    def get_value(self, name):
        """
        Get current trackbar value.
        
        Args:
            name (str): Trackbar name
            
        Returns:
            int: Current trackbar value
        """
        return cv2.getTrackbarPos(name, self.window_name)
    
    
    
    def get_all_values(self):
        """
        Get all trackbar values.
        
        Returns:
            dict: Dictionary of trackbar names and their current values
        """
        values = {}
        for name in self.trackbars:
            values[name] = self.get_value(name)
        return values
    
    def reset_to_defaults(self):
        """Reset all trackbars to their default values."""
        for name, info in self.trackbars.items():
            cv2.setTrackbarPos(name, self.window_name, info['default'])
    
    def create_brightness_contrast_trackbars(self):
        """Create standard brightness and contrast trackbars."""
        self.create_trackbar('Brightness', 100, 200)  # 0-200, default 100 (maps to -100 to +100)
        self.create_trackbar('Contrast', 100, 200)    # 1-200, default 100 (maps to 0.01 to 2.0)
    
    def get_brightness_contrast_values(self):
        """
        Get brightness and contrast values in proper format.
        
        Returns:
            tuple: (brightness, contrast) where brightness is -100 to +100 and contrast is 0.01 to 2.0
        """
        brightness = self.get_value('Brightness') - 100  # Convert to -100 to +100
        contrast = self.get_value('Contrast') / 100.0     # Convert to 0.01 to 2.0
        return brightness, contrast
