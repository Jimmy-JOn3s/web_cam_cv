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
    
    def remove_trackbar(self, name):
        """
        Remove a trackbar (by recreating window without it).
        
        Args:
            name (str): Trackbar name to remove
        """
        if name in self.trackbars:
            # Store current values of remaining trackbars
            remaining_values = {}
            for trackbar_name in self.trackbars:
                if trackbar_name != name:
                    remaining_values[trackbar_name] = self.get_value(trackbar_name)
            
            # Remove from our tracking
            del self.trackbars[name]
            
            # Destroy and recreate window to remove the trackbar
            cv2.destroyWindow(self.window_name)
            cv2.namedWindow(self.window_name)
            
            # Recreate remaining trackbars
            for trackbar_name, info in self.trackbars.items():
                cv2.createTrackbar(trackbar_name, self.window_name, 
                                 remaining_values.get(trackbar_name, info['default']), 
                                 info['max'], info['callback'])
    
    def get_value(self, name):
        """
        Get current trackbar value.
        
        Args:
            name (str): Trackbar name
            
        Returns:
            int: Current trackbar value
        """
        return cv2.getTrackbarPos(name, self.window_name)
    
    def set_value(self, name, value):
        """
        Set trackbar value.
        
        Args:
            name (str): Trackbar name
            value (int): Value to set
        """
        if name in self.trackbars:
            max_val = self.trackbars[name]['max']
            value = max(0, min(value, max_val))
            cv2.setTrackbarPos(name, self.window_name, value)
    
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
            self.set_value(name, info['default'])
    
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
    
    def create_color_space_trackbar(self):
        """Create trackbar for color space selection."""
        # 0: BGR, 1: GRAY, 2: HSV
        self.create_trackbar('Color Space', 0, 2)
    
    def get_color_space_value(self):
        """
        Get selected color space.
        
        Returns:
            int: Color space index (0: BGR, 1: GRAY, 2: HSV)
        """
        return self.get_value('Color Space')
