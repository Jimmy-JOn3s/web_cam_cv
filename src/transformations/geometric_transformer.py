"""
Geometric Transformations - Translation, Rotation, and Scaling
"""
import cv2
import numpy as np


class GeometricTransformer:
    """Handles various geometric transformations."""
    
    def __init__(self):
        """Initialize transformer."""
        self.parameters = {
            'translate_x': 0,
            'translate_y': 0,
            'rotation_angle': 0,
            'scale_x': 1.0,
            'scale_y': 1.0
        }
    
    
    
    def rotate_image(self, image, angle, center=None):
        """
        Rotate image by specified angle.
        
        Args:
            image (numpy.ndarray): Input image
            angle (float): Rotation angle in degrees
            center (tuple): Rotation center (x, y). If None, uses image center
            
        Returns:
            numpy.ndarray: Rotated image
        """
        height, width = image.shape[:2]
        
        if center is None:
            center = (width // 2, height // 2)
        
        # Create rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Apply rotation
        rotated = cv2.warpAffine(image, rotation_matrix, (width, height))
        
        return rotated
    
    def scale_image(self, image, scale_x, scale_y, interpolation=cv2.INTER_LINEAR):
        """
        Scale image by specified factors.
        
        Args:
            image (numpy.ndarray): Input image
            scale_x (float): Scaling factor in x direction
            scale_y (float): Scaling factor in y direction
            interpolation: Interpolation method
            
        Returns:
            numpy.ndarray: Scaled image
        """
        height, width = image.shape[:2]
        
        # Calculate new dimensions
        new_width = int(width * scale_x)
        new_height = int(height * scale_y)
        
        # Apply scaling
        scaled = cv2.resize(image, (new_width, new_height), interpolation=interpolation)
        
        # If scaled image is smaller, pad it to original size
        if new_width < width or new_height < height:
            result = np.zeros_like(image)
            y_offset = (height - new_height) // 2
            x_offset = (width - new_width) // 2
            result[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = scaled
            return result
        else:
            # If larger, crop to original size
            y_offset = (new_height - height) // 2
            x_offset = (new_width - width) // 2
            return scaled[y_offset:y_offset+height, x_offset:x_offset+width]
    
    def combined_transform(self, image, tx=0, ty=0, angle=0, scale_x=1.0, scale_y=1.0):
        """
        Apply combined geometric transformations.
        
        Args:
            image (numpy.ndarray): Input image
            tx (float): Translation in x
            ty (float): Translation in y
            angle (float): Rotation angle in degrees
            scale_x (float): Scale factor in x
            scale_y (float): Scale factor in y
            
        Returns:
            numpy.ndarray: Transformed image
        """
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        # Create combined transformation matrix
        # 1. Scale
        scale_matrix = np.array([[scale_x, 0, 0],
                               [0, scale_y, 0],
                               [0, 0, 1]], dtype=np.float32)
        
        # 2. Rotation
        cos_a = np.cos(np.radians(angle))
        sin_a = np.sin(np.radians(angle))
        rotation_matrix = np.array([[cos_a, -sin_a, 0],
                                  [sin_a, cos_a, 0],
                                  [0, 0, 1]], dtype=np.float32)
        
        # 3. Translation
        translation_matrix = np.array([[1, 0, tx],
                                     [0, 1, ty],
                                     [0, 0, 1]], dtype=np.float32)
        
        # Combine transformations: T * R * S
        combined_matrix = translation_matrix @ rotation_matrix @ scale_matrix
        
        # Convert to 2x3 matrix for cv2.warpAffine
        affine_matrix = combined_matrix[:2, :]
        
        # Apply transformation
        transformed = cv2.warpAffine(image, affine_matrix, (width, height))
        
        return transformed
    
    def create_trackbars(self, window_name):
        """Create trackbars for geometric transformations."""
        # Translation trackbars (range: -200 to 200, default: 100 = 0)
        cv2.createTrackbar('Translate X', window_name, 100, 200, lambda x: None)
        cv2.createTrackbar('Translate Y', window_name, 100, 200, lambda x: None)
        
        # Rotation trackbar (range: 0 to 360 degrees)
        cv2.createTrackbar('Rotation', window_name, 0, 360, lambda x: None)
        
        # Scale trackbars (range: 0.1 to 3.0, scaled by 10, default: 10 = 1.0)
        cv2.createTrackbar('Scale X', window_name, 10, 30, lambda x: None)
        cv2.createTrackbar('Scale Y', window_name, 10, 30, lambda x: None)
    
    def get_trackbar_values(self, window_name):
        """Get current transformation parameters from trackbars."""
        tx = cv2.getTrackbarPos('Translate X', window_name) - 100  # Convert to -100 to 100
        ty = cv2.getTrackbarPos('Translate Y', window_name) - 100  # Convert to -100 to 100
        angle = cv2.getTrackbarPos('Rotation', window_name)
        scale_x = cv2.getTrackbarPos('Scale X', window_name) / 10.0  # Convert to 0.0 to 3.0
        scale_y = cv2.getTrackbarPos('Scale Y', window_name) / 10.0  # Convert to 0.0 to 3.0
        
        return {
            'tx': tx,
            'ty': ty,
            'angle': angle,
            'scale_x': max(0.1, scale_x),
            'scale_y': max(0.1, scale_y)
        }
    
    def apply_transforms(self, image, **kwargs):
        """
        Apply transformations based on parameters.
        
        Args:
            image (numpy.ndarray): Input image
            **kwargs: Transformation parameters
            
        Returns:
            numpy.ndarray: Transformed image
        """
        tx = kwargs.get('tx', 0)
        ty = kwargs.get('ty', 0)
        angle = kwargs.get('angle', 0)
        scale_x = kwargs.get('scale_x', 1.0)
        scale_y = kwargs.get('scale_y', 1.0)
        
        return self.combined_transform(image, tx, ty, angle, scale_x, scale_y)
    

