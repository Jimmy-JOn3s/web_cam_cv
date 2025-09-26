"""
Webcam Manager - Handles webcam initialization and frame capture
"""
import cv2
import numpy as np


class WebcamManager:
    """Manages webcam operations including initialization, frame capture, and cleanup."""
    
    def __init__(self, camera_index=0, default_width=640, default_height=480):
        """
        Initialize webcam manager.
        
        Args:
            camera_index (int): Camera index (usually 0 for default camera)
            default_width (int): Default frame width for resizing
            default_height (int): Default frame height for resizing
        """
        self.camera_index = camera_index
        self.default_width = default_width
        self.default_height = default_height
        self.cap = None
        self.is_opened = False
        
    def initialize(self):
        """Initialize the webcam."""
        self.cap = cv2.VideoCapture(self.camera_index)
        self.is_opened = self.cap.isOpened()
        
        if not self.is_opened:
            raise IOError("Cannot open webcam")
        
        # Set webcam resolution to standard 480p
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.default_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.default_height)
        
        # Get actual resolution (webcam might not support exact resolution)
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Requested resolution: {self.default_width}x{self.default_height}")
        print(f"Actual webcam resolution: {actual_width}x{actual_height}")
        
        return self.is_opened
    
    def read_frame(self, resize=False):
        """
        Read a frame from the webcam.
        
        Args:
            resize (bool): Whether to resize the frame to default dimensions
            
        Returns:
            tuple: (success, frame) where success is bool and frame is numpy array
        """
        if not self.cap:
            return False, None
            
        ret, frame = self.cap.read()
        
        if ret and resize:
            frame = cv2.resize(frame, (self.default_width, self.default_height))
            
        return ret, frame
    

    
    def release(self):
        """Release the webcam resource."""
        if self.cap:
            self.cap.release()
            self.is_opened = False
