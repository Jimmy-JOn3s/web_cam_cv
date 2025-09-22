"""
Display Manager - Handles window display and layout management
"""
import cv2
import numpy as np


class DisplayManager:
    """Manages display windows and layout."""
    
    def __init__(self):
        """Initialize display manager."""
        self.windows = {}
        self.layout_mode = 'single'  # 'single', 'side_by_side', 'grid'
        
    def create_window(self, name, flags=cv2.WINDOW_AUTOSIZE):
        """
        Create a named window.
        
        Args:
            name (str): Window name
            flags: OpenCV window flags
        """
        cv2.namedWindow(name, flags)
        self.windows[name] = {'flags': flags, 'active': True}
        
    def destroy_window(self, name):
        """
        Destroy a specific window.
        
        Args:
            name (str): Window name
        """
        if name in self.windows:
            cv2.destroyWindow(name)
            del self.windows[name]
    
    def destroy_all_windows(self):
        """Destroy all windows."""
        cv2.destroyAllWindows()
        self.windows.clear()
    
    def display_image(self, window_name, image, title_overlay=None):
        """
        Display an image in a window.
        
        Args:
            window_name (str): Window name
            image (numpy.ndarray): Image to display
            title_overlay (str): Optional title to overlay on image
        """
        display_image = image.copy()
        
        if title_overlay:
            self.add_text_overlay(display_image, title_overlay)
        
        cv2.imshow(window_name, display_image)
    
    def add_text_overlay(self, image, text, position=(10, 30), font=cv2.FONT_HERSHEY_SIMPLEX, 
                        font_scale=0.7, color=(255, 255, 255), thickness=2):
        """
        Add text overlay to an image.
        
        Args:
            image (numpy.ndarray): Image to add text to
            text (str): Text to add
            position (tuple): Text position (x, y)
            font: OpenCV font type
            font_scale (float): Font scale
            color (tuple): Text color (BGR)
            thickness (int): Text thickness
        """
        # Handle grayscale images
        if len(image.shape) == 2:
            color = (255,)  # White for grayscale
        
        cv2.putText(image, text, position, font, font_scale, color, thickness)
    
    def create_side_by_side_display(self, image1, image2, title1="Original", title2="Processed"):
        """
        Create side-by-side display of two images.
        
        Args:
            image1 (numpy.ndarray): First image
            image2 (numpy.ndarray): Second image
            title1 (str): Title for first image
            title2 (str): Title for second image
            
        Returns:
            numpy.ndarray: Combined image
        """
        # Ensure images have the same height
        h1, w1 = image1.shape[:2]
        h2, w2 = image2.shape[:2]
        
        max_height = max(h1, h2)
        
        # Resize images if needed
        if h1 != max_height:
            image1 = cv2.resize(image1, (int(w1 * max_height / h1), max_height))
        if h2 != max_height:
            image2 = cv2.resize(image2, (int(w2 * max_height / h2), max_height))
        
        # Handle different channel counts
        if len(image1.shape) == 2:
            image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
        if len(image2.shape) == 2:
            image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)
        
        # Add titles
        self.add_text_overlay(image1, title1)
        self.add_text_overlay(image2, title2)
        
        # Concatenate horizontally
        combined = np.hstack((image1, image2))
        
        return combined
    
    def create_grid_display(self, images, titles=None, grid_size=None):
        """
        Create a grid display of multiple images.
        
        Args:
            images (list): List of images to display
            titles (list): List of titles for each image
            grid_size (tuple): Grid size (rows, cols). If None, auto-calculate
            
        Returns:
            numpy.ndarray: Grid image
        """
        if not images:
            return np.zeros((100, 100, 3), dtype=np.uint8)
        
        num_images = len(images)
        
        if grid_size is None:
            # Auto-calculate grid size
            cols = int(np.ceil(np.sqrt(num_images)))
            rows = int(np.ceil(num_images / cols))
        else:
            rows, cols = grid_size
        
        # Get the size for all images (use first image as reference)
        ref_height, ref_width = images[0].shape[:2]
        
        # Prepare all images
        processed_images = []
        for i, img in enumerate(images):
            # Resize to reference size
            if img.shape[:2] != (ref_height, ref_width):
                img = cv2.resize(img, (ref_width, ref_height))
            
            # Convert to BGR if grayscale
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
            # Add title if provided
            if titles and i < len(titles):
                self.add_text_overlay(img, titles[i])
            
            processed_images.append(img)
        
        # Fill empty slots with black images
        while len(processed_images) < rows * cols:
            black_img = np.zeros((ref_height, ref_width, 3), dtype=np.uint8)
            processed_images.append(black_img)
        
        # Create grid
        grid_rows = []
        for row in range(rows):
            row_images = processed_images[row * cols:(row + 1) * cols]
            grid_row = np.hstack(row_images)
            grid_rows.append(grid_row)
        
        grid = np.vstack(grid_rows)
        return grid
    
    def add_control_instructions(self, image, instructions):
        """
        Add control instructions to the bottom of an image.
        
        Args:
            image (numpy.ndarray): Image to add instructions to
            instructions (list): List of instruction strings
        """
        if not instructions:
            return
        
        height, width = image.shape[:2]
        text_height = 25
        start_y = height - (len(instructions) * text_height) - 10
        
        for i, instruction in enumerate(instructions):
            y_pos = start_y + (i * text_height)
            self.add_text_overlay(image, instruction, (10, y_pos), font_scale=0.5)
    
    def wait_for_key(self, delay=1):
        """
        Wait for a key press.
        
        Args:
            delay (int): Delay in milliseconds
            
        Returns:
            int: Key code (masked to 8 bits)
        """
        return cv2.waitKey(delay) & 0xFF
