"""
Image Processing Application - Main application class
"""
import cv2
import numpy as np
import sys
import os

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.core.webcam_manager import WebcamManager
from src.core.image_processor import ImageProcessor, ColorSpace
from src.color_spaces.histogram_analyzer import HistogramAnalyzer
from src.filters.gaussian_filter import GaussianFilter
from src.filters.bilateral_filter import BilateralFilter
from src.edge_detection.canny_detector import CannyEdgeDetector
from src.edge_detection.hough_lines import HoughLineDetector
from src.panorama.panorama_creator import PanoramaCreator
from src.transformations.geometric_transformer import GeometricTransformer
from src.calibration.camera_calibrator import CameraCalibrator
from src.ui.trackbar_manager import TrackbarManager
from src.ui.display_manager import DisplayManager


class ImageProcessingApp:
    """Main application class for real-time image processing."""
    
    def __init__(self):
        """Initialize the application."""
        self.webcam = WebcamManager()
        self.processor = ImageProcessor()
        self.histogram_analyzer = HistogramAnalyzer()
        
        # Filters
        self.gaussian_filter = GaussianFilter()
        self.bilateral_filter = BilateralFilter()
        
        # Advanced features
        self.canny_detector = CannyEdgeDetector()
        self.hough_detector = HoughLineDetector()
        self.panorama_creator = PanoramaCreator()
        self.geometric_transformer = GeometricTransformer()
        self.camera_calibrator = CameraCalibrator()
        
        # UI managers
        self.display_manager = DisplayManager()
        self.main_window = 'Image Processing App'
        self.histogram_window = 'Histogram'
        self.controls_window = 'Brightness & Contrast'
        self.gaussian_window = 'Gaussian Filter'
        self.bilateral_window = 'Bilateral Filter'
        self.canny_window = 'Canny Edge Detection'
        self.hough_window = 'Hough Line Detection'
        self.transform_window = 'Geometric Transforms'
        self.calibration_window = 'Camera Calibration'
        
        # State variables
        self.current_color_space = ColorSpace.BGR
        self.show_histogram = False
        self.gaussian_enabled = False
        self.bilateral_enabled = False
        self.canny_enabled = False
        self.hough_enabled = False
        self.transform_enabled = False
        self.calibration_mode = False
        self.running = False
        
    def initialize(self):
        """Initialize the application."""
        try:
            # Initialize webcam
            self.webcam.initialize()
            
            # Create main window and trackbars
            self.display_manager.create_window(self.main_window)
            
            # Create separate controls window for brightness and contrast
            self.display_manager.create_window(self.controls_window)
            self.trackbar_manager = TrackbarManager(self.controls_window)
            
            # Create control trackbars
            self.setup_trackbars()
            
            # Arrange windows
            self.arrange_windows()
            
            print("Application initialized successfully!")
            print("Controls:")
            print("  1: BGR color space")
            print("  2: Grayscale color space") 
            print("  3: HSV color space")
            print("  h: Toggle histogram display")
            print("  g: Toggle Gaussian filter")
            print("  b: Toggle Bilateral filter")
            print("  e: Toggle Canny edge detection")
            print("  l: Toggle Hough line detection")
            print("  t: Toggle geometric transformations")
            print("  c: Toggle camera calibration mode")
            print("  p: Add current frame to panorama")
            print("  m: Create panorama from collected images")
            print("  SPACE: Capture calibration image (in calibration mode)")
            print("  r: Reset all parameters")
            print("  s: Save current frame")
            print("  q: Quit")
            print()
            print("Note: Each feature opens in its own separate window!")
            
            return True
            
        except Exception as e:
            print(f"Error initializing application: {e}")
            return False
    
    def setup_trackbars(self):
        """Setup only basic trackbars for the application."""
        # Only create brightness and contrast trackbars by default
        self.trackbar_manager.create_brightness_contrast_trackbars()
        
    def create_gaussian_trackbars(self):
        """Create trackbars for Gaussian filter in separate window."""
        self.display_manager.create_window(self.gaussian_window)
        self.gaussian_filter.create_trackbars(self.gaussian_window)
        
    def create_bilateral_trackbars(self):
        """Create trackbars for Bilateral filter in separate window."""
        self.display_manager.create_window(self.bilateral_window)
        self.bilateral_filter.create_trackbars(self.bilateral_window)
        
    def remove_gaussian_trackbars(self):
        """Remove Gaussian filter trackbars by destroying window."""
        if self.gaussian_window in self.display_manager.windows:
            self.display_manager.destroy_window(self.gaussian_window)
                
    def remove_bilateral_trackbars(self):
        """Remove Bilateral filter trackbars by destroying window."""
        if self.bilateral_window in self.display_manager.windows:
            self.display_manager.destroy_window(self.bilateral_window)
    
    def create_canny_trackbars(self):
        """Create trackbars for Canny edge detection in separate window."""
        self.display_manager.create_window(self.canny_window)
        self.canny_detector.create_trackbars(self.canny_window)
        
    def remove_canny_trackbars(self):
        """Remove Canny edge detection trackbars by destroying window."""
        if self.canny_window in self.display_manager.windows:
            self.display_manager.destroy_window(self.canny_window)
    
    def create_hough_trackbars(self):
        """Create trackbars for Hough line detection in separate window."""
        self.display_manager.create_window(self.hough_window)
        self.hough_detector.create_trackbars(self.hough_window)
        
    def remove_hough_trackbars(self):
        """Remove Hough line detection trackbars by destroying window."""
        if self.hough_window in self.display_manager.windows:
            self.display_manager.destroy_window(self.hough_window)
    
    def create_transform_trackbars(self):
        """Create trackbars for geometric transformations in separate window."""
        self.display_manager.create_window(self.transform_window)
        self.geometric_transformer.create_trackbars(self.transform_window)
        
    def remove_transform_trackbars(self):
        """Remove geometric transformation trackbars by destroying window."""
        if self.transform_window in self.display_manager.windows:
            self.display_manager.destroy_window(self.transform_window)
    
    def create_calibration_window(self):
        """Create window for camera calibration."""
        self.display_manager.create_window(self.calibration_window)
        
    def remove_calibration_window(self):
        """Remove camera calibration window."""
        if self.calibration_window in self.display_manager.windows:
            self.display_manager.destroy_window(self.calibration_window)
    
    def arrange_windows(self):
        """Arrange windows in a nice layout."""
        try:
            # Main video window - center-left
            cv2.moveWindow(self.main_window, 50, 50)
            
            # Controls window - top-right
            cv2.moveWindow(self.controls_window, 700, 50)
            
        except Exception as e:
            # Window positioning might not work on all systems
            print(f"Note: Could not arrange windows automatically: {e}")
    
    def arrange_filter_windows(self):
        """Arrange filter windows when they are created."""
        try:
            y_offset = 200
            
            if self.gaussian_enabled and self.gaussian_window in self.display_manager.windows:
                cv2.moveWindow(self.gaussian_window, 700, y_offset)
                y_offset += 150
            
            if self.bilateral_enabled and self.bilateral_window in self.display_manager.windows:
                cv2.moveWindow(self.bilateral_window, 700, y_offset)
                y_offset += 150
            
            if self.canny_enabled and self.canny_window in self.display_manager.windows:
                cv2.moveWindow(self.canny_window, 900, 200)
            
            if self.hough_enabled and self.hough_window in self.display_manager.windows:
                cv2.moveWindow(self.hough_window, 900, 350)
            
            if self.transform_enabled and self.transform_window in self.display_manager.windows:
                cv2.moveWindow(self.transform_window, 1100, 200)
            
            if self.calibration_mode and self.calibration_window in self.display_manager.windows:
                cv2.moveWindow(self.calibration_window, 1100, 350)
                
            if self.show_histogram and self.histogram_window in self.display_manager.windows:
                cv2.moveWindow(self.histogram_window, 700, y_offset)
                
        except Exception as e:
            # Window positioning might not work on all systems
            pass
    
    def process_frame(self, frame):
        """
        Process a single frame with current settings.
        
        Args:
            frame (numpy.ndarray): Input frame
            
        Returns:
            numpy.ndarray: Processed frame
        """
        processed = frame.copy()
        
        # Apply brightness and contrast
        brightness, contrast = self.trackbar_manager.get_brightness_contrast_values()
        processed = self.processor.adjust_brightness_contrast(processed, brightness, contrast)
        
        # Apply geometric transformations first (if enabled)
        if self.transform_enabled:
            transform_params = self.geometric_transformer.get_trackbar_values(self.transform_window)
            processed = self.geometric_transformer.apply_transforms(processed, **transform_params)
        
        # Apply filters if enabled
        if self.gaussian_enabled:
            gaussian_params = self.gaussian_filter.get_trackbar_values(self.gaussian_window)
            processed = self.gaussian_filter.apply(processed, **gaussian_params)
        
        if self.bilateral_enabled:
            bilateral_params = self.bilateral_filter.get_trackbar_values(self.bilateral_window)
            processed = self.bilateral_filter.apply(processed, **bilateral_params)
        
        # Apply edge detection (Canny) if enabled
        if self.canny_enabled:
            canny_params = self.canny_detector.get_trackbar_values(self.canny_window)
            edge_image = self.canny_detector.apply(processed, **canny_params)
            
            # Apply Hough line detection on edge image if enabled
            if self.hough_enabled:
                hough_params = self.hough_detector.get_trackbar_values(self.hough_window)
                processed = self.hough_detector.apply(edge_image, **hough_params)
            else:
                processed = edge_image
        elif self.hough_enabled:
            # If Hough is enabled but Canny is not, first apply Canny with default params
            edge_image = self.canny_detector.apply(processed)
            hough_params = self.hough_detector.get_trackbar_values(self.hough_window)
            processed = self.hough_detector.apply(edge_image, **hough_params)
        
        # Apply color space conversion
        if self.current_color_space == ColorSpace.GRAY:
            processed = self.processor.convert_color_space(processed, ColorSpace.GRAY)
        elif self.current_color_space == ColorSpace.HSV:
            processed = self.processor.convert_color_space(processed, ColorSpace.HSV)
        
        return processed
    
    def update_histogram(self, image):
        """
        Update histogram display.
        
        Args:
            image (numpy.ndarray): Image to create histogram for
        """
        if self.show_histogram:
            if self.histogram_window not in self.display_manager.windows:
                self.display_manager.create_window(self.histogram_window)
            
            hist_image = self.histogram_analyzer.create_histogram_image(image)
            self.display_manager.display_image(self.histogram_window, hist_image, "Histogram")
        else:
            if self.histogram_window in self.display_manager.windows:
                self.display_manager.destroy_window(self.histogram_window)
    
    def handle_keyboard_input(self, key):
        """
        Handle keyboard input.
        
        Args:
            key (int): Key code
            
        Returns:
            bool: True to continue running, False to quit
        """
        if key == ord('q'):
            return False
        elif key == ord('1'):
            # Reset all color space modes
            self.current_color_space = ColorSpace.BGR
        elif key == ord('2'):
            # Toggle grayscale mode
            self.current_color_space = ColorSpace.GRAY
        elif key == ord('3'):
            # Toggle HSV mode
            self.current_color_space = ColorSpace.HSV
        elif key == ord('h'):
            # Toggle histogram display
            self.show_histogram = not self.show_histogram
            self.arrange_filter_windows()
            print(f"Histogram display: {'ON' if self.show_histogram else 'OFF'}")
        elif key == ord('g'):
            # Toggle Gaussian filter
            self.gaussian_enabled = not self.gaussian_enabled
            if self.gaussian_enabled:
                self.create_gaussian_trackbars()
                self.arrange_filter_windows()
                print("Gaussian filter: ON - Separate window created")
            else:
                self.remove_gaussian_trackbars()
                print("Gaussian filter: OFF - Window closed")
        elif key == ord('b'):
            # Toggle Bilateral filter
            self.bilateral_enabled = not self.bilateral_enabled
            if self.bilateral_enabled:
                self.create_bilateral_trackbars()
                self.arrange_filter_windows()
                print("Bilateral filter: ON - Separate window created")
            else:
                self.remove_bilateral_trackbars()
                print("Bilateral filter: OFF - Window closed")
        elif key == ord('e'):
            # Toggle Canny edge detection
            self.canny_enabled = not self.canny_enabled
            if self.canny_enabled:
                self.create_canny_trackbars()
                self.arrange_filter_windows()
                print("Canny edge detection: ON - Separate window created")
            else:
                self.remove_canny_trackbars()
                print("Canny edge detection: OFF - Window closed")
        elif key == ord('l'):
            # Toggle Hough line detection
            self.hough_enabled = not self.hough_enabled
            if self.hough_enabled:
                self.create_hough_trackbars()
                self.arrange_filter_windows()
                print("Hough line detection: ON - Separate window created")
            else:
                self.remove_hough_trackbars()
                print("Hough line detection: OFF - Window closed")
        elif key == ord('t'):
            # Toggle geometric transformations
            self.transform_enabled = not self.transform_enabled
            if self.transform_enabled:
                self.create_transform_trackbars()
                self.arrange_filter_windows()
                print("Geometric transformations: ON - Separate window created")
            else:
                self.remove_transform_trackbars()
                print("Geometric transformations: OFF - Window closed")
        elif key == ord('c'):
            # Toggle camera calibration mode
            self.calibration_mode = not self.calibration_mode
            if self.calibration_mode:
                self.create_calibration_window()
                self.arrange_filter_windows()
                print("Camera calibration mode: ON")
                print("Press SPACE to capture calibration images")
            else:
                self.remove_calibration_window()
                print("Camera calibration mode: OFF")
        elif key == ord('p'):
            # Add current frame to panorama
            ret, frame = self.webcam.read_frame()
            if ret:
                success = self.panorama_creator.add_image(frame)
                if success:
                    status = self.panorama_creator.get_status()
                    print(f"Frame added to panorama collection. Total images: {status['num_images']}")
                else:
                    print("Failed to add frame to panorama collection")
        elif key == ord('m'):
            # Create panorama
            panorama = self.panorama_creator.create_panorama()
            if panorama is not None:
                self.panorama_creator.save_panorama(panorama)
                print(f"Panorama created! Size: {panorama.shape[1]}x{panorama.shape[0]} pixels")
                
                # Display panorama in a new window with proper sizing
                window_name = 'Panorama Result'
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                
                # Resize panorama for display if it's too large
                h, w = panorama.shape[:2]
                max_width = 1200
                max_height = 800
                
                if w > max_width or h > max_height:
                    # Calculate scaling factor to fit within max dimensions
                    scale_w = max_width / w
                    scale_h = max_height / h
                    scale = min(scale_w, scale_h)
                    
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    
                    display_panorama = cv2.resize(panorama, (new_w, new_h))
                    print(f"Panorama resized for display: {new_w}x{new_h} pixels")
                else:
                    display_panorama = panorama
                
                cv2.imshow(window_name, display_panorama)
                cv2.moveWindow(window_name, 100, 100)
                print("Panorama displayed in separate window. Press any key in panorama window to close it.")
            else:
                print("Could not create panorama. Make sure you've added at least 2 images with 'p' key.")
        elif key == 32:  # SPACE key
            # Capture calibration image (only in calibration mode)
            if self.calibration_mode:
                ret, frame = self.webcam.read_frame()
                if ret:
                    self.camera_calibrator.add_calibration_image(frame)
        elif key == ord('r'):
            # Reset all parameters and disable all filters
            self.reset_all_settings()
        elif key == ord('s'):
            self.save_current_frame()
        
        return True
    
    def save_current_frame(self):
        """Save the current processed frame."""
        ret, frame = self.webcam.read_frame()
        if ret:
            processed = self.process_frame(frame)
            filename = f"processed_frame_{cv2.getTickCount()}.jpg"
            cv2.imwrite(filename, processed)
            print(f"Frame saved as {filename}")
    
    def reset_all_settings(self):
        """Reset all settings to default values."""
        # Reset brightness and contrast
        self.trackbar_manager.reset_to_defaults()
        
        # Reset color space
        self.current_color_space = ColorSpace.BGR
        
        # Disable all filters and remove their trackbars
        if self.gaussian_enabled:
            self.gaussian_enabled = False
            self.remove_gaussian_trackbars()
        
        if self.bilateral_enabled:
            self.bilateral_enabled = False
            self.remove_bilateral_trackbars()
        
        if self.canny_enabled:
            self.canny_enabled = False
            self.remove_canny_trackbars()
        
        if self.hough_enabled:
            self.hough_enabled = False
            self.remove_hough_trackbars()
        
        if self.transform_enabled:
            self.transform_enabled = False
            self.remove_transform_trackbars()
        
        if self.calibration_mode:
            self.calibration_mode = False
            self.remove_calibration_window()
        
        # Hide histogram
        self.show_histogram = False
        
        # Clear panorama images
        self.panorama_creator.clear_images()
        
        print("All settings reset to defaults")
    
    def add_info_overlay(self, image):
        """
        Add information overlay to the image.
        
        Args:
            image (numpy.ndarray): Image to add overlay to
        """
        info_lines = [
            "1=BGR 2=Gray 3=HSV | h=Hist g=Gauss b=Bilat e=Canny l=Hough t=Trans c=Calib | p=+Pano m=MakePano r=Reset s=Save q=Quit"
        ]
        
        # Add current settings info
        color_space_names = {ColorSpace.BGR: "BGR", ColorSpace.GRAY: "Grayscale", ColorSpace.HSV: "HSV"}
        current_cs = color_space_names[self.current_color_space]
        
        gaussian_status = "ON" if self.gaussian_enabled else "OFF"
        bilateral_status = "ON" if self.bilateral_enabled else "OFF"
        canny_status = "ON" if self.canny_enabled else "OFF"
        hough_status = "ON" if self.hough_enabled else "OFF"
        transform_status = "ON" if self.transform_enabled else "OFF"
        calib_status = "ON" if self.calibration_mode else "OFF"
        histogram_status = "ON" if self.show_histogram else "OFF"
        
        panorama_info = self.panorama_creator.get_status()
        
        info_lines.extend([
            f"ColorSpace: {current_cs} | Filters: G={gaussian_status} B={bilateral_status} E={canny_status} L={hough_status}",
            f"Transform: {transform_status} | Calibration: {calib_status} | Histogram: {histogram_status} | Panorama: {panorama_info['num_images']} imgs"
        ])
        
        self.display_manager.add_control_instructions(image, info_lines)
    
    def run(self):
        """Run the main application loop."""
        if not self.initialize():
            return
        
        self.running = True
        
        try:
            while self.running:
                # Read frame from webcam
                ret, frame = self.webcam.read_frame()
                if not ret:
                    print("Failed to read frame from webcam")
                    break
                
                # Process frame
                if self.calibration_mode:
                    # In calibration mode, show chessboard detection
                    display_frame = self.camera_calibrator.draw_chessboard_corners(frame)
                    
                    # Show calibration info in separate window
                    if self.calibration_window in self.display_manager.windows:
                        calib_info = self.camera_calibrator.get_calibration_info()
                        info_text = f"Calibration Images: {calib_info['num_images']}/10\n"
                        info_text += f"Ready to Calibrate: {calib_info['ready_to_calibrate']}\n"
                        if calib_info['is_calibrated']:
                            info_text += f"Calibration Error: {calib_info['calibration_error']:.3f}\n"
                        
                        # Create info image
                        info_img = np.zeros((200, 400, 3), dtype=np.uint8)
                        y_pos = 30
                        for line in info_text.split('\n'):
                            if line.strip():
                                cv2.putText(info_img, line, (10, y_pos), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                                y_pos += 30
                        
                        cv2.imshow(self.calibration_window, info_img)
                else:
                    # Normal processing mode
                    processed_frame = self.process_frame(frame)
                    display_frame = processed_frame.copy()
                
                # Add information overlay
                self.add_info_overlay(display_frame)
                
                # Display the frame
                self.display_manager.display_image(self.main_window, display_frame)
                
                # Update histogram if enabled (only in normal mode)
                if not self.calibration_mode:
                    self.update_histogram(display_frame)
                
                # Handle keyboard input
                key = self.display_manager.wait_for_key(1)
                if key != 255:  # Key was pressed
                    if not self.handle_keyboard_input(key):
                        break
        
        except KeyboardInterrupt:
            print("\nApplication interrupted by user")
        except Exception as e:
            print(f"Error in main loop: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources."""
        print("Cleaning up...")
        self.webcam.release()
        self.display_manager.destroy_all_windows()
        print("Application closed")


def main():
    """Main entry point."""
    app = ImageProcessingApp()
    app.run()


if __name__ == "__main__":
    main()
