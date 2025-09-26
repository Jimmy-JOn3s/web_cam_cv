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
from src.ar.ar_engine import ARController
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
        self.ar_controller = ARController()
        
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
        self.ar_window = 'Augmented Reality'
        
        # State variables
        self.current_color_space = ColorSpace.BGR
        self.show_histogram = False
        self.gaussian_enabled = False
        self.bilateral_enabled = False
        self.canny_enabled = False
        self.hough_enabled = False
        self.transform_enabled = False
        self.calibration_mode = False
        self.ar_mode = False
        self.panorama_mode = False
        self.panorama_window = 'Panorama Capture'
        self.running = False
        
    def initialize(self):
        """Initialize the application."""
        try:
            # Initialize webcam
            self.webcam.initialize()
            
            # Create main window as resizable
            cv2.namedWindow(self.main_window, cv2.WINDOW_NORMAL)
            
            # Get a sample frame to determine optimal window size
            ret, sample_frame = self.webcam.read_frame()
            if ret:
                frame_height, frame_width = sample_frame.shape[:2]
                # Use simple fixed max dimensions to avoid GUI conflicts
                max_width = 800
                max_height = 600
                
                # Calculate scale factor to fit within max dimensions
                scale_w = max_width / frame_width
                scale_h = max_height / frame_height
                scale = min(scale_w, scale_h, 1.0)  # Don't scale up, only down
                
                display_width = int(frame_width * scale)
                display_height = int(frame_height * scale)
                
                cv2.resizeWindow(self.main_window, display_width, display_height)
                print(f"Frame: {frame_width}x{frame_height}")
                print(f"Display size: {display_width}x{display_height} (scale: {scale:.2f})")
            
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
            print("  a: Toggle Augmented Reality mode")
            print("  p: Toggle interactive panorama capture mode")
            print("  SPACE: Add frame to panorama (in panorama mode)")
            print("  m: Create final panorama from collected images")
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
        
    def create_canny_trackbars(self):
        """Create trackbars for Canny edge detection in separate window."""
        self.display_manager.create_window(self.canny_window)
        self.canny_detector.create_trackbars(self.canny_window)
    
    def create_hough_trackbars(self):
        """Create trackbars for Hough line detection in separate window."""
        self.display_manager.create_window(self.hough_window)
        self.hough_detector.create_trackbars(self.hough_window)
    
    def create_transform_trackbars(self):
        """Create trackbars for geometric transformations in separate window."""
        self.display_manager.create_window(self.transform_window)
        self.geometric_transformer.create_trackbars(self.transform_window)
    
    def create_calibration_window(self):
        """Create window for camera calibration."""
        self.display_manager.create_window(self.calibration_window)
    
    def create_panorama_capture_window(self):
        """Create interactive panorama capture window."""
        self.display_manager.create_window(self.panorama_window)
        # Make it a wide window for panorama preview
        cv2.namedWindow(self.panorama_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.panorama_window, 1200, 400)  # Wide window
        try:
            cv2.moveWindow(self.panorama_window, 100, 200)
        except:
            pass
        
    def remove_window_if_exists(self, window_name):
        """Helper to remove window if it exists."""
        if window_name in self.display_manager.windows:
            self.display_manager.destroy_window(window_name)
    
    def setup_ar_mode(self):
        """Setup Augmented Reality mode."""
        try:
            # Load T-Rex model with bigger scale
            trex_path = os.path.join(os.path.dirname(__file__), 'assets', 'trex_model.obj')
            success = self.ar_controller.load_trex_model(trex_path, scale=15.0)
            
            if not success:
                print("Failed to load T-Rex model!")
                self.ar_mode = False
                return
            
            # Check if camera is calibrated
            if self.camera_calibrator.camera_matrix is not None:
                self.ar_controller.set_camera_calibration(
                    self.camera_calibrator.camera_matrix,
                    self.camera_calibrator.distortion_coefficients
                )
                print("Using existing camera calibration for AR")
            else:
                print("WARNING: Camera not calibrated! AR may not work properly.")
                print("Please calibrate camera first using 'c' key")
                
                # Use default camera parameters (rough estimates)
                # These are typical values for a webcam - not accurate!
                height, width = 480, 640  # Standard 480p resolution
                fx = fy = width * 0.7  # Rough focal length estimate
                cx, cy = width / 2, height / 2  # Center point
                
                camera_matrix = np.array([
                    [fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]
                ], dtype=np.float32)
                
                dist_coeffs = np.zeros(5, dtype=np.float32)  # No distortion
                
                self.ar_controller.set_camera_calibration(camera_matrix, dist_coeffs)
                print("Using default camera parameters (inaccurate)")
            
            print("AR Mode Setup Complete!")
            print("- Place ArUco marker in view")
            print("- T-Rex will appear on the marker")
            
        except Exception as e:
            print(f"Error setting up AR mode: {e}")
            self.ar_mode = False
    
    def arrange_windows(self):
        """Arrange windows in a nice layout to avoid overlaps."""
        try:
            # Main video window - top-left, leaving space for panorama
            cv2.moveWindow(self.main_window, 50, 50)
            
            # Controls window - top-right
            cv2.moveWindow(self.controls_window, 900, 50)
            
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
            
            if self.panorama_mode and self.panorama_window in self.display_manager.windows:
                cv2.moveWindow(self.panorama_window, 100, 200)
                
        except Exception as e:
            # Window positioning might not work on all systems
            pass
    
    def process_frame(self, frame):
        """
        Process a single frame with current settings (optimized).
        
        Args:
            frame (numpy.ndarray): Input frame
            
        Returns:
            numpy.ndarray: Processed frame
        """
        # Start with original frame
        processed = frame
        
        # Only copy frame if we need to modify it
        needs_processing = (
            self.trackbar_manager.get_brightness_contrast_values() != (0, 1.0) or
            self.transform_enabled or self.gaussian_enabled or self.bilateral_enabled or
            self.canny_enabled or self.hough_enabled or 
            self.current_color_space != ColorSpace.BGR
        )
        
        if not needs_processing:
            return frame
        
        processed = frame.copy()
        
        # Apply brightness and contrast only if changed from defaults
        brightness, contrast = self.trackbar_manager.get_brightness_contrast_values()
        if brightness != 0 or contrast != 1.0:
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
            processed = self.canny_detector.apply(processed, **canny_params)
        
        # Apply Hough line detection independently (works on original image)
        if self.hough_enabled:
            # Adjustable Hough detection with trackbar parameters
            hough_params = self.hough_detector.get_trackbar_values(self.hough_window)
            processed = self.hough_detector.apply(processed, **hough_params)
        
        # Apply color space conversion last
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
    
    def update_panorama_capture(self, current_frame):
        """
        Update interactive panorama capture window.
        
        Args:
            current_frame (numpy.ndarray): Current webcam frame
        """
        if self.panorama_mode:
            if self.panorama_window not in self.display_manager.windows:
                self.create_panorama_capture_window()
            
            # Get panorama status
            status = self.panorama_creator.get_status()
            
            # Create a wide canvas for panorama preview
            canvas_width = 1200
            canvas_height = 400
            canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
            
            # Add title and instructions
            cv2.putText(canvas, "Interactive Panorama Capture Mode", (20, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(canvas, f"Captured Images: {status['num_images']}", (20, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(canvas, "Press SPACE to capture | ESC to exit panorama mode", (20, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Show current frame (resized to fit)
            frame_display_height = 200
            frame_aspect = current_frame.shape[1] / current_frame.shape[0]
            frame_display_width = int(frame_display_height * frame_aspect)
            
            current_small = cv2.resize(current_frame, (frame_display_width, frame_display_height))
            
            # Position current frame preview
            y_start = 150
            x_start = 20
            canvas[y_start:y_start + frame_display_height, 
                   x_start:x_start + frame_display_width] = current_small
            
            # Add border around current frame
            cv2.rectangle(canvas, (x_start-2, y_start-2), 
                         (x_start + frame_display_width+2, y_start + frame_display_height+2), 
                         (0, 255, 0), 2)
            cv2.putText(canvas, "Current View", (x_start, y_start-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Show captured images as thumbnails
            if status['num_images'] > 0:
                thumbnail_width = 120
                thumbnail_height = 80
                x_offset = x_start + frame_display_width + 40
                
                for i in range(min(status['num_images'], 8)):  # Show max 8 thumbnails
                    if i < len(self.panorama_creator.images):
                        img = self.panorama_creator.images[i]
                        thumbnail = cv2.resize(img, (thumbnail_width, thumbnail_height))
                        
                        x_pos = x_offset + (i % 4) * (thumbnail_width + 10)
                        y_pos = y_start + (i // 4) * (thumbnail_height + 30)
                        
                        if y_pos + thumbnail_height < canvas_height:
                            canvas[y_pos:y_pos + thumbnail_height, 
                                   x_pos:x_pos + thumbnail_width] = thumbnail
                            
                            # Add thumbnail border and number
                            cv2.rectangle(canvas, (x_pos-1, y_pos-1), 
                                         (x_pos + thumbnail_width+1, y_pos + thumbnail_height+1), 
                                         (255, 255, 255), 1)
                            cv2.putText(canvas, f"{i+1}", (x_pos+5, y_pos+15), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Show overlap indicator
            if status['num_images'] > 0:
                cv2.putText(canvas, "Tip: Overlap ~30% with previous image for better stitching", 
                           (20, canvas_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.imshow(self.panorama_window, canvas)
        else:
            if self.panorama_window in self.display_manager.windows:
                self.display_manager.destroy_window(self.panorama_window)
    
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
                self.remove_window_if_exists(self.gaussian_window)
                print("Gaussian filter: OFF - Window closed")
        elif key == ord('b'):
            # Toggle Bilateral filter
            self.bilateral_enabled = not self.bilateral_enabled
            if self.bilateral_enabled:
                self.create_bilateral_trackbars()
                self.arrange_filter_windows()
                print("Bilateral filter: ON - Separate window created")
            else:
                self.remove_window_if_exists(self.bilateral_window)
                print("Bilateral filter: OFF - Window closed")
        elif key == ord('e'):
            # Toggle Canny edge detection
            self.canny_enabled = not self.canny_enabled
            if self.canny_enabled:
                self.create_canny_trackbars()
                self.arrange_filter_windows()
                print("Canny edge detection: ON - Separate window created")
            else:
                self.remove_window_if_exists(self.canny_window)
                print("Canny edge detection: OFF - Window closed")
        elif key == ord('l'):
            # Toggle Hough line detection
            self.hough_enabled = not self.hough_enabled
            if self.hough_enabled:
                self.create_hough_trackbars()
                self.arrange_filter_windows()
                print("Hough line detection: ON - Separate window created")
            else:
                self.remove_window_if_exists(self.hough_window)
                print("Hough line detection: OFF - Window closed")
        elif key == ord('t'):
            # Toggle geometric transformations
            self.transform_enabled = not self.transform_enabled
            if self.transform_enabled:
                self.create_transform_trackbars()
                self.arrange_filter_windows()
                print("Geometric transformations: ON - Separate window created")
            else:
                self.remove_window_if_exists(self.transform_window)
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
                self.remove_window_if_exists(self.calibration_window)
                print("Camera calibration mode: OFF")
        elif key == ord('a'):
            # Toggle Augmented Reality mode
            self.ar_mode = not self.ar_mode
            if self.ar_mode:
                self.setup_ar_mode()
                print("Augmented Reality mode: ON")
                print("AR Controls: v=Render mode, n=Animation, x=Axis")
            else:
                print("Augmented Reality mode: OFF")
        # AR specific controls (only work in AR mode)
        elif self.ar_mode and key == ord('v'):
            self.ar_controller.cycle_render_mode()
        elif self.ar_mode and key == ord('n'):
            self.ar_controller.toggle_animation()
        elif self.ar_mode and key == ord('x'):
            self.ar_controller.toggle_axis()
        elif key == ord('p'):
            # Toggle interactive panorama capture mode
            self.panorama_mode = not self.panorama_mode
            if self.panorama_mode:
                self.create_panorama_capture_window()
                self.arrange_filter_windows()
                print("Interactive panorama capture mode: ON")
                print("Press SPACE to capture frames, ESC to exit panorama mode")
            else:
                self.remove_window_if_exists(self.panorama_window)
                print("Interactive panorama capture mode: OFF")
        elif key == ord('m'):
            # Create panorama
            panorama = self.panorama_creator.create_panorama()
            if panorama is not None:
                self.panorama_creator.save_panorama(panorama)
                print(f"Panorama created! Size: {panorama.shape[1]}x{panorama.shape[0]} pixels")
                
                # Display panorama in a new window with auto-sizing
                window_name = 'Panorama Result'
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                
                # Get screen dimensions for smart scaling
                h, w = panorama.shape[:2]
                
                # Screen size limits (adjust based on typical screen sizes)
                screen_width = 1920   # Typical screen width
                screen_height = 1080  # Typical screen height
                max_width = int(screen_width * 0.9)    # Use 90% of screen width
                max_height = int(screen_height * 0.8)  # Use 80% of screen height
                
                # Calculate display size
                if w > max_width or h > max_height:
                    # Calculate scaling factor to fit within screen
                    scale_w = max_width / w
                    scale_h = max_height / h
                    scale = min(scale_w, scale_h)
                    
                    display_w = int(w * scale)
                    display_h = int(h * scale)
                    
                    # Resize panorama for display
                    display_panorama = cv2.resize(panorama, (display_w, display_h))
                    print(f"Panorama auto-scaled for display: {display_w}x{display_h} pixels (scale: {scale:.2f})")
                else:
                    display_panorama = panorama
                    display_w, display_h = w, h
                    print(f"Panorama displayed at original size: {w}x{h} pixels")
                
                # Auto-resize window to exactly fit the panorama
                cv2.resizeWindow(window_name, display_w, display_h)
                
                # Show the panorama
                cv2.imshow(window_name, display_panorama)
                
                # Position panorama window away from main UI
                try:
                    cv2.moveWindow(window_name, 50, 100)
                except:
                    pass  # Window positioning might not work on all systems
                
                print("Panorama displayed with auto-sized window. Press any key in panorama window to close it.")
            else:
                print("Could not create panorama. Make sure you've added at least 2 images with 'p' key.")
        elif key == 32:  # SPACE key
            if self.calibration_mode:
                # Capture calibration image (only in calibration mode)
                ret, frame = self.webcam.read_frame()
                if ret:
                    self.camera_calibrator.add_calibration_image(frame)
            elif self.panorama_mode:
                # Capture frame for panorama (in panorama mode)
                ret, frame = self.webcam.read_frame()
                if ret:
                    success = self.panorama_creator.add_image(frame)
                    if success:
                        status = self.panorama_creator.get_status()
                        print(f"Frame added to panorama collection. Total images: {status['num_images']}")
                    else:
                        print("Failed to add frame to panorama collection")
        elif key == 27:  # ESC key
            if self.panorama_mode:
                # Exit panorama mode
                self.panorama_mode = False
                self.remove_window_if_exists(self.panorama_window)
                print("Exited panorama capture mode")
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
            self.remove_window_if_exists(self.gaussian_window)
        
        if self.bilateral_enabled:
            self.bilateral_enabled = False
            self.remove_window_if_exists(self.bilateral_window)
        
        if self.canny_enabled:
            self.canny_enabled = False
            self.remove_window_if_exists(self.canny_window)
        
        if self.hough_enabled:
            self.hough_enabled = False
            self.remove_window_if_exists(self.hough_window)
        
        if self.transform_enabled:
            self.transform_enabled = False
            self.remove_window_if_exists(self.transform_window)
        
        if self.calibration_mode:
            self.calibration_mode = False
            self.remove_window_if_exists(self.calibration_window)
        
        if self.panorama_mode:
            self.panorama_mode = False
            self.remove_window_if_exists(self.panorama_window)
        
        # Hide histogram
        self.show_histogram = False
        
        # Clear panorama images
        self.panorama_creator.clear_images()
        
        # Clear calibration data
        self.camera_calibrator.clear_calibration_data()
        
        print("All settings reset to defaults")
    
    def add_info_overlay(self, image):
        """
        Add information overlay to the image.
        
        Args:
            image (numpy.ndarray): Image to add overlay to
        """
        info_lines = [
            "1=BGR 2=Gray 3=HSV | h=Hist g=Gauss b=Bilat e=Canny l=Hough t=Trans c=Calib a=AR | p=PanoMode m=MakePano r=Reset s=Save q=Quit"
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
        ar_status = "ON" if self.ar_mode else "OFF"
        panorama_status = "ON" if self.panorama_mode else "OFF"
        histogram_status = "ON" if self.show_histogram else "OFF"
        
        panorama_info = self.panorama_creator.get_status()
        
        # Different status info for AR mode
        if self.ar_mode:
            info_lines.extend([
                f"AR MODE: {ar_status} | v=RenderMode n=Animation x=Axis | Place ArUco marker in view",
                f"T-Rex Model Loaded | Camera: {'Calibrated' if self.camera_calibrator.camera_matrix is not None else 'Default'}"
            ])
        else:
            info_lines.extend([
                f"ColorSpace: {current_cs} | Filters: G={gaussian_status} B={bilateral_status} E={canny_status} L={hough_status}",
                f"Transform: {transform_status} | Calib: {calib_status} | AR: {ar_status} | PanoMode: {panorama_status} | Hist: {histogram_status} | Frames: {panorama_info['num_images']}"
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
                    # Use the consolidated calibration processing method
                    display_frame, calib_info = self.camera_calibrator.process_calibration_frame(frame)
                    
                    # Show calibration info in separate window
                    if self.calibration_window in self.display_manager.windows:
                        info_text = f"AUTO-CAPTURE MODE\n"
                        info_text += f"Images: {calib_info['num_calibration_images']}/{calib_info['target_images']}\n"
                        info_text += f"Ready to Calibrate: {calib_info['ready_to_calibrate']}\n"
                        if calib_info['is_calibrated']:
                            info_text += f"Calibration Error: {calib_info['calibration_error']:.3f}\n"
                        info_text += "\nHow it works:\n"
                        info_text += "• Point camera at chessboard pattern\n"
                        info_text += "• Auto-captures every 2 seconds when detected\n"
                        info_text += "• Move to different angles for variety\n"
                        info_text += "• Manual capture: Press SPACE\n"
                        info_text += f"• Need {calib_info['target_images']} images for calibration"
                        
                        # Create info image
                        info_img = np.zeros((300, 450, 3), dtype=np.uint8)
                        y_pos = 25
                        for line in info_text.split('\n'):
                            if line.strip():
                                cv2.putText(info_img, line, (10, y_pos), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                                y_pos += 25
                        
                        cv2.imshow(self.calibration_window, info_img)
                elif self.ar_mode:
                    # AR mode processing
                    display_frame = self.ar_controller.process_frame(frame)
                else:
                    # Normal processing mode
                    processed_frame = self.process_frame(frame)
                    display_frame = processed_frame.copy()
                
                # Add information overlay
                self.add_info_overlay(display_frame)
                
                # Display the frame
                self.display_manager.display_image(self.main_window, display_frame)
                
                # Update histogram if enabled (only in normal mode)
                if not self.calibration_mode and not self.ar_mode:
                    self.update_histogram(display_frame)
                
                # Update panorama capture window if enabled
                if self.panorama_mode:
                    self.update_panorama_capture(frame)
                
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
