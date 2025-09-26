"""
Camera Calibration Module
"""
import cv2
import numpy as np
import time


class CameraCalibrator:
    """Handles camera calibration using chessboard patterns."""
    
    def __init__(self, chessboard_size=(9, 6)):
        """
        Initialize camera calibrator.
        
        Args:
            chessboard_size (tuple): Number of internal corners (width, height)
        """
        self.chessboard_size = chessboard_size
        self.calibration_images = []
        self.object_points = []  # 3D points in real world space
        self.image_points = []   # 2D points in image plane
        
        # Camera parameters
        self.camera_matrix = None
        self.distortion_coefficients = None
        self.rotation_vectors = None
        self.translation_vectors = None
        self.calibration_error = None
        
        # Auto-capture settings
        self.last_capture_time = 0
        self.capture_interval = 2.0  # seconds between auto-captures
        self.target_images = 15  # Number of images needed for calibration
        
        # Prepare object points (0,0,0), (1,0,0), (2,0,0), ..., (8,5,0)
        self.objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        
    def process_calibration_frame(self, frame):
        """
        Process frame in calibration mode with auto-capture and visual feedback.
        
        Args:
            frame (numpy.ndarray): Input frame
            
        Returns:
            tuple: (processed_frame, calibration_status)
        """
        try:
            display_frame = self.draw_chessboard_corners(frame)
            
            # Get calibration status
            calib_info = self.get_calibration_info()
            
            # Auto-calibrate when enough images are collected
            if calib_info['ready_to_calibrate'] and not calib_info['is_calibrated']:
                print("Auto-calibrating camera...")
                height, width = frame.shape[:2]
                success = self.calibrate_camera((width, height))
                if success:
                    print("✅ Camera calibration completed successfully!")
                else:
                    print("❌ Camera calibration failed. Try capturing more images.")
            
            return display_frame, calib_info
            
        except Exception as e:
            print(f"Error in calibration mode: {e}")
            error_frame = frame.copy()
            cv2.putText(error_frame, f"Calibration Error: {str(e)}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return error_frame, self.get_calibration_info()
    
    def auto_capture_calibration_image(self, image):
        """
        Auto-capture calibration image if chessboard is detected and enough time has passed.
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            bool: True if image was captured
        """
        current_time = time.time()
        
        # Check if enough time has passed since last capture
        if current_time - self.last_capture_time < self.capture_interval:
            return False
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)
        
        if ret:
            # Refine corner positions
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # Store object points and image points
            self.object_points.append(self.objp)
            self.image_points.append(corners_refined)
            self.calibration_images.append(image.copy())
            
            # Update capture time
            self.last_capture_time = current_time
            
            print(f"✅ Auto-captured calibration image {len(self.calibration_images)}/{self.target_images}")
            return True
        
        return False
    
    def add_calibration_image(self, image):
        """
        Add an image for calibration if it contains a chessboard pattern.
        
        Args:
            image (numpy.ndarray): Calibration image
            
        Returns:
            bool: True if chessboard was found and added
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)
        
        if ret:
            # Refine corner positions
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # Store object points and image points
            self.object_points.append(self.objp)
            self.image_points.append(corners_refined)
            self.calibration_images.append(image.copy())
            
            print(f"Chessboard detected! Total calibration images: {len(self.calibration_images)}")
            return True
        else:
            print("No chessboard pattern found in image")
            return False
    
    def draw_chessboard_corners(self, image):
        """
        Draw chessboard corners on image if detected and show auto-capture status.
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            numpy.ndarray: Image with drawn corners and status
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        result = image.copy()
        
        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)
        
        current_time = time.time()
        time_since_last = current_time - self.last_capture_time
        can_capture = time_since_last >= self.capture_interval
        
        if ret:
            # Draw corners
            cv2.drawChessboardCorners(result, self.chessboard_size, corners, ret)
            
            # Auto-capture if enough time has passed
            if can_capture and len(self.calibration_images) < self.target_images:
                self.auto_capture_calibration_image(image)
                status_color = (0, 255, 0)  # Green
                status_text = f'CAPTURED! ({len(self.calibration_images)}/{self.target_images})'
            else:
                if len(self.calibration_images) >= self.target_images:
                    status_color = (0, 255, 255)  # Yellow
                    status_text = f'Complete! ({len(self.calibration_images)}/{self.target_images})'
                else:
                    # Show countdown
                    remaining_time = self.capture_interval - time_since_last
                    status_color = (0, 165, 255)  # Orange
                    status_text = f'Next capture in {remaining_time:.1f}s ({len(self.calibration_images)}/{self.target_images})'
        else:
            status_color = (0, 0, 255)  # Red
            status_text = f'No Chessboard - Move closer or improve lighting ({len(self.calibration_images)}/{self.target_images})'
        
        # Add status text with background
        cv2.rectangle(result, (5, 5), (result.shape[1]-5, 100), (0, 0, 0), -1)
        cv2.putText(result, 'AUTO-CAPTURE MODE', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(result, status_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # Add manual capture option
        cv2.putText(result, 'Press SPACE for manual capture', (10, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return result
    
    def calibrate_camera(self, image_size):
        """
        Perform camera calibration.
        
        Args:
            image_size (tuple): Image size (width, height)
            
        Returns:
            bool: True if calibration successful
        """
        if len(self.object_points) < 10:
            print(f"Need at least 10 calibration images, have {len(self.object_points)}")
            return False
        
        print("Performing camera calibration...")
        
        try:
            # Calibrate camera
            ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                self.object_points, self.image_points, image_size, None, None
            )
            
            if ret:
                self.camera_matrix = camera_matrix
                self.distortion_coefficients = dist_coeffs
                self.rotation_vectors = rvecs
                self.translation_vectors = tvecs
                
                # Calculate reprojection error
                total_error = 0
                for i in range(len(self.object_points)):
                    projected_points, _ = cv2.projectPoints(
                        self.object_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
                    )
                    error = cv2.norm(self.image_points[i], projected_points, cv2.NORM_L2) / len(projected_points)
                    total_error += error
                
                self.calibration_error = total_error / len(self.object_points)
                
                print("Camera calibration successful!")
                print(f"Reprojection error: {self.calibration_error:.3f} pixels")
                print(f"Focal length (fx, fy): ({camera_matrix[0,0]:.2f}, {camera_matrix[1,1]:.2f})")
                print(f"Principal point (cx, cy): ({camera_matrix[0,2]:.2f}, {camera_matrix[1,2]:.2f})")
                
                return True
            else:
                print("Camera calibration failed")
                return False
                
        except Exception as e:
            print(f"Error during calibration: {e}")
            return False
    
    def get_calibration_info(self):
        """Get current calibration status."""
        return {
            'num_calibration_images': len(self.calibration_images),
            'target_images': self.target_images,
            'is_calibrated': self.camera_matrix is not None,
            'calibration_error': self.calibration_error,
            'chessboard_size': self.chessboard_size,
            'ready_to_calibrate': len(self.calibration_images) >= self.target_images
        }
    
    def clear_calibration_data(self):
        """Clear all calibration data and reset state."""
        self.calibration_images.clear()
        self.object_points.clear()
        self.image_points.clear()
        self.camera_matrix = None
        self.distortion_coefficients = None
        self.rotation_vectors = None
        self.translation_vectors = None
        self.calibration_error = None
        self.last_capture_time = 0
        print("Calibration data cleared")
