"""
Camera Calibration Module
"""
import cv2
import numpy as np
import pickle
import os
from datetime import datetime


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
        
        # Prepare object points (0,0,0), (1,0,0), (2,0,0), ..., (8,5,0)
        self.objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        
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
        Draw chessboard corners on image if detected.
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            numpy.ndarray: Image with drawn corners
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        result = image.copy()
        
        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)
        
        if ret:
            # Draw corners
            cv2.drawChessboardCorners(result, self.chessboard_size, corners, ret)
            
            # Add text indicating detection
            cv2.putText(result, f'Chessboard Detected ({self.chessboard_size[0]}x{self.chessboard_size[1]})', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(result, 'No Chessboard Pattern', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Add instruction text
        cv2.putText(result, f'Calibration Images: {len(self.calibration_images)}', 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(result, 'Press SPACE to capture, C to calibrate', 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
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
    
    def undistort_image(self, image):
        """
        Undistort image using calibration parameters.
        
        Args:
            image (numpy.ndarray): Distorted image
            
        Returns:
            numpy.ndarray: Undistorted image
        """
        if self.camera_matrix is None or self.distortion_coefficients is None:
            print("Camera not calibrated yet")
            return image
        
        return cv2.undistort(image, self.camera_matrix, self.distortion_coefficients)
    
    def save_calibration(self, filename=None):
        """
        Save calibration parameters to file.
        
        Args:
            filename (str): Output filename
        """
        if self.camera_matrix is None:
            print("No calibration data to save")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"camera_calibration_{timestamp}.pkl"
        
        calibration_data = {
            'camera_matrix': self.camera_matrix,
            'distortion_coefficients': self.distortion_coefficients,
            'rotation_vectors': self.rotation_vectors,
            'translation_vectors': self.translation_vectors,
            'calibration_error': self.calibration_error,
            'chessboard_size': self.chessboard_size,
            'num_images': len(self.calibration_images)
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(calibration_data, f)
        
        print(f"Calibration data saved to {filename}")
    
    def load_calibration(self, filename):
        """
        Load calibration parameters from file.
        
        Args:
            filename (str): Input filename
            
        Returns:
            bool: True if loading successful
        """
        try:
            with open(filename, 'rb') as f:
                calibration_data = pickle.load(f)
            
            self.camera_matrix = calibration_data['camera_matrix']
            self.distortion_coefficients = calibration_data['distortion_coefficients']
            self.rotation_vectors = calibration_data.get('rotation_vectors')
            self.translation_vectors = calibration_data.get('translation_vectors')
            self.calibration_error = calibration_data.get('calibration_error')
            self.chessboard_size = calibration_data.get('chessboard_size', self.chessboard_size)
            
            print(f"Calibration data loaded from {filename}")
            print(f"Reprojection error: {self.calibration_error:.3f} pixels")
            return True
            
        except Exception as e:
            print(f"Error loading calibration: {e}")
            return False
    
    def clear_calibration_data(self):
        """Clear all calibration data."""
        self.calibration_images.clear()
        self.object_points.clear()
        self.image_points.clear()
        self.camera_matrix = None
        self.distortion_coefficients = None
        self.rotation_vectors = None
        self.translation_vectors = None
        self.calibration_error = None
        print("Calibration data cleared")
    
    def get_calibration_info(self):
        """Get current calibration status."""
        return {
            'num_calibration_images': len(self.calibration_images),
            'is_calibrated': self.camera_matrix is not None,
            'calibration_error': self.calibration_error,
            'chessboard_size': self.chessboard_size,
            'ready_to_calibrate': len(self.calibration_images) >= 10
        }
    
    def create_chessboard_pattern(self, square_size=30, save_path='chessboard_pattern.png'):
        """
        Create a chessboard pattern for printing.
        
        Args:
            square_size (int): Size of each square in pixels
            save_path (str): Path to save the pattern
        """
        pattern_width = (self.chessboard_size[0] + 1) * square_size
        pattern_height = (self.chessboard_size[1] + 1) * square_size
        
        pattern = np.zeros((pattern_height, pattern_width), dtype=np.uint8)
        
        for i in range(self.chessboard_size[1] + 1):
            for j in range(self.chessboard_size[0] + 1):
                if (i + j) % 2 == 0:
                    y1, y2 = i * square_size, (i + 1) * square_size
                    x1, x2 = j * square_size, (j + 1) * square_size
                    pattern[y1:y2, x1:x2] = 255
        
        cv2.imwrite(save_path, pattern)
        print(f"Chessboard pattern saved to {save_path}")
        print(f"Pattern size: {self.chessboard_size[0]+1} x {self.chessboard_size[1]+1} squares")
        print(f"Print this pattern and use it for calibration")
