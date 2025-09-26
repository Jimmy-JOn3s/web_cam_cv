"""
ArUco Marker Detection and Pose Estimation
Handles marker detection and camera pose calculation for AR
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List


class ArUcoDetector:
    """ArUco marker detection and pose estimation"""
    
    def __init__(self, marker_size: float = 0.05):
        """
        Initialize ArUco detector
        
        Args:
            marker_size: Real-world size of marker in meters (default: 5cm)
        """
        self.marker_size = marker_size
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.detector_params = cv2.aruco.DetectorParameters()
        
        # Create detector
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.detector_params)
        
    def detect_markers(self, image: np.ndarray) -> Tuple[List[np.ndarray], List[int]]:
        """
        Detect ArUco markers in image
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (corners, ids) where:
            - corners: List of corner coordinates for each detected marker
            - ids: List of marker IDs
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        corners, ids, _ = self.detector.detectMarkers(gray)
        
        if ids is not None:
            ids = ids.flatten()
        else:
            ids = []
            
        return corners, ids
    
    def estimate_pose(self, corners: List[np.ndarray], camera_matrix: np.ndarray, 
                     dist_coeffs: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Estimate pose of detected markers
        
        Args:
            corners: Detected marker corners
            camera_matrix: Camera intrinsic matrix
            dist_coeffs: Camera distortion coefficients
            
        Returns:
            Tuple of (rvecs, tvecs) - rotation and translation vectors
        """
        if len(corners) == 0:
            return [], []
        
        # Define 3D coordinates of marker corners (in marker coordinate system)
        # Marker is in XY plane, Z=0
        half_size = self.marker_size / 2
        object_points = np.array([
            [-half_size, -half_size, 0],  # Bottom-left
            [ half_size, -half_size, 0],  # Bottom-right
            [ half_size,  half_size, 0],  # Top-right
            [-half_size,  half_size, 0]   # Top-left
        ], dtype=np.float32)
        
        rvecs = []
        tvecs = []
        
        for corner in corners:
            # Estimate pose for each marker
            success, rvec, tvec = cv2.solvePnP(
                object_points,
                corner[0],  # corner is shape (1, 4, 2), we need (4, 2)
                camera_matrix,
                dist_coeffs
            )
            
            if success:
                rvecs.append(rvec)
                tvecs.append(tvec)
        
        return rvecs, tvecs
    
    def draw_markers(self, image: np.ndarray, corners: List[np.ndarray], 
                    ids: List[int]) -> np.ndarray:
        """
        Draw detected markers on image
        
        Args:
            image: Input image
            corners: Detected marker corners
            ids: Marker IDs
            
        Returns:
            Image with markers drawn
        """
        result = image.copy()
        
        if len(corners) > 0:
            # Draw marker boundaries
            cv2.aruco.drawDetectedMarkers(result, corners, np.array(ids))
        
        return result
    
    def draw_axis(self, image: np.ndarray, camera_matrix: np.ndarray, 
                  dist_coeffs: np.ndarray, rvec: np.ndarray, tvec: np.ndarray,
                  length: float = 0.03) -> np.ndarray:
        """
        Draw 3D coordinate axis on marker
        
        Args:
            image: Input image
            camera_matrix: Camera intrinsic matrix
            dist_coeffs: Camera distortion coefficients
            rvec: Rotation vector
            tvec: Translation vector
            length: Axis length in meters
            
        Returns:
            Image with axis drawn
        """
        # Define 3D axis points
        axis_points = np.array([
            [0, 0, 0],        # Origin
            [length, 0, 0],   # X-axis (red)
            [0, length, 0],   # Y-axis (green)
            [0, 0, -length]   # Z-axis (blue) - negative because camera looks down -Z
        ], dtype=np.float32)
        
        # Project to image coordinates
        projected_points, _ = cv2.projectPoints(
            axis_points, rvec, tvec, camera_matrix, dist_coeffs
        )
        
        projected_points = projected_points.reshape(-1, 2).astype(int)
        
        result = image.copy()
        
        # Draw axis lines
        origin = tuple(projected_points[0])
        x_end = tuple(projected_points[1])
        y_end = tuple(projected_points[2])
        z_end = tuple(projected_points[3])
        
        # X-axis (red)
        cv2.arrowedLine(result, origin, x_end, (0, 0, 255), 3)
        # Y-axis (green)
        cv2.arrowedLine(result, origin, y_end, (0, 255, 0), 3)
        # Z-axis (blue)
        cv2.arrowedLine(result, origin, z_end, (255, 0, 0), 3)
        
        return result


class PoseEstimator:
    """Camera pose estimation utilities"""
    
    @staticmethod
    def project_points(points_3d: np.ndarray, rvec: np.ndarray, tvec: np.ndarray,
                      camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> np.ndarray:
        """
        Project 3D points to image coordinates
        
        Args:
            points_3d: 3D points in world coordinates (Nx3)
            rvec: Rotation vector
            tvec: Translation vector
            camera_matrix: Camera intrinsic matrix
            dist_coeffs: Camera distortion coefficients
            
        Returns:
            2D image coordinates (Nx2)
        """
        if len(points_3d) == 0:
            return np.array([])
        
        projected, _ = cv2.projectPoints(
            points_3d.astype(np.float32),
            rvec, tvec,
            camera_matrix, dist_coeffs
        )
        
        return projected.reshape(-1, 2)
    
    @staticmethod
    def draw_wireframe(image: np.ndarray, vertices_2d: np.ndarray, 
                      edges: List[Tuple[int, int]], color: Tuple[int, int, int] = (0, 255, 0),
                      thickness: int = 2) -> np.ndarray:
        """
        Draw wireframe model on image
        
        Args:
            image: Input image
            vertices_2d: 2D projected vertices
            edges: List of edge connections (vertex index pairs)
            color: Line color (B, G, R)
            thickness: Line thickness
            
        Returns:
            Image with wireframe drawn
        """
        result = image.copy()
        
        if len(vertices_2d) == 0:
            return result
        
        vertices_2d = vertices_2d.astype(int)
        
        for edge in edges:
            v1_idx, v2_idx = edge
            
            # Check if vertex indices are valid
            if 0 <= v1_idx < len(vertices_2d) and 0 <= v2_idx < len(vertices_2d):
                pt1 = tuple(vertices_2d[v1_idx])
                pt2 = tuple(vertices_2d[v2_idx])
                
                # Only draw if both points are within image bounds
                h, w = image.shape[:2]
                if (0 <= pt1[0] < w and 0 <= pt1[1] < h and 
                    0 <= pt2[0] < w and 0 <= pt2[1] < h):
                    cv2.line(result, pt1, pt2, color, thickness)
        
        return result
    
    @staticmethod
    def draw_solid_model(image: np.ndarray, vertices_2d: np.ndarray,
                        faces: List[List[int]], color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """
        Draw solid 3D model using face filling
        
        Args:
            image: Input image
            vertices_2d: 2D projected vertices
            faces: List of face vertex indices
            color: Fill color (B, G, R)
            
        Returns:
            Image with solid model drawn
        """
        result = image.copy()
        
        if len(vertices_2d) == 0:
            return result
        
        vertices_2d = vertices_2d.astype(int)
        
        for face in faces:
            if len(face) >= 3:
                # Get face vertices
                face_points = []
                valid_face = True
                
                for vertex_idx in face:
                    if 0 <= vertex_idx < len(vertices_2d):
                        face_points.append(vertices_2d[vertex_idx])
                    else:
                        valid_face = False
                        break
                
                if valid_face and len(face_points) >= 3:
                    # Convert to numpy array
                    face_points = np.array(face_points, dtype=np.int32)
                    
                    # Fill triangle/polygon
                    cv2.fillPoly(result, [face_points], color)
                    
                    # Draw outline
                    cv2.polylines(result, [face_points], True, (0, 0, 0), 1)
        
        return result
