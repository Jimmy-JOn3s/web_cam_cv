"""
Augmented Reality Engine
Main AR processing pipeline that combines camera, markers, and 3D models
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List
from .aruco_detector import ArUcoDetector, PoseEstimator
from .model_loader import OBJLoader, Model3D


class AREngine:
    """Main AR processing engine"""
    
    def __init__(self, camera_matrix: Optional[np.ndarray] = None,
                 dist_coeffs: Optional[np.ndarray] = None):
        """
        Initialize AR Engine
        
        Args:
            camera_matrix: Camera intrinsic matrix (3x3)
            dist_coeffs: Camera distortion coefficients
        """
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        
        # ArUco detector
        self.aruco_detector = ArUcoDetector(marker_size=0.05)  # 5cm markers
        
        # 3D models
        self.models = {}
        
        # Rendering options
        self.render_mode = 'wireframe'  # 'wireframe', 'solid', 'both'
        self.show_axis = True
        self.show_markers = True
        
    def set_camera_parameters(self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray):
        """Set camera calibration parameters"""
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
    
    def load_model(self, model_name: str, obj_file_path: str, scale: float = 1.0) -> bool:
        """
        Load 3D model from OBJ file
        
        Args:
            model_name: Name to identify the model
            obj_file_path: Path to OBJ file
            scale: Scale factor for the model
            
        Returns:
            bool: True if loaded successfully
        """
        try:
            obj_loader = OBJLoader()
            success = obj_loader.load_obj(obj_file_path, scale)
            
            if success:
                # Center and scale the model appropriately
                obj_loader.center_model()
                obj_loader.scale_model(0.15)  # 15cm maximum dimension (5x larger!)
                
                model_3d = Model3D(obj_loader)
                self.models[model_name] = model_3d
                
                print(f"Loaded model '{model_name}': {len(obj_loader.get_vertices())} vertices")
                return True
            else:
                print(f"Failed to load model from {obj_file_path}")
                return False
                
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def set_model_transform(self, model_name: str, position: Tuple[float, float, float] = (0, 0, 0),
                           rotation: Tuple[float, float, float] = (0, 0, 0), scale: float = 1.0):
        """
        Set model transformation
        
        Args:
            model_name: Name of the model
            position: (x, y, z) position relative to marker
            rotation: (rx, ry, rz) rotation in degrees
            scale: Scale factor
        """
        if model_name in self.models:
            model = self.models[model_name]
            model.set_position(*position)
            model.set_rotation(*rotation)
            model.set_scale(scale)
    
    def set_render_mode(self, mode: str):
        """Set rendering mode: 'wireframe', 'solid', or 'both'"""
        if mode in ['wireframe', 'solid', 'both']:
            self.render_mode = mode
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process frame and render AR content
        
        Args:
            frame: Input camera frame
            
        Returns:
            Frame with AR content rendered
        """
        if self.camera_matrix is None or self.dist_coeffs is None:
            # Draw error message
            result = frame.copy()
            cv2.putText(result, "Camera not calibrated!", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return result
        
        # Detect ArUco markers
        corners, ids = self.aruco_detector.detect_markers(frame)
        
        if len(corners) == 0:
            # No markers detected
            result = frame.copy()
            cv2.putText(result, "No markers detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            return result
        
        # Estimate marker poses
        rvecs, tvecs = self.aruco_detector.estimate_pose(corners, self.camera_matrix, self.dist_coeffs)
        
        result = frame.copy()
        
        # Draw markers if enabled
        if self.show_markers:
            result = self.aruco_detector.draw_markers(result, corners, ids)
        
        # Render 3D content for each detected marker
        for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
            # Draw coordinate axis if enabled
            if self.show_axis:
                result = self.aruco_detector.draw_axis(result, self.camera_matrix,
                                                     self.dist_coeffs, rvec, tvec)
            
            # Render 3D models
            for model_name, model in self.models.items():
                result = self._render_model(result, model, rvec, tvec)
        
        # Draw status information
        self._draw_status(result, len(corners))
        
        return result
    
    def _render_model(self, image: np.ndarray, model: Model3D,
                     rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
        """
        Render a 3D model at the marker position
        
        Args:
            image: Input image
            model: 3D model to render
            rvec: Marker rotation vector
            tvec: Marker translation vector
            
        Returns:
            Image with model rendered
        """
        try:
            # Get transformed model vertices
            vertices_3d = model.get_transformed_vertices()
            
            if len(vertices_3d) == 0:
                return image
            
            # Project 3D vertices to 2D image coordinates
            vertices_2d = PoseEstimator.project_points(
                vertices_3d, rvec, tvec, self.camera_matrix, self.dist_coeffs
            )
            
            if len(vertices_2d) == 0:
                return image
            
            result = image.copy()
            
            # Render based on mode
            if self.render_mode in ['wireframe', 'both']:
                # Draw wireframe
                edges = model.get_edges()
                result = PoseEstimator.draw_wireframe(
                    result, vertices_2d, edges, color=(0, 255, 0), thickness=2
                )
            
            if self.render_mode in ['solid', 'both']:
                # Draw solid model with transparency
                faces = model.get_faces()
                
                # Create overlay for transparency
                overlay = result.copy()
                overlay = PoseEstimator.draw_solid_model(
                    overlay, vertices_2d, faces, color=(0, 200, 100)
                )
                
                # Blend with original image
                alpha = 0.7
                result = cv2.addWeighted(result, alpha, overlay, 1 - alpha, 0)
            
            return result
            
        except Exception as e:
            print(f"Error rendering model: {e}")
            return image
    
    def _draw_status(self, image: np.ndarray, num_markers: int):
        """Draw status information on image"""
        status_lines = [
            f"Markers detected: {num_markers}",
            f"Render mode: {self.render_mode}",
            f"Models loaded: {len(self.models)}"
        ]
        
        for i, line in enumerate(status_lines):
            y_pos = image.shape[0] - 80 + i * 25
            cv2.putText(image, line, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(image, line, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)


class ARController:
    """High-level AR application controller"""
    
    def __init__(self):
        self.ar_engine = AREngine()
        self.is_calibrated = False
        
        # Animation parameters
        self.animation_enabled = True
        self.rotation_speed = 1.0  # degrees per frame
        self.current_rotation = 0.0
        
    def set_camera_calibration(self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray):
        """Set camera calibration parameters"""
        self.ar_engine.set_camera_parameters(camera_matrix, dist_coeffs)
        self.is_calibrated = True
        print("Camera calibration set successfully")
    
    def load_trex_model(self, obj_file_path: str, scale: float = 3.0) -> bool:
        """
        Load T-Rex model with appropriate scaling
        
        Args:
            obj_file_path: Path to T-Rex OBJ file
            scale: Scale factor to make model visible
            
        Returns:
            bool: True if loaded successfully
        """
        success = self.ar_engine.load_model("trex", obj_file_path, scale)
        
        if success:
            # Position T-Rex slightly above the marker
            self.ar_engine.set_model_transform(
                "trex",
                position=(0, 0, -0.05),  # 5cm above marker (higher)
                rotation=(90, 0, 0),     # Stand upright
                scale=3.0                # 3x larger final scale
            )
            print("T-Rex model loaded and positioned")
        
        return success
    
    def toggle_animation(self):
        """Toggle model animation on/off"""
        self.animation_enabled = not self.animation_enabled
        print(f"Animation {'enabled' if self.animation_enabled else 'disabled'}")
    
    def cycle_render_mode(self):
        """Cycle through render modes"""
        modes = ['wireframe', 'solid', 'both']
        current_mode = self.ar_engine.render_mode
        current_idx = modes.index(current_mode)
        next_mode = modes[(current_idx + 1) % len(modes)]
        
        self.ar_engine.set_render_mode(next_mode)
        print(f"Render mode: {next_mode}")
    
    def toggle_axis(self):
        """Toggle coordinate axis display"""
        self.ar_engine.show_axis = not self.ar_engine.show_axis
        print(f"Axis display {'enabled' if self.ar_engine.show_axis else 'disabled'}")
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process frame with animation
        
        Args:
            frame: Input camera frame
            
        Returns:
            Frame with AR content
        """
        # Update animation
        if self.animation_enabled and "trex" in self.ar_engine.models:
            self.current_rotation += self.rotation_speed
            if self.current_rotation >= 360:
                self.current_rotation = 0
            
            # Update T-Rex rotation
            self.ar_engine.set_model_transform(
                "trex",
                position=(0, 0, -0.05),
                rotation=(90, self.current_rotation, 0),
                scale=3.0
            )
        
        return self.ar_engine.process_frame(frame)
