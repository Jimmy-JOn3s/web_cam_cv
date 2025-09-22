"""
Custom Panorama Creator - Implementation without OpenCV panorama functions
"""
import cv2
import numpy as np
from datetime import datetime


class PanoramaCreator:
    """Custom panorama creation using feature matching and homography."""
    
    def __init__(self):
        """Initialize panorama creator."""
        self.images = []
        self.feature_detector = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher()
        
    def add_image(self, image):
        """
        Add an image to the panorama sequence.
        
        Args:
            image (numpy.ndarray): Image to add
            
        Returns:
            bool: True if image was added successfully
        """
        try:
            if image is None or image.size == 0:
                print("Error: Invalid image provided")
                return False
            
            # Limit number of images to prevent memory issues
            if len(self.images) >= 10:
                print("Warning: Maximum number of images (10) reached. Removing oldest image.")
                self.images.pop(0)
            
            self.images.append(image.copy())
            print(f"Added image {len(self.images)} to panorama sequence")
            return True
        except Exception as e:
            print(f"Error adding image: {e}")
            return False
    
    def clear_images(self):
        """Clear all stored images."""
        self.images.clear()
        print("Cleared all images from panorama sequence")
    
    def detect_and_describe(self, image):
        """
        Detect keypoints and compute descriptors.
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            tuple: (keypoints, descriptors)
        """
        try:
            if image is None or image.size == 0:
                return [], None
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            keypoints, descriptors = self.feature_detector.detectAndCompute(gray, None)
            
            # Ensure we have valid results
            if keypoints is None:
                keypoints = []
            if descriptors is None:
                descriptors = np.array([])
            
            return keypoints, descriptors
        except Exception as e:
            print(f"Error in feature detection: {e}")
            return [], None
    
    def match_features(self, desc1, desc2, ratio_threshold=0.75):
        """
        Match features between two images using ratio test.
        
        Args:
            desc1 (numpy.ndarray): Descriptors from first image
            desc2 (numpy.ndarray): Descriptors from second image
            ratio_threshold (float): Ratio threshold for Lowe's test
            
        Returns:
            list: List of good matches
        """
        try:
            if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
                return []
            
            # Ensure descriptors are valid
            if desc1.shape[0] < 2 or desc2.shape[0] < 2:
                return []
            
            # Find matches using KNN
            matches = self.matcher.knnMatch(desc1, desc2, k=2)
            
            # Apply ratio test (Lowe's test)
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < ratio_threshold * n.distance:
                        good_matches.append(m)
            
            return good_matches
        except Exception as e:
            print(f"Error in feature matching: {e}")
            return []
    
    def find_homography_ransac(self, src_pts, dst_pts, max_iterations=1000, threshold=4.0):
        """
        Find homography using RANSAC algorithm.
        
        Args:
            src_pts (numpy.ndarray): Source points
            dst_pts (numpy.ndarray): Destination points
            max_iterations (int): Maximum RANSAC iterations
            threshold (float): Inlier threshold
            
        Returns:
            tuple: (homography_matrix, inlier_mask)
        """
        try:
            if len(src_pts) < 4 or len(dst_pts) < 4:
                return None, None
            
            src_pts = np.array(src_pts, dtype=np.float32)
            dst_pts = np.array(dst_pts, dtype=np.float32)
            
            best_homography = None
            best_inliers = 0
            best_mask = None
            
            # Reduce iterations for stability
            max_iterations = min(max_iterations, 500)
            
            for iteration in range(max_iterations):
                try:
                    # Randomly select 4 points
                    indices = np.random.choice(len(src_pts), 4, replace=False)
                    src_sample = src_pts[indices]
                    dst_sample = dst_pts[indices]
                    
                    # Compute homography from 4 points
                    homography = self.compute_homography_4points(src_sample, dst_sample)
                    
                    if homography is not None and not np.any(np.isnan(homography)) and not np.any(np.isinf(homography)):
                        # Test all points
                        inliers = 0
                        mask = np.zeros(len(src_pts), dtype=bool)
                        
                        for i in range(len(src_pts)):
                            try:
                                # Transform point using homography
                                src_pt = np.array([src_pts[i][0], src_pts[i][1], 1.0])
                                transformed_pt = homography @ src_pt
                                
                                if abs(transformed_pt[2]) > 1e-8:  # Avoid division by zero
                                    transformed_pt /= transformed_pt[2]
                                    
                                    # Calculate distance to destination point
                                    distance = np.sqrt((transformed_pt[0] - dst_pts[i][0])**2 + 
                                                     (transformed_pt[1] - dst_pts[i][1])**2)
                                    
                                    if distance < threshold:
                                        inliers += 1
                                        mask[i] = True
                            except:
                                continue
                        
                        # Update best solution
                        if inliers > best_inliers:
                            best_inliers = inliers
                            best_homography = homography
                            best_mask = mask
                except:
                    continue
            
            # Require minimum number of inliers
            if best_inliers < 8:
                return None, None
            
            return best_homography, best_mask
        except Exception as e:
            print(f"Error in RANSAC homography estimation: {e}")
            return None, None
    
    def compute_homography_4points(self, src_pts, dst_pts):
        """
        Compute homography from 4 point correspondences.
        
        Args:
            src_pts (numpy.ndarray): 4 source points
            dst_pts (numpy.ndarray): 4 destination points
            
        Returns:
            numpy.ndarray: 3x3 homography matrix
        """
        try:
            if len(src_pts) != 4 or len(dst_pts) != 4:
                return None
            
            # Build system of equations Ah = 0
            A = []
            
            for i in range(4):
                x, y = float(src_pts[i][0]), float(src_pts[i][1])
                u, v = float(dst_pts[i][0]), float(dst_pts[i][1])
                
                A.append([-x, -y, -1, 0, 0, 0, u*x, u*y, u])
                A.append([0, 0, 0, -x, -y, -1, v*x, v*y, v])
            
            A = np.array(A, dtype=np.float64)
            
            # Check for degenerate cases
            if np.linalg.matrix_rank(A) < 8:
                return None
            
            # Solve using SVD
            _, _, Vt = np.linalg.svd(A)
            h = Vt[-1]
            
            # Reshape to 3x3 matrix
            homography = h.reshape(3, 3)
            
            # Normalize and validate
            if abs(homography[2, 2]) > 1e-10:
                homography /= homography[2, 2]
            else:
                return None
            
            # Check for valid homography (no NaN or infinite values)
            if np.any(np.isnan(homography)) or np.any(np.isinf(homography)):
                return None
            
            return homography
        except Exception as e:
            print(f"Error computing homography: {e}")
            return None
    
    def warp_image(self, image, homography, output_shape):
        """
        Warp image using homography matrix.
        
        Args:
            image (numpy.ndarray): Input image
            homography (numpy.ndarray): 3x3 homography matrix
            output_shape (tuple): (height, width) of output
            
        Returns:
            numpy.ndarray: Warped image
        """
        try:
            height, width = output_shape
            
            # Limit output size for memory safety
            if height > 8000 or width > 8000:
                print(f"Warning: Output size too large ({width}x{height}), limiting to 8000x8000")
                scale = min(8000/width, 8000/height)
                height = int(height * scale)
                width = int(width * scale)
            
            warped = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Inverse homography for backward mapping
            try:
                inv_homography = np.linalg.inv(homography)
            except np.linalg.LinAlgError:
                print("Error: Homography matrix is singular")
                return warped
            
            # Process in chunks to avoid memory issues
            chunk_size = 1000
            
            for y_start in range(0, height, chunk_size):
                y_end = min(y_start + chunk_size, height)
                
                for x_start in range(0, width, chunk_size):
                    x_end = min(x_start + chunk_size, width)
                    
                    # Create coordinate grids for this chunk
                    y_coords, x_coords = np.mgrid[y_start:y_end, x_start:x_end]
                    ones = np.ones_like(x_coords)
                    
                    # Stack coordinates
                    coords = np.stack([x_coords.ravel(), y_coords.ravel(), ones.ravel()])
                    
                    # Transform coordinates
                    src_coords = inv_homography @ coords
                    
                    # Normalize homogeneous coordinates
                    valid_mask = np.abs(src_coords[2]) > 1e-10
                    if not np.any(valid_mask):
                        continue
                    
                    src_coords[:, valid_mask] /= src_coords[2, valid_mask]
                    
                    # Get integer coordinates
                    chunk_height, chunk_width = y_end - y_start, x_end - x_start
                    src_x = src_coords[0].reshape(chunk_height, chunk_width)
                    src_y = src_coords[1].reshape(chunk_height, chunk_width)
                    
                    # Create masks for valid coordinates
                    valid_x = (src_x >= 0) & (src_x < image.shape[1] - 1)
                    valid_y = (src_y >= 0) & (src_y < image.shape[0] - 1)
                    valid = valid_x & valid_y
                    
                    if not np.any(valid):
                        continue
                    
                    # Simple nearest neighbor interpolation for stability
                    x_round = np.round(src_x).astype(int)
                    y_round = np.round(src_y).astype(int)
                    
                    # Ensure indices are within bounds
                    x_round = np.clip(x_round, 0, image.shape[1] - 1)
                    y_round = np.clip(y_round, 0, image.shape[0] - 1)
                    
                    # Sample pixels
                    valid_indices = np.where(valid)
                    for i, j in zip(valid_indices[0], valid_indices[1]):
                        warped[y_start + i, x_start + j] = image[y_round[i, j], x_round[i, j]]
            
            return warped
            
        except Exception as e:
            print(f"Error in image warping: {e}")
            return np.zeros((output_shape[0], output_shape[1], 3), dtype=np.uint8)
    
    def blend_images(self, img1, img2, mask1, mask2):
        """
        Blend two images using masks.
        
        Args:
            img1 (numpy.ndarray): First image
            img2 (numpy.ndarray): Second image
            mask1 (numpy.ndarray): Mask for first image
            mask2 (numpy.ndarray): Mask for second image
            
        Returns:
            numpy.ndarray: Blended image
        """
        # Create overlap mask
        overlap = mask1 & mask2
        
        # Initialize result
        result = np.zeros_like(img1)
        
        # Copy non-overlapping regions
        result[mask1 & ~overlap] = img1[mask1 & ~overlap]
        result[mask2 & ~overlap] = img2[mask2 & ~overlap]
        
        # Blend overlapping regions (simple averaging)
        if np.any(overlap):
            result[overlap] = (img1[overlap].astype(float) + img2[overlap].astype(float)) / 2
            result[overlap] = result[overlap].astype(np.uint8)
        
        return result
    
    def create_panorama(self):
        """
        Create panorama from stored images.
        
        Returns:
            numpy.ndarray: Panorama image
        """
        try:
            if len(self.images) < 2:
                print("Need at least 2 images to create panorama")
                return None
            
            print(f"Creating panorama from {len(self.images)} images...")
            
            # Start with first image
            panorama = self.images[0].copy()
            
            for i in range(1, len(self.images)):
                print(f"Stitching image {i+1}...")
                
                try:
                    # Detect features in both images
                    kp1, desc1 = self.detect_and_describe(panorama)
                    kp2, desc2 = self.detect_and_describe(self.images[i])
                    
                    if len(kp1) < 10 or len(kp2) < 10:
                        print(f"Not enough features detected in image {i+1}")
                        continue
                    
                    # Match features
                    matches = self.match_features(desc1, desc2)
                    
                    if len(matches) < 10:
                        print(f"Not enough matches found ({len(matches)}), skipping image {i+1}")
                        continue
                    
                    # Extract matching points
                    src_pts = np.array([kp2[m.trainIdx].pt for m in matches], dtype=np.float32)
                    dst_pts = np.array([kp1[m.queryIdx].pt for m in matches], dtype=np.float32)
                    
                    # Find homography
                    homography, mask = self.find_homography_ransac(src_pts, dst_pts)
                    
                    if homography is None:
                        print(f"Could not find homography for image {i+1}")
                        continue
                    
                    # Calculate output canvas size with safety checks
                    h1, w1 = panorama.shape[:2]
                    h2, w2 = self.images[i].shape[:2]
                    
                    # Transform corners of second image
                    corners = np.array([[0, 0, 1], [w2, 0, 1], [w2, h2, 1], [0, h2, 1]], dtype=np.float64).T
                    transformed_corners = homography @ corners
                    
                    # Safely handle homogeneous coordinates
                    for j in range(transformed_corners.shape[1]):
                        if abs(transformed_corners[2, j]) > 1e-10:
                            transformed_corners[:, j] /= transformed_corners[2, j]
                        else:
                            print(f"Invalid transformation for image {i+1}")
                            break
                    else:
                        # Calculate bounding box
                        all_x = np.concatenate([transformed_corners[0], [0, w1, w1, 0]])
                        all_y = np.concatenate([transformed_corners[1], [0, 0, h1, h1]])
                        
                        min_x, max_x = int(np.floor(np.min(all_x))), int(np.ceil(np.max(all_x)))
                        min_y, max_y = int(np.floor(np.min(all_y))), int(np.ceil(np.max(all_y)))
                        
                        # Limit panorama size to prevent memory issues
                        max_size = 8000
                        if (max_x - min_x) > max_size or (max_y - min_y) > max_size:
                            print(f"Panorama too large for image {i+1}, skipping")
                            continue
                        
                        # Create translation matrix to handle negative coordinates
                        translation = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]], dtype=np.float64)
                        
                        # Update homography
                        homography = translation @ homography
                        
                        # Calculate output size
                        output_width = max_x - min_x
                        output_height = max_y - min_y
                        
                        # Warp second image
                        warped_img2 = self.warp_image(self.images[i], homography, (output_height, output_width))
                        
                        if warped_img2 is None:
                            print(f"Failed to warp image {i+1}")
                            continue
                        
                        # Create new panorama canvas
                        new_panorama = np.zeros((output_height, output_width, 3), dtype=np.uint8)
                        
                        # Place first image (panorama) with translation
                        x_offset, y_offset = -min_x, -min_y
                        
                        # Ensure we don't exceed array bounds
                        end_y = min(y_offset + h1, output_height)
                        end_x = min(x_offset + w1, output_width)
                        
                        if x_offset >= 0 and y_offset >= 0 and end_x > x_offset and end_y > y_offset:
                            new_panorama[y_offset:end_y, x_offset:end_x] = panorama[:end_y-y_offset, :end_x-x_offset]
                            
                            # Create masks
                            mask1 = np.zeros((output_height, output_width), dtype=bool)
                            mask1[y_offset:end_y, x_offset:end_x] = True
                            
                            mask2 = np.any(warped_img2 > 0, axis=2)
                            
                            # Blend images
                            panorama = self.blend_images(new_panorama, warped_img2, mask1, mask2)
                            
                            print(f"Successfully stitched image {i+1}")
                        else:
                            print(f"Invalid offset for image {i+1}")
                            continue
                    
                except Exception as e:
                    print(f"Error processing image {i+1}: {e}")
                    continue
            
            return panorama
            
        except Exception as e:
            print(f"Error creating panorama: {e}")
            return None
    
    def save_panorama(self, panorama, filename=None):
        """
        Save panorama to file.
        
        Args:
            panorama (numpy.ndarray): Panorama image
            filename (str): Output filename
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"panorama_{timestamp}.jpg"
        
        if panorama is not None:
            cv2.imwrite(filename, panorama)
            print(f"Panorama saved as {filename}")
        else:
            print("No panorama to save")
    
    def get_status(self):
        """Get current status of panorama creator."""
        return {
            'num_images': len(self.images),
            'ready_to_create': len(self.images) >= 2
        }
