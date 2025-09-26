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
        self.matcher = cv2.BFMatcher()
        
        # Lab 4 style stitching parameters
        self.stitching_params = {
            'ratio_threshold': 0.75,       # Lab 4 uses 0.75 for Lowe's ratio test
            'ransac_threshold': 5.0,       # RANSAC threshold for homography
            'min_matches': 10,             # Minimum good matches required
            'min_inliers': 8,              # Minimum inliers for valid homography
            'sift_features': 1000,         # Number of SIFT features to detect
            'blend_mode': 0,               # Simple overlay (Lab 4 style)
            'max_panorama_width': 4000,    # Maximum panorama width
            'max_panorama_height': 2000    # Maximum panorama height
        }

        self.update_sift_detector()
    
    def update_sift_detector(self):
        """Update SIFT detector with current parameters."""
        self.feature_detector = cv2.SIFT_create(nfeatures=self.stitching_params['sift_features'])
    
    def update_stitching_params(self, **kwargs):
        """Update stitching parameters."""
        for key, value in kwargs.items():
            if key in self.stitching_params:
                self.stitching_params[key] = value
        
        # Update SIFT detector if features count changed
        if 'sift_features' in kwargs:
            self.update_sift_detector()
    
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
            
            # Limit number of images to prevent memory issues and stitching complexity
            if len(self.images) >= 3:
                print("Warning: Maximum number of images (3) reached for optimal stitching.")
                print("Creating panorama from current images and starting fresh...")
                # Auto-create panorama and reset for new sequence
                panorama = self.create_panorama()
                if panorama is not None:
                    # Save the current panorama
                    self.save_panorama(panorama)
                    # Start fresh with the new image
                    self.images.clear()
                else:
                    # If panorama creation fails, just remove oldest image
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
    
    def calculate_homography_lab4_style(self, img1, img2):
        """
        Calculate homography between two images using Lab 4 methodology.
        
        Args:
            img1 (numpy.ndarray): Reference image (panorama)
            img2 (numpy.ndarray): Image to warp to img1
            
        Returns:
            numpy.ndarray: 3x3 homography matrix or None
        """
        try:
            # Convert to grayscale
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2
            
            # Detect keypoints and compute descriptors
            kp1, des1 = self.feature_detector.detectAndCompute(gray1, None)
            kp2, des2 = self.feature_detector.detectAndCompute(gray2, None)
            
            if des1 is None or des2 is None or len(des1) < 4 or len(des2) < 4:
                print("Not enough features detected")
                return None
            
            # Match descriptors using BFMatcher with KNN
            matches = self.matcher.knnMatch(des1, des2, k=2)
            
            # Apply Lowe's ratio test (Lab 4 style)
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < self.stitching_params['ratio_threshold'] * n.distance:
                        good_matches.append(m)
            
            if len(good_matches) < self.stitching_params['min_matches']:
                print(f"Not enough good matches: {len(good_matches)}")
                return None
            
            # Extract matching points (Lab 4 style)
            src_pts = np.zeros((len(good_matches), 2), dtype=np.float32)
            dst_pts = np.zeros((len(good_matches), 2), dtype=np.float32)
            
            for i, match in enumerate(good_matches):
                dst_pts[i, :] = kp1[match.queryIdx].pt  # panorama points
                src_pts[i, :] = kp2[match.trainIdx].pt  # new image points
            
            # Find homography using RANSAC (Lab 4 style)
            homography, mask = cv2.findHomography(src_pts, dst_pts, 
                                                cv2.RANSAC, 
                                                self.stitching_params['ransac_threshold'])
            
            if homography is None:
                print("Homography calculation failed")
                return None
            
            # Count inliers
            inliers = np.sum(mask) if mask is not None else 0
            if inliers < self.stitching_params['min_inliers']:
                print(f"Not enough inliers: {inliers}")
                return None
            
            print(f"Found {len(good_matches)} matches, {inliers} inliers")
            return homography
            
        except Exception as e:
            print(f"Error in homography calculation: {e}")
            return None
    
    def crop_to_content(self, image):
        """
        Crop image to remove large black borders.
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            numpy.ndarray: Cropped image
        """
        try:
            if image is None or image.size == 0:
                return image
            
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Find non-zero pixels (content)
            coords = cv2.findNonZero(gray)
            if coords is not None:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(coords)
                
                # Add small margin
                margin = 10
                x = max(0, x - margin)
                y = max(0, y - margin)
                w = min(image.shape[1] - x, w + 2 * margin)
                h = min(image.shape[0] - y, h + 2 * margin)
                
                # Crop the image
                return image[y:y+h, x:x+w]
            else:
                return image
                
        except Exception as e:
            print(f"Error in content cropping: {e}")
            return image
    
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
    
    def match_features(self, desc1, desc2, ratio_threshold=None):
        """
        Match features between two images using ratio test.
        
        Args:
            desc1 (numpy.ndarray): Descriptors from first image
            desc2 (numpy.ndarray): Descriptors from second image
            ratio_threshold (float): Ratio threshold for Lowe's test (uses default if None)
            
        Returns:
            list: List of good matches
        """
        if ratio_threshold is None:
            ratio_threshold = self.stitching_params['ratio_threshold']
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
    
    def find_homography_ransac(self, src_pts, dst_pts, max_iterations=1000, threshold=None):
        """
        Find homography using RANSAC algorithm.
        
        Args:
            src_pts (numpy.ndarray): Source points
            dst_pts (numpy.ndarray): Destination points
            max_iterations (int): Maximum RANSAC iterations
            threshold (float): Inlier threshold (uses default if None)
            
        Returns:
            tuple: (homography_matrix, inlier_mask)
        """
        if threshold is None:
            threshold = self.stitching_params['ransac_threshold']
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
            if best_inliers < self.stitching_params['min_inliers']:
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
        Custom panroma creation with image stitching.
        
        Returns:
            numpy.ndarray: Panorama image
        """
        try:
            if len(self.images) < 2:
                print("Need at least 2 images to create panorama")
                return None
            
            print(f"Creating panorama from {len(self.images)} images using exact Lab 4 method...")
            
            # Convert images to the order: image1, image2, image3, image4...
            images = [img.copy() for img in self.images]
            
            # Lab 4 Step 1: Convert to grayscale for feature detection
            gray_images = []
            for img in images:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray_images.append(gray)
            
            # Lab 4 Step 2: Initialize SIFT detector
            sift = cv2.SIFT_create()
            
            # Lab 4 Step 3: Detect keypoints and compute descriptors
            keypoints = []
            descriptors = []
            for gray in gray_images:
                kp, des = sift.detectAndCompute(gray, None)
                keypoints.append(kp)
                descriptors.append(des)
            
            # Lab 4 Step 4: Initialize brute force matcher
            bf = cv2.BFMatcher()
            
            # Lab 4 Step 5: Match descriptors between adjacent images
            def match_adjacent_images(des1, des2):
                matches = bf.knnMatch(des1, des2, k=2)
                good_matches = []
                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < 0.75 * n.distance:  # Lab 4 ratio test
                            good_matches.append([m])
                return good_matches
            
            # Calculate all homographies between adjacent images
            homographies = []
            for i in range(len(images) - 1):
                good_matches = match_adjacent_images(descriptors[i], descriptors[i + 1])
                
                if len(good_matches) < 10:
                    print(f"Not enough matches between image {i+1} and {i+2}")
                    return None
                
                # Lab 4 ransac_transform function
                src_pts = np.zeros((len(good_matches), 2), dtype=np.float32)
                dst_pts = np.zeros((len(good_matches), 2), dtype=np.float32)
                
                for j, match in enumerate(good_matches):
                    dst_pts[j, :] = keypoints[i][match[0].queryIdx].pt
                    src_pts[j, :] = keypoints[i + 1][match[0].trainIdx].pt
                
                M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
                homographies.append(M)
                print(f"Computed homography {i+1}->{i+2}")
            
            # Lab 4 Step 6: Sequential stitching (reverse order like Lab 4)
            # Start from the last image and work backwards
            panorama = images[-1].copy()
            
            # Stitch images in reverse order (Lab 4 approach)
            for i in range(len(images) - 2, -1, -1):
                print(f"Stitching image {i+1} to panorama...")
                
                current_img = images[i]
                M = homographies[i]
                
                if M is None:
                    print(f"No homography for image {i+1}, skipping")
                    continue
                
                # Lab 4 canvas size calculation
                h, w = current_img.shape[:2]
                pano_width = w + panorama.shape[1]
                pano_height = max(h, panorama.shape[0])
                
                print(f"Canvas size: {pano_width}x{pano_height}")
                
                # Lab 4 warp and stitch
                # Warp panorama to current image coordinate system
                warped_pano = cv2.warpPerspective(panorama, M, (pano_width, pano_height))
                
                # Create new panorama canvas
                new_panorama = np.zeros((pano_height, pano_width, 3), dtype=np.uint8)
                
                # Place current image at origin (Lab 4 style)
                new_panorama[:h, :w] = current_img
                
                # Overlay warped panorama
                mask = np.any(warped_pano > 0, axis=2)
                new_panorama[mask] = warped_pano[mask]
                
                panorama = new_panorama
                print(f"Successfully stitched image {i+1}")
            
            # Crop to remove black borders
            panorama = self.crop_to_content(panorama)
            print(f"Final panorama size: {panorama.shape[1]}x{panorama.shape[0]} pixels")
            
            return panorama
            
        except Exception as e:
            print(f"Error creating panorama: {e}")
            return None
    
    def save_panorama(self, panorama, filename=None):
        """
        Save panorama to file in dedicated panorama folder.
        
        Args:
            panorama (numpy.ndarray): Panorama image
            filename (str): Output filename
        """
        if panorama is None:
            print("No panorama to save")
            return
        
        # Create panorama directory if it doesn't exist
        import os
        panorama_dir = "panorama"
        if not os.path.exists(panorama_dir):
            os.makedirs(panorama_dir)
            print(f"Created directory: {panorama_dir}")
        
        # Generate filename with timestamp if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"panorama_{timestamp}.jpg"
        
        # Full path including panorama directory
        full_path = os.path.join(panorama_dir, filename)
        
        # Save panorama
        cv2.imwrite(full_path, panorama)
        print(f"Panorama saved as {full_path}")
        
        # Also save a thumbnail for quick preview
        thumbnail_height = 200
        h, w = panorama.shape[:2]
        thumbnail_width = int((thumbnail_height * w) / h)
        thumbnail = cv2.resize(panorama, (thumbnail_width, thumbnail_height))
        
        thumbnail_name = f"thumb_{filename}"
        thumbnail_path = os.path.join(panorama_dir, thumbnail_name)
        cv2.imwrite(thumbnail_path, thumbnail)
        print(f"Thumbnail saved as {thumbnail_path}")
    
    def get_status(self):
        """Get current status of panorama creator."""
        return {
            'num_images': len(self.images),
            'ready_to_create': len(self.images) >= 2
        }
