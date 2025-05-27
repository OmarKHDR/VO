import cv2
import numpy as np
import sys
from pathlib import Path
import os
import json

def visual_odometry(cap=None):
    """
    Simple Visual Odometry function that integrates with your existing main.py structure
    """
    
    # Get video file path from user
    video_path = input("Enter path to your video file: ").strip() or "test.mp4"
    
    if not os.path.exists(video_path):
        print(f"[✗] Video file not found: {video_path}")
        return
    
    # Try to load camera calibration
    calibration_path = "calibration/camera_calibration.npz"
    camera_matrix = None
    dist_coeffs = None
    
    try:
        data = np.load(calibration_path)
        camera_matrix = data['camera_matrix']
        dist_coeffs = data['dist_coeffs']
        print(f"[✓] Using camera calibration from {calibration_path}")
    except:
        print("[!] No calibration found, using default parameters")
        print("    For better results, run calibration first")
    
    # Load scale factor from calibration params
    try:
        with open("calibration_params.json") as f:
            params = json.load(f)
            scale_factor = params.get("vo_scale_factor", 0.1)
            print(f"[✓] Using scale factor: {scale_factor}")
    except:
        scale_factor = 0.1
        print(f"[!] Could not load scale factor, using default: {scale_factor}")
    
    # Create and run simple visual odometry
    vo = SimpleVisualOdometry(camera_matrix, dist_coeffs, scale_factor)
    vo.process_video(video_path, show_matches=True)

class SimpleVisualOdometry:
    """Simple Visual Odometry with drift reduction"""
    
    def __init__(self, camera_matrix=None, dist_coeffs=None, scale_factor=0.1):
        # Default camera parameters if none provided
        if camera_matrix is None:
            self.camera_matrix = np.array([
                [800, 0, 320],
                [0, 800, 240], 
                [0, 0, 1]
            ], dtype=np.float32)
            self.dist_coeffs = np.zeros((4, 1))
        else:
            self.camera_matrix = camera_matrix
            self.dist_coeffs = dist_coeffs
        
        # Manual scale factor
        self.scale_factor = scale_factor
        print(f"[✓] Initialized with scale factor: {self.scale_factor}")
        
        # Feature detection - more features for better tracking
        self.detector = cv2.ORB_create(nfeatures=1000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Simple pose tracking
        self.position = np.array([0.0, 0.0, 0.0])  # x, y, z position
        self.orientation = np.eye(3)  # rotation matrix
        
        # Path tracking
        self.path = [[0.0, 0.0, 0.0]]  # Start at origin
        self.total_distance = 0.0
        
        # Drift reduction filters
        self.motion_history = []
        self.motion_history_size = 5
        self.max_translation_per_frame = 0.5  # Maximum meters per frame
        self.max_rotation_per_frame = 0.2     # Maximum radians per frame
        
        # Visualization
        self.trajectory_img = np.zeros((600, 600, 3), dtype=np.uint8)
        
        # Previous frame data
        self.prev_gray = None
        self.prev_kp = None
        self.prev_desc = None
        self.frame_count = 0
    
    def filter_motion(self, R, t):
        """Filter motion to reduce drift and noise"""
        # Calculate rotation angle
        rotation_angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
        
        # Calculate translation magnitude
        translation_magnitude = np.linalg.norm(t)
        
        # Reject unrealistic motions
        if rotation_angle > self.max_rotation_per_frame:
            return None  # Skip this frame - too much rotation
        
        if translation_magnitude > self.max_translation_per_frame:
            # Scale down large translations
            t = t * (self.max_translation_per_frame / translation_magnitude)
            translation_magnitude = self.max_translation_per_frame
        
        # Store motion in history for smoothing
        motion = {'R': R, 't': t, 'rot_angle': rotation_angle, 'trans_mag': translation_magnitude}
        self.motion_history.append(motion)
        
        if len(self.motion_history) > self.motion_history_size:
            self.motion_history.pop(0)
        
        # Apply temporal smoothing
        if len(self.motion_history) >= 3:
            # Average recent rotations (simplified)
            avg_rot_angle = np.mean([m['rot_angle'] for m in self.motion_history[-3:]])
            avg_trans_mag = np.mean([m['trans_mag'] for m in self.motion_history[-3:]])
            
            # If current motion is very different from recent average, reduce it
            if rotation_angle > avg_rot_angle * 2:
                # Reduce rotation
                axis_angle = rotation_angle * (R - R.T) / 2
                reduced_angle = min(rotation_angle, avg_rot_angle * 1.5)
                scale_factor = reduced_angle / rotation_angle if rotation_angle > 0 else 1
                R = np.eye(3) + scale_factor * axis_angle
            
            if translation_magnitude > avg_trans_mag * 2:
                # Reduce translation
                t = t * min(1.0, avg_trans_mag * 1.5 / translation_magnitude)
        
        return R, t
    
    def process_video(self, video_path, show_matches=False):
        """Main video processing function"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"[✗] Could not open video: {video_path}")
            return
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"[✓] Processing video: {total_frames} frames at {fps:.1f} FPS")
        print("Press 'q' to quit, 's' to save path image")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process current frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            kp, desc = self.detector.detectAndCompute(gray, None)
            
            if self.prev_gray is not None and desc is not None and self.prev_desc is not None:
                # Match features
                matches = self.matcher.match(self.prev_desc, desc)
                
                if len(matches) > 15:  # Need minimum matches
                    # Sort matches by distance (quality)
                    matches = sorted(matches, key=lambda x: x.distance)
                    
                    # Use only very good matches - more selective
                    num_good = min(80, len(matches) // 3)  # Top 1/3 of matches
                    good_matches = matches[:num_good]
                    
                    if len(good_matches) >= 10:  # Minimum for essential matrix
                        # Extract matched points
                        pts1 = np.float32([self.prev_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                        pts2 = np.float32([kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                        
                        # Estimate motion using essential matrix with stricter RANSAC
                        E, mask = cv2.findEssentialMat(pts1, pts2, self.camera_matrix, 
                                                      method=cv2.RANSAC, prob=0.9999, threshold=0.5)
                        
                        if E is not None and mask is not None:
                            # Only use inlier points
                            inlier_pts1 = pts1[mask.ravel() == 1]
                            inlier_pts2 = pts2[mask.ravel() == 1]
                            
                            if len(inlier_pts1) >= 8:
                                # Recover pose
                                _, R, t, _ = cv2.recoverPose(E, inlier_pts1, inlier_pts2, self.camera_matrix)
                                
                                # Apply motion filtering to reduce drift
                                filtered_motion = self.filter_motion(R, t)
                                
                                if filtered_motion is not None:
                                    R_filtered, t_filtered = filtered_motion
                                    
                                    # Apply filtered motion with manual scale
                                    translation = self.scale_factor * t_filtered.flatten()
                                    
                                    # Transform translation to world coordinates
                                    world_translation = self.orientation @ translation
                                    
                                    # Update position
                                    self.position += world_translation
                                    
                                    # Update orientation (with smoothing)
                                    self.orientation = R_filtered @ self.orientation
                                    
                                    # Add to path
                                    self.path.append(self.position.copy().tolist())
                                    
                                    # Calculate distance moved
                                    if len(self.path) > 1:
                                        prev_pos = np.array(self.path[-2])
                                        curr_pos = np.array(self.path[-1])
                                        distance = np.linalg.norm(curr_pos - prev_pos)
                                        self.total_distance += distance
                        
                        # Show matches if requested
                        if show_matches:
                            match_img = cv2.drawMatches(
                                self.prev_gray, self.prev_kp, gray, kp, 
                                good_matches[:20], None, 
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                            )
                            cv2.imshow('Feature Matches', cv2.resize(match_img, (1200, 400)))
            
            # Update previous frame
            self.prev_gray = gray.copy()
            self.prev_kp = kp
            self.prev_desc = desc
            self.frame_count += 1
            
            # Draw info on frame
            self.draw_info(frame)
            
            # Update trajectory visualization
            self.update_trajectory()
            
            # Show results
            cv2.imshow('Simple Visual Odometry', cv2.resize(frame, (800, 600)))
            cv2.imshow('Trajectory', self.trajectory_img)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_path_image()
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Save final results
        self.save_path_image()
        self.print_results()
        self.save_path_csv()
    
    def draw_info(self, frame):
        """Draw information on the frame"""
        # Add semi-transparent background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (400, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw text information
        cv2.putText(frame, f"Distance: {self.total_distance:.2f}m", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Position: ({self.position[0]:.1f}, {self.position[1]:.1f}, {self.position[2]:.1f})", 
                   (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Frame: {self.frame_count}", (10, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Scale: {self.scale_factor}", (10, 105), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    def update_trajectory(self):
        """Update trajectory visualization"""
        # Clear trajectory image periodically to prevent clutter
        if self.frame_count % 200 == 0:
            self.trajectory_img.fill(0)
        
        # Draw coordinate axes
        center = (300, 300)
        cv2.line(self.trajectory_img, (center[0], 0), (center[0], 600), (50, 50, 50), 1)  # Y axis
        cv2.line(self.trajectory_img, (0, center[1]), (600, center[1]), (50, 50, 50), 1)  # X axis
        
        # Draw trajectory path
        if len(self.path) > 1:
            scale = 50  # Scale for visualization
            for i in range(1, len(self.path)):
                pt1 = self.path[i-1]
                pt2 = self.path[i]
                
                # Convert to image coordinates (top-down view)
                x1 = int(pt1[0] * scale) + center[0]
                y1 = int(-pt1[2] * scale) + center[1]  # -Z for forward direction
                x2 = int(pt2[0] * scale) + center[0]
                y2 = int(-pt2[2] * scale) + center[1]
                
                # Keep points within image bounds
                if (0 <= x1 < 600 and 0 <= y1 < 600 and 
                    0 <= x2 < 600 and 0 <= y2 < 600):
                    cv2.line(self.trajectory_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw current position
        if self.path:
            curr_pos = self.path[-1]
            x = int(curr_pos[0] * 50) + center[0]
            y = int(-curr_pos[2] * 50) + center[1]
            if 0 <= x < 600 and 0 <= y < 600:
                cv2.circle(self.trajectory_img, (x, y), 5, (0, 0, 255), -1)
        
        # Draw start position
        cv2.circle(self.trajectory_img, center, 8, (255, 255, 0), 2)  # Yellow circle for start
    
    def save_path_image(self):
        """Save a clean path visualization"""
        if len(self.path) < 2:
            print("[!] Not enough path data to save")
            return
        
        # Create output directory
        os.makedirs("vo_output", exist_ok=True)
        
        # Create high-resolution image
        img_size = 1000
        path_img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
        
        # Convert path to numpy array
        path_array = np.array(self.path)
        
        # Calculate bounds with padding
        min_x, max_x = path_array[:, 0].min(), path_array[:, 0].max()
        min_z, max_z = path_array[:, 2].min(), path_array[:, 2].max()
        
        # Add padding
        padding = max(2.0, max(max_x - min_x, max_z - min_z) * 0.1)
        min_x -= padding
        max_x += padding
        min_z -= padding
        max_z += padding
        
        # Calculate scale to fit image
        range_x = max_x - min_x
        range_z = max_z - min_z
        scale = min((img_size - 100) / range_x, (img_size - 100) / range_z) if range_x > 0 and range_z > 0 else 50
        
        def world_to_image(pos):
            x = int((pos[0] - min_x) * scale + 50)
            y = int((pos[2] - min_z) * scale + 50)
            return (x, y)
        
        # Draw grid
        grid_spacing = 50
        for i in range(0, img_size, grid_spacing):
            cv2.line(path_img, (i, 0), (i, img_size), (230, 230, 230), 1)
            cv2.line(path_img, (0, i), (img_size, i), (230, 230, 230), 1)
        
        # Draw path
        for i in range(1, len(self.path)):
            pt1 = world_to_image(self.path[i-1])
            pt2 = world_to_image(self.path[i])
            cv2.line(path_img, pt1, pt2, (0, 100, 255), 3)
        
        # Mark start and end
        start_pt = world_to_image(self.path[0])
        end_pt = world_to_image(self.path[-1])
        
        cv2.circle(path_img, start_pt, 10, (0, 255, 0), -1)  # Green start
        cv2.circle(path_img, end_pt, 10, (0, 0, 255), -1)    # Red end
        
        cv2.putText(path_img, "START", (start_pt[0] + 15, start_pt[1]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 150, 0), 2)
        cv2.putText(path_img, "END", (end_pt[0] + 15, end_pt[1]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 0, 0), 2)
        
        # Add information panel
        info_panel = np.zeros((200, 400, 3), dtype=np.uint8)
        info_text = [
            f"Simple Visual Odometry Results",
            f"Scale Factor: {self.scale_factor}",
            f"Total Distance: {self.total_distance:.2f}m",
            f"Total Frames: {self.frame_count}",
            f"Path Points: {len(self.path)}",
            f"Final Position: ({self.position[0]:.2f}, {self.position[1]:.2f}, {self.position[2]:.2f})"
        ]
        
        for i, text in enumerate(info_text):
            cv2.putText(info_panel, text, (10, 25 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Combine images
        final_img = np.zeros((img_size + 200, img_size, 3), dtype=np.uint8)
        final_img[:img_size, :] = path_img
        final_img[img_size:img_size+200, :400] = info_panel
        
        # Save images
        cv2.imwrite("vo_output/trajectory_path.png", final_img)
        cv2.imwrite("vo_output/trajectory_path_small.png", cv2.resize(final_img, (500, 600)))
        
        print(f"[✓] Path images saved to vo_output/")
    
    def save_path_csv(self):
        """Save path data to CSV file"""
        os.makedirs("vo_output", exist_ok=True)
        
        with open("vo_output/path.csv", 'w') as f:
            f.write("frame,x,y,z\n")
            for i, pos in enumerate(self.path):
                f.write(f"{i},{pos[0]:.6f},{pos[1]:.6f},{pos[2]:.6f}\n")
        
        print("[✓] Path data saved to vo_output/path.csv")
    
    def print_results(self):
        """Print final results"""
        end_pos = self.path[-1] if self.path else [0, 0, 0]
        drift = np.linalg.norm(np.array(end_pos))
        
        print(f"\n=== SIMPLE VISUAL ODOMETRY RESULTS ===")
        print(f"Scale factor used: {self.scale_factor}")
        print(f"Total frames processed: {self.frame_count}")
        print(f"Total distance traveled: {self.total_distance:.2f}m")
        print(f"Final position: ({end_pos[0]:.2f}, {end_pos[1]:.2f}, {end_pos[2]:.2f})")
        print(f"Distance from start: {drift:.2f}m")
        
        # Save results
        with open("vo_output/results.txt", 'w') as f:
            f.write("Simple Visual Odometry Results\n")
            f.write("=" * 30 + "\n")
            f.write(f"Scale factor: {self.scale_factor}\n")
            f.write(f"Total frames: {self.frame_count}\n")
            f.write(f"Total distance: {self.total_distance:.2f}m\n")
            f.write(f"Final position: ({end_pos[0]:.2f}, {end_pos[1]:.2f}, {end_pos[2]:.2f})\n")
            f.write(f"Distance from start: {drift:.2f}m\n")
        
        print("[✓] Results saved to vo_output/results.txt")
        print("=" * 40)