import cv2
import numpy as np
import sys
from pathlib import Path
import os

def visual_odometry(cap=None):
    """
    Visual Odometry function that integrates with your existing main.py structure
    """
    
    # Get video file path from user
    video_path = input("Enter path to your video file: ").strip()
    
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
    
    # Create and run visual odometry
    vo = VideoVisualOdometry(camera_matrix, dist_coeffs)
    vo.process_video(video_path, show_matches=True)

class VideoVisualOdometry:
    """Simple Visual Odometry with path image output"""
    
    def __init__(self, camera_matrix=None, dist_coeffs=None):
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
        
        # Feature detection
        self.detector = cv2.ORB_create(nfeatures=2000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Pose tracking
        self.R = np.eye(3, dtype=np.float64)
        self.t = np.zeros((3, 1), dtype=np.float64)
        
        # Path tracking
        self.path = []
        self.start_pos = None
        self.total_distance = 0.0
        
        # Visualization
        self.trajectory_img = np.zeros((600, 600, 3), dtype=np.uint8)
        
        # Previous frame data
        self.prev_gray = None
        self.prev_kp = None
        self.prev_desc = None
        self.frame_count = 0
    
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
                matches = sorted(matches, key=lambda x: x.distance)
                
                if len(matches) > 30:
                    # Get good matches
                    good_matches = matches[:len(matches)//2]
                    
                    # Extract points
                    pts1 = np.float32([self.prev_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    pts2 = np.float32([kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    
                    # Estimate motion
                    if len(pts1) >= 8:
                        E, mask = cv2.findEssentialMat(pts1, pts2, self.camera_matrix, method=cv2.RANSAC)
                        
                        if E is not None:
                            _, R, t, _ = cv2.recoverPose(E, pts1, pts2, self.camera_matrix)
                            
                            # Update pose with scale
                            scale = 0.1  # Simple fixed scale
                            self.t = self.t + scale * (self.R @ t)
                            self.R = R @ self.R
                            
                            # Store current position
                            pos = [self.t[0,0], self.t[1,0], self.t[2,0]]
                            self.path.append(pos)
                            
                            # Set start position
                            if self.start_pos is None:
                                self.start_pos = pos.copy()
                            
                            # Calculate distance
                            if len(self.path) > 1:
                                prev_pos = self.path[-2]
                                dist = np.linalg.norm(np.array(pos) - np.array(prev_pos))
                                self.total_distance += dist
                    
                    # Show matches if requested
                    if show_matches:
                        match_img = cv2.drawMatches(
                            self.prev_gray, self.prev_kp, gray, kp, 
                            good_matches[:50], None, 
                            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                        )
                        cv2.imshow('Matches', cv2.resize(match_img, (1200, 400)))
            
            # Update previous frame
            self.prev_gray = gray.copy()
            self.prev_kp = kp
            self.prev_desc = desc
            self.frame_count += 1
            
            # Draw trajectory
            self.draw_info(frame)
            
            # Show result
            cv2.imshow('Visual Odometry', cv2.resize(frame, (800, 600)))
            cv2.imshow('Trajectory', self.trajectory_img)
            
            # Handle input
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
    
    def draw_info(self, frame):
        """Draw trajectory and information"""
        # Clear trajectory image periodically
        if self.frame_count % 100 == 0:
            self.trajectory_img.fill(0)
        
        # Draw trajectory
        if len(self.path) > 1:
            for i in range(1, len(self.path)):
                pt1 = self.path[i-1]
                pt2 = self.path[i]
                
                x1 = int(-pt1[0] * 20) + 300
                y1 = int(-pt1[2] * 20) + 300
                x2 = int(-pt2[0] * 20) + 300
                y2 = int(-pt2[2] * 20) + 300
                
                cv2.line(self.trajectory_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Current position
        if self.path:
            x = int(-self.t[0,0] * 20) + 300
            y = int(-self.t[2,0] * 20) + 300
            cv2.circle(self.trajectory_img, (x, y), 3, (0, 0, 255), -1)
        
        # Add info text
        cv2.putText(frame, f"Distance: {self.total_distance:.2f}m", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Position: ({self.t[0,0]:.1f}, {self.t[1,0]:.1f}, {self.t[2,0]:.1f})", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Frame: {self.frame_count}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    def save_path_image(self):
        """Save path as a clean image from top-down view"""
        if not self.path:
            print("[!] No path data to save")
            return
        
        # Create output directory
        os.makedirs("vo_output", exist_ok=True)
        
        # Convert path to numpy array for easier manipulation
        path_array = np.array(self.path)
        
        # Get path bounds
        min_x, max_x = path_array[:, 0].min(), path_array[:, 0].max()
        min_z, max_z = path_array[:, 2].min(), path_array[:, 2].max()
        
        # Add padding
        padding = 2.0
        min_x -= padding
        max_x += padding
        min_z -= padding
        max_z += padding
        
        # Create high resolution image
        img_size = 1000
        path_img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255  # White background
        
        # Scale factor to fit path in image
        scale_x = (img_size - 100) / (max_x - min_x) if max_x != min_x else 50
        scale_z = (img_size - 100) / (max_z - min_z) if max_z != min_z else 50
        scale = min(scale_x, scale_z)
        
        # Convert path coordinates to image coordinates
        def world_to_image(x, z):
            img_x = int((x - min_x) * scale + 50)
            img_y = int((z - min_z) * scale + 50)
            return img_x, img_y
        
        # Draw path
        if len(self.path) > 1:
            for i in range(1, len(self.path)):
                pt1 = self.path[i-1]
                pt2 = self.path[i]
                
                img_pt1 = world_to_image(pt1[0], pt1[2])
                img_pt2 = world_to_image(pt2[0], pt2[2])
                
                # Draw line with varying thickness based on speed
                cv2.line(path_img, img_pt1, img_pt2, (0, 100, 255), 3)  # Orange path
        
        # Mark start position
        if self.start_pos:
            start_img = world_to_image(self.start_pos[0], self.start_pos[2])
            cv2.circle(path_img, start_img, 8, (0, 255, 0), -1)  # Green start
            cv2.putText(path_img, "START", (start_img[0] + 15, start_img[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 150, 0), 2)
        
        # Mark end position
        if self.path:
            end_pos = self.path[-1]
            end_img = world_to_image(end_pos[0], end_pos[2])
            cv2.circle(path_img, end_img, 8, (0, 0, 255), -1)  # Red end
            cv2.putText(path_img, "END", (end_img[0] + 15, end_img[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 0, 0), 2)
        
        # Add grid lines for reference
        grid_spacing = 50
        for i in range(0, img_size, grid_spacing):
            cv2.line(path_img, (i, 0), (i, img_size), (220, 220, 220), 1)
            cv2.line(path_img, (0, i), (img_size, i), (220, 220, 220), 1)
        
        # Add information text
        info_text = [
            f"Total Distance: {self.total_distance:.2f}m",
            f"Total Frames: {self.frame_count}",
            f"Path Points: {len(self.path)}",
            f"X Range: {min_x+padding:.1f} to {max_x-padding:.1f}m",
            f"Z Range: {min_z+padding:.1f} to {max_z-padding:.1f}m"
        ]
        
        # Add black background for text
        cv2.rectangle(path_img, (10, 10), (350, 150), (0, 0, 0), -1)
        cv2.rectangle(path_img, (10, 10), (350, 150), (255, 255, 255), 2)
        
        for i, text in enumerate(info_text):
            cv2.putText(path_img, text, (20, 35 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add compass
        compass_center = (img_size - 80, 80)
        cv2.circle(path_img, compass_center, 30, (0, 0, 0), 2)
        # North arrow
        cv2.arrowedLine(path_img, compass_center, 
                       (compass_center[0], compass_center[1] - 25), 
                       (0, 0, 255), 2, tipLength=0.3)
        cv2.putText(path_img, "N", (compass_center[0] - 5, compass_center[1] - 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Save the image
        image_path = "vo_output/trajectory_path.png"
        cv2.imwrite(image_path, path_img)
        print(f"[✓] Path image saved to {os.path.abspath(image_path)}")
        
        # Also save a smaller version for quick viewing
        small_img = cv2.resize(path_img, (500, 500))
        small_path = "vo_output/trajectory_path_small.png"
        cv2.imwrite(small_path, small_img)
        print(f"[✓] Small path image saved to {os.path.abspath(small_path)}")
    
    def calculate_drift_error(self):
        """Calculate drift error"""
        if not self.path or not self.start_pos:
            return 0.0
        
        # Simple drift: distance from start to end
        end_pos = self.path[-1]
        drift = np.linalg.norm(np.array(end_pos) - np.array(self.start_pos))
        return drift
    
    def print_results(self):
        """Print final results"""
        drift_error = self.calculate_drift_error()
        
        print(f"\n=== RESULTS ===")
        print(f"Total frames: {self.frame_count}")
        print(f"Total distance: {self.total_distance:.2f}m")
        
        if self.path:
            print(f"Start position: ({self.start_pos[0]:.2f}, {self.start_pos[1]:.2f}, {self.start_pos[2]:.2f})")
            print(f"End position: ({self.path[-1][0]:.2f}, {self.path[-1][1]:.2f}, {self.path[-1][2]:.2f})")
            print(f"Drift error: {drift_error:.2f}m")
            
            # Save results to text file
            with open("vo_output/results.txt", 'w') as f:
                f.write(f"Total frames: {self.frame_count}\n")
                f.write(f"Total distance: {self.total_distance:.2f}m\n")
                f.write(f"Start position: ({self.start_pos[0]:.2f}, {self.start_pos[1]:.2f}, {self.start_pos[2]:.2f})\n")
                f.write(f"End position: ({self.path[-1][0]:.2f}, {self.path[-1][1]:.2f}, {self.path[-1][2]:.2f})\n")
                f.write(f"Drift error: {drift_error:.2f}m\n")
            
            print("[✓] Results saved to vo_output/results.txt")
            print("[✓] Path visualization saved as trajectory_path.png")