"""
IntelliCount: AI Object Detection for Vehicle Tracking and Traffic Insights
---------------------------------------------------------------------------
A capstone project for real-time vehicle detection, tracking, and analysis using YOLOv8.

This script provides a comprehensive solution for traffic analysis by:
1. Detecting and classifying vehicles (cars, trucks, buses, motorcycles)
2. Tracking vehicles across frames with unique IDs
3. Generating real-time statistics and visualizations
4. Creating PDF and JSON reports for further analysis

The system is optimized for performance on CPU systems and provides an interactive
dashboard for monitoring traffic patterns.
"""

import os
import time
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
import argparse  # For command-line arguments
import torch  # Add explicit torch import

def get_center(x1, y1, x2, y2):
    """Gets the center point of a bounding box"""
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))

def is_new_detection(center, previous_centers, min_distance=30):
    """Checks if this detection is new or if we've seen it before"""
    for prev_center in previous_centers:
        distance = np.sqrt((center[0] - prev_center[0])**2 + (center[1] - prev_center[1])**2)
        if distance < min_distance:
            return False
    return True

def draw_bounding_box(frame, x1, y1, x2, y2, label, confidence, track_id=None, speed=None):
    """Draws box around detected vehicles with labels and confidence"""
    # Colors for different vehicle types (BGR format)
    colors = {
        'car': (0, 255, 0),       # Green
        'truck': (0, 0, 255),     # Red
        'bus': (255, 0, 0),       # Blue
        'motorcycle': (255, 255, 0),  # Cyan
        'bicycle': (255, 0, 255)  # Magenta
    }
    
    color = colors.get(label.lower(), (200, 200, 200))  # Gray for anything else
    
    # Draw rectangle
    thickness = max(2, int((x2-x1)/100))  # Thicker lines for bigger objects
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    
    # Create label text with ID
    text = f"{label}"
    if track_id is not None:
        text += f" #{track_id}"
    # No speed - wasn't accurate enough
    text += f": {confidence:.2f}"
    
    # Draw filled background for text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8  # Bigger text is easier to read
    text_size = cv2.getTextSize(text, font, font_scale, 2)[0]
    
    # Text background
    cv2.rectangle(frame, (x1, y1 - text_size[1] - 5), 
                 (x1 + text_size[0], y1), color, -1)
    
    # Text
    cv2.putText(frame, text, (x1, y1 - 5), font, font_scale, (0, 0, 0), 2)
    
    return frame

def create_dashboard(frame_width, current_stats, total_stats, fps):
    """Creates a dashboard showing stats about detected vehicles"""
    # Dashboard dimensions
    dashboard_height = 200  # Tall enough for all our info
    dashboard = np.zeros((dashboard_height, frame_width, 3), dtype=np.uint8)
    dashboard[:, :] = (40, 40, 40)  # Dark gray background
    
    # Add title and separator line
    cv2.putText(dashboard, "Traffic Analysis Dashboard", (20, 35),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.line(dashboard, (0, 50), (frame_width, 50), (100, 100, 100), 1)
    
    # Add FPS counter
    cv2.putText(dashboard, f"FPS: {fps:.1f}", (frame_width - 170, 35),
               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    
    # Colors for vehicle types
    type_colors = {
        'car': (0, 255, 0),       # Green
        'truck': (0, 0, 255),     # Red
        'bus': (255, 0, 0),       # Blue
        'motorcycle': (255, 255, 0),  # Cyan
        'bicycle': (255, 0, 255)  # Magenta
    }
    
    # Left panel: Current frame stats
    cv2.putText(dashboard, "Current Frame", (20, 80),
               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220, 220, 220), 2)
    
    y_pos = 115  # Starting position for text
    x_pos = 40
    total_current = sum(current_stats.values())
    cv2.putText(dashboard, f"Active Vehicles: {total_current}", (x_pos, y_pos),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    
    y_pos = 150  # Next line of text
    for vehicle_type, count in sorted(current_stats.items()):
        if count > 0:
            color = type_colors.get(vehicle_type, (200, 200, 200))
            cv2.putText(dashboard, f"{vehicle_type.capitalize()}: {count}", (x_pos, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            x_pos += 220  # Space between items
            if x_pos > frame_width // 2 - 50:  # Start new row if needed
                x_pos = 40
                y_pos += 35  # Line spacing
    
    # Right panel: Total stats
    cv2.putText(dashboard, "Total Vehicles Detected", (frame_width // 2 + 20, 80),
               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220, 220, 220), 2)
    
    y_pos = 115  # Starting position
    x_pos = frame_width // 2 + 40
    total_count = sum(total_stats.values())
    cv2.putText(dashboard, f"Total Count: {total_count}", (x_pos, y_pos),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    
    y_pos = 150  # Next line
    for vehicle_type, count in sorted(total_stats.items()):
        if count > 0:
            color = type_colors.get(vehicle_type, (200, 200, 200))
            cv2.putText(dashboard, f"{vehicle_type.capitalize()}: {count}", (x_pos, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            x_pos += 220  # Space between items
            if x_pos > frame_width - 50:  # Move to next line if needed
                x_pos = frame_width // 2 + 40
                y_pos += 35  # Line spacing
                
    # Add divider between left and right panels
    cv2.line(dashboard, (frame_width // 2 - 10, 55), (frame_width // 2 - 10, dashboard_height - 10), (100, 100, 100), 1)
    
    return dashboard

def add_top_left_overlay(frame, current_stats, total_stats, fps, model_name="yolov8m.pt", device="cpu"):
    """
    Adds a cool overlay in the corner with live stats
    """
    # Calculate total vehicles in current frame
    vehicles_in_frame = sum(current_stats.values())
    total_vehicles = sum(total_stats.values())
    
    # Create transparent dark background
    overlay = frame.copy()
    x_offset, y_offset = 10, 10
    width, height = 380, 220  # Size of our overlay box
    cv2.rectangle(overlay, (x_offset, y_offset), (x_offset + width, y_offset + height), 
                 (0, 0, 0), -1)
    
    # Apply transparency
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    # Add text with yellow color for better visibility
    text_color = (0, 255, 255)  # Yellow
    font = cv2.FONT_HERSHEY_SIMPLEX
    line_height = 30  # Space between lines
    font_size = 0.8  # Good readable size
    
    # Add title to the overlay
    cv2.putText(frame, "Traffic Analysis", 
               (x_offset + 10, y_offset + 25), font, 0.9, text_color, 2)
    
    # Display data
    cv2.putText(frame, f"Vehicles in frame: {vehicles_in_frame}", 
               (x_offset + 10, y_offset + line_height + 25), font, font_size, text_color, 2)
    
    cv2.putText(frame, f"Total vehicles: {total_vehicles}", 
               (x_offset + 10, y_offset + 2*line_height + 25), font, font_size, text_color, 2)
    
    # Add vehicle type counts for the current frame
    y_pos = y_offset + 3*line_height + 25
    for vehicle_type, count in current_stats.items():
        if count > 0:
            cv2.putText(frame, f"{vehicle_type}: {count}", 
                       (x_offset + 10, y_pos), font, font_size, text_color, 2)
            y_pos += line_height
    
    # Add technical info
    cv2.putText(frame, f"Model: {model_name}", 
               (x_offset + 10, y_offset + 8*line_height + 5), font, font_size, text_color, 2)
    
    cv2.putText(frame, f"Device: {device}", 
               (x_offset + 10, y_offset + 9*line_height + 5), font, font_size, text_color, 2)
    
    # Calculate average confidence of detections in the current frame
    if hasattr(main, 'confidence_values') and main.confidence_values:
        avg_conf = sum(main.confidence_values) / len(main.confidence_values)
        cv2.putText(frame, f"Avg Confidence: {avg_conf:.2f}", 
                   (x_offset + 10, y_offset + 10*line_height + 5), font, font_size, text_color, 2)
    
    # Add detection rate (FPS)
    cv2.putText(frame, f"Detection Rate: {fps:.2f} FPS", 
               (x_offset + 10, y_offset + 11*line_height + 5), font, font_size, text_color, 2)
    
    return frame

def generate_json_report(total_vehicle_counts, processing_time, fps, model_name, device, output_dir="outputs"):
    """
    Generates a JSON report for later analysis or API use
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(output_dir, f"traffic_report_{timestamp}.json")
    
    # Create report data
    report_data = {
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "device": device,
        "processing_stats": {
            "total_processing_time_seconds": processing_time,
            "average_fps": fps
        },
        "vehicle_counts": {
            "total": sum(total_vehicle_counts.values()),
            "by_type": total_vehicle_counts
        }
    }
    
    # Write JSON file
    with open(json_path, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print(f"JSON report saved to: {json_path}")
    return json_path

def generate_pdf_report(total_vehicle_counts, processing_time, fps, model_name, device, output_dir="outputs"):
    """
    Creates a nice PDF report with all the detection data
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = os.path.join(output_dir, f"traffic_report_{timestamp}.pdf")
    
    # Setup PDF
    with PdfPages(pdf_path) as pdf:
        # Create the report page
        plt.figure(figsize=(8.5, 11))
        plt.suptitle("Report Generator", fontsize=16, y=0.98)
        
        # Add traffic analysis report header
        plt.text(0.5, 0.9, "Traffic Analysis Report", fontsize=14, ha='center')
        
        # Add report details with good spacing
        y_position = 0.82  # Starting position
        
        # Model info
        plt.text(0.1, y_position, f"Model: {model_name}", fontsize=12)
        y_position -= 0.06  # Move down
        
        # Device info
        plt.text(0.1, y_position, f"Device: {device}", fontsize=12)
        y_position -= 0.06  # Move down
        
        # Processing time
        plt.text(0.1, y_position, f"Total Processing Time: {processing_time:.2f} seconds", fontsize=12)
        y_position -= 0.06  # Move down
        
        # FPS info
        plt.text(0.1, y_position, f"Average FPS: {fps:.2f}", fontsize=12)
        y_position -= 0.06  # Move down
        
        # Total vehicles
        plt.text(0.1, y_position, f"Total Vehicles Detected: {sum(total_vehicle_counts.values())}", fontsize=12)
        
        # Add vehicle counts section
        y_position -= 0.08  # Extra space before the section
        plt.text(0.1, y_position, "Vehicle Counts by Type:", fontsize=12)
        
        # Add each vehicle type
        for vehicle_type, count in sorted(total_vehicle_counts.items()):
            y_position -= 0.06  # Space between items
            plt.text(0.1, y_position, f"- {vehicle_type.capitalize()}: {count}", fontsize=12)
        
        # Remove axis
        plt.axis('off')
        pdf.savefig()
        plt.close()
    
    print(f"PDF report saved to: {pdf_path}")
    return pdf_path

def parse_arguments():
    """Sets up the command line options you can use"""
    parser = argparse.ArgumentParser(description="Vehicle Detection and Tracking System")
    
    # Input/output options
    parser.add_argument("--video", type=str, default="8698-213454544.mp4",
                       help="Path to input video file")
    parser.add_argument("--output-dir", type=str, default="outputs",
                       help="Directory to save output files")
    parser.add_argument("--output-name", type=str, default=None,
                       help="Custom name for output files (default: derived from input)")
    
    # Model options
    parser.add_argument("--model", type=str, default="yolov8m.pt",
                       help="Path to YOLOv8 model")
    parser.add_argument("--conf", type=float, default=0.45,
                       help="Confidence threshold for detection")
    parser.add_argument("--iou", type=float, default=0.45,
                       help="IoU threshold for NMS")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                       help="Device to run inference on (default: cpu)")
    
    # Display options
    parser.add_argument("--no-display", action="store_true",
                       help="Disable display window while processing")
    parser.add_argument("--window-size", type=str, default="1280,720",
                       help="Window size for display (width,height)")
    
    # Tracking options
    parser.add_argument("--tracker-distance", type=int, default=50,
                       help="Distance threshold for tracking")
    parser.add_argument("--track-timeout", type=float, default=2.0,
                       help="Time in seconds before removing inactive tracks")
    
    # Report options
    parser.add_argument("--report", action="store_true", default=True,
                       help="Generate reports after processing")
    
    # Parse arguments
    return parser.parse_args()

def main():
    """
    Main function that coordinates vehicle detection and tracking.
    
    This function:
    1. Initializes the system and parses command-line arguments
    2. Sets up video input/output and display
    3. Performs detection and tracking on each frame
    4. Creates visualizations and dashboards
    5. Generates summary reports
    
    Returns:
        None
    """
    # Force PyTorch to use CPU
    torch.cuda.is_available = lambda : False
    
    # Parse command-line arguments
    args = parse_arguments()
    
    # Extract arguments
    video_path = args.video
    model_name = args.model
    confidence_threshold = args.conf
    iou_threshold = args.iou
    output_dir = args.output_dir
    display_enabled = not args.no_display  # Display is now enabled by default
    generate_reports = args.report
    tracker_distance = args.tracker_distance
    track_timeout = args.track_timeout
    
    # Create outputs folder if it doesn't exist
    outputs_dir = output_dir
    os.makedirs(outputs_dir, exist_ok=True)
    
    # Check if video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return
    
    print(f"Using video file: {video_path}")
    
    # Load YOLO model with explicit CPU device
    try:
        model = YOLO(model_name)
        # Force model to CPU
        model.to('cpu')
        print(f"YOLO model {model_name} loaded successfully")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return
    
    # Always use CPU
    device = "cpu"
    print(f"Using device: {device}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video properties: {frame_width}x{frame_height} at {fps} FPS")
    
    # Create output video writer
    video_name = args.output_name if args.output_name else os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(outputs_dir, f"output_{video_name}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    combined_height = frame_height + 200  # Add height for dashboard
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, combined_height))
    
    # Create window if display is enabled
    window_name = "Vehicle Detection"
    if display_enabled:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        # Instead of fullscreen, maximize the window while keeping controls visible
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        # Get screen resolution
        try:
            import ctypes
            user32 = ctypes.windll.user32
            screen_width, screen_height = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
            # Resize to almost full screen but keep window controls
            cv2.resizeWindow(window_name, screen_width - 100, screen_height - 100)
        except:
            # Fallback to specified window size if getting screen resolution fails
            window_width, window_height = map(int, args.window_size.split(','))
            cv2.resizeWindow(window_name, window_width, window_height)
    
    # Initialize tracking variables
    previous_centers = []
    total_vehicle_counts = {"car": 0, "truck": 0, "bus": 0, "motorcycle": 0}
    current_frame_vehicles = {"car": 0, "truck": 0, "bus": 0, "motorcycle": 0}
    
    # Add list for tracking confidence values
    main.confidence_values = []
    
    # FPS calculation
    frame_count = 0
    start_time = time.time()
    current_fps = 0
    
    # Vehicle tracking with simple IDs
    next_id = 1
    tracked_vehicles = {}  # {id: {center, type, last_seen}}
    
    print("Processing video - press 'q' to quit")
                
            # Process frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video")
            break
        
        frame_count += 1
        
        # Calculate FPS
        current_time = time.time()
        elapsed_time = current_time - start_time
        if elapsed_time > 1.0:
            current_fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()
            
        # Reset current frame counts and confidence values
        for vehicle_type in current_frame_vehicles:
            current_frame_vehicles[vehicle_type] = 0
        main.confidence_values = []
            
        # Run detection with explicit CPU device
        # COCO classes: 2=car, 3=motorcycle, 5=bus, 7=truck - just looking for vehicles
        results = model(frame, classes=[2, 3, 5, 7], conf=confidence_threshold, iou=iou_threshold, device='cpu')
        
        # Current frame's centers
        current_centers = []
        
        # Process detections
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get detection info
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                
                # Store confidence value for average calculation
                main.confidence_values.append(confidence)
                
                # Calculate center
                center = get_center(x1, y1, x2, y2)
                current_centers.append(center)
                
                # Track this vehicle
                matched_id = None
                min_distance = tracker_distance  # Distance threshold for matching
                
                # Try to match with existing tracks
                for vehicle_id, vehicle_data in tracked_vehicles.items():
                    if current_time - vehicle_data['last_seen'] < track_timeout:  # If track is still active
                        dist = np.sqrt((center[0] - vehicle_data['center'][0])**2 + 
                                      (center[1] - vehicle_data['center'][1])**2)
                        if dist < min_distance:
                            min_distance = dist
                            matched_id = vehicle_id
                
                if matched_id is not None:
                    # Update existing track
                    tracked_vehicles[matched_id]['center'] = center
                    tracked_vehicles[matched_id]['last_seen'] = current_time
                    tracked_vehicles[matched_id]['bbox'] = (x1, y1, x2, y2)
                    tracked_vehicles[matched_id]['confidence'] = max(confidence, tracked_vehicles[matched_id]['confidence'])
                    
                    # Use the detected ID
                    vehicle_id = matched_id
                else:
                    # Check if this is a new detection based on previous centers
                    if is_new_detection(center, previous_centers):
                        # This is a new vehicle - count it
                        if class_name.lower() in total_vehicle_counts:
                            total_vehicle_counts[class_name.lower()] += 1
                    
                    # Create new track
                    tracked_vehicles[next_id] = {
                        'center': center,
                        'type': class_name.lower(),
                        'last_seen': current_time,
                        'first_seen': current_time,
                        'bbox': (x1, y1, x2, y2),
                        'confidence': confidence,
                    }
                    vehicle_id = next_id
                    next_id += 1
                
                # Increment current frame vehicle count
                if class_name.lower() in current_frame_vehicles:
                    current_frame_vehicles[class_name.lower()] += 1
                
                # Draw bounding box with ID
                frame = draw_bounding_box(frame, x1, y1, x2, y2, class_name, confidence, vehicle_id)
        
        # Update previous centers
        previous_centers = current_centers.copy()
        
        # Remove old tracks
        old_ids = []
        for vehicle_id, vehicle_data in tracked_vehicles.items():
            if current_time - vehicle_data['last_seen'] > track_timeout:  # If track is too old
                old_ids.append(vehicle_id)
        
        for vehicle_id in old_ids:
            tracked_vehicles.pop(vehicle_id)
        
        # Add top-left overlay
        frame = add_top_left_overlay(frame, current_frame_vehicles, total_vehicle_counts, 
                                    current_fps, model_name, device)
            
        # Create bottom dashboard
        dashboard = create_dashboard(frame_width, current_frame_vehicles, total_vehicle_counts, current_fps)
        
        # Combine frame with dashboard
        combined_frame = np.vstack((frame, dashboard))
        
        # Write frame to output video
        out.write(combined_frame)
        
        # Display frame if enabled
        if display_enabled:
            cv2.imshow(window_name, combined_frame)
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("User terminated video processing")
                break
    
    # Cleanup
    cap.release()
    out.release()
    if display_enabled:
        cv2.destroyAllWindows()
    
    # Generate reports at the end of processing
    if generate_reports:
        total_processing_time = time.time() - start_time
        avg_fps = frame_count / max(total_processing_time, 0.001)
        
        # Generate JSON and PDF reports
        json_path = generate_json_report(
            total_vehicle_counts,
            total_processing_time,
            avg_fps,
            model_name,
            device,
            outputs_dir
        )
        
        pdf_path = generate_pdf_report(
            total_vehicle_counts,
            total_processing_time,
            avg_fps,
            model_name,
            device,
            outputs_dir
        )
        
        print(f"Reports generated: {json_path}, {pdf_path}")
    
    print("Vehicle detection complete")
    print("Vehicle counts:", total_vehicle_counts)
    print(f"Total vehicles detected: {sum(total_vehicle_counts.values())}")
    print(f"Output video saved to: {output_path}")
        
if __name__ == "__main__":
    main()