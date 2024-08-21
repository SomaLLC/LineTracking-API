import cv2
import os
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("../models/yolov9m.pt") 

# Ask user for video path
video_path = input("Enter the path to the video file: ")

# Create an output directory to save the images
output_dir = "output_frames"
os.makedirs(output_dir, exist_ok=True)

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if the video file was opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

frame_count = 0

# Process the video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run YOLO model on the frame
    results = model(frame)
    
    # Draw all detections
    for result in results:
        for obj in result.boxes.data:
            label = result.names[int(obj.cls)]
            bbox = obj.xyxy
            conf = obj.conf
            # Draw bounding box and label
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Save the frame as an image
    output_path = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
    cv2.imwrite(output_path, frame)
    
    frame_count += 1

# Release the video capture
cap.release()

print(f"Processed {frame_count} frames. Images with detections are saved in '{output_dir}'.")
