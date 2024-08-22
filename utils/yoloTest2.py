import firebase_admin
from firebase_admin import credentials, storage
import cv2
import os
from ultralytics import YOLO

# Initialize Firebase Admin SDK
cred = credentials.Certificate("../../credentials.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'test-421b9.appspot.com'
})

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
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create a VideoWriter object to save the video
output_video_path = "output_video.avi"
fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Using H.264 codec for better compatibility
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Process the video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run YOLO model on the frame
    results = model(frame)
    
    # Draw all detections
    for result in results:
        for obj,label,bbox,confidence in zip(result.boxes.data, result.boxes.cls, result.boxes.xyxy, result.boxes.conf):
            print(f"\n\n\n Detected {label} with confidence {confidence:.2f} at bbox {bbox}")

            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                label_text = f"{label} {confidence:.2f} ({int(bbox[0])}, {int(bbox[1])}, {int(bbox[2])}, {int(bbox[3])})"
                cv2.putText(frame, label_text, (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
 
    # Save the frame as an image
    output_path = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
    cv2.imwrite(output_path, frame)
    
    # Write the frame to the video file
    out.write(frame)
    
    frame_count += 1

# Release the video capture and writer
cap.release()
out.release()

print(f"Processed {frame_count} frames. Video saved as '{output_video_path}'.")

# Upload the video to Firebase Storage
bucket = storage.bucket()
blob = bucket.blob(output_video_path)
blob.upload_from_filename(output_video_path)

# Make the file public
blob.make_public()

# Get the public URL
firebase_url = blob.public_url
print(f"Video uploaded to Firebase Storage. Public URL: {firebase_url}")

