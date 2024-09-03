import firebase_admin
from firebase_admin import credentials, storage
import cv2
import os
import random
from ultralytics import SAM, YOLO
import numpy as np
import torch

# Initialize Firebase Admin SDK
cred = credentials.Certificate("../../credentials.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'test-421b9.appspot.com'
})

torch.cuda.set_device(0)

# Load the YOLO and SAM models
yolo_model = YOLO("../models/yolov9m.pt")
sam_model = SAM("../models/sam2_t.pt")

# Hardcode video path for easy debugging
video_path = "https://drive.google.com/uc?export=download&id=1LY5zikXCmg8OPRAhCuBGagcfh4f5Ns_Z"

# Create an output directory to save the images
output_dir = "output_frames"
os.makedirs(output_dir, exist_ok=True)

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if the video file was opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

frame_count = 0
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create a VideoWriter object to save the video
output_video_path = "output_video_combined.avi"
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Randomly skip frames to start processing at a random point
start_frame = random.randint(0, total_frames // 2)
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

# Process the video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO model on the frame
    yolo_results = yolo_model(frame)

    # Process each detected box with SAM model
    for result in yolo_results:
        for obj, label, bbox, confidence in zip(result.boxes.data, result.boxes.cls, result.boxes.xyxy, result.boxes.conf):
            print(f"Detected {label} with confidence {confidence:.2f} at bbox {bbox}")

            # Crop the detected box from the frame
            x1, y1, x2, y2 = map(int, bbox)
            cropped_image = frame[y1:y2, x1:x2]

            # Run SAM model on the cropped image
            sam_results = sam_model(cropped_image)

            # Apply the mask to the frame (assuming the model returns a mask)
            for sam_result in sam_results:
                if sam_result.masks is not None:
                    for mask in sam_result.masks:
                        coords = mask.xy  # Get the mask coordinates (individual points)

                        # Create an empty mask
                        mask_array = np.zeros((frame_height, frame_width), dtype=np.uint8)

                        # Draw the points on the mask
                        for point in coords[0]:  # Assuming coords[0] is a numpy array of points
                            x, y = int(point[0]), int(point[1])
                            if 0 <= x < frame_width and 0 <= y < frame_height:
                                mask_array[y, x] = 255  # Set the point in the mask

                        # Apply the mask to the frame
                        frame[mask_array > 0] = [0, 255, 0]  # Set the masked areas to green

    # Save the masked frame as an image
    output_path = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
    cv2.imwrite(output_path, frame)

    # Write the masked frame to the video file
    out.write(frame)

    frame_count += 1

    percentage_processed = (frame_count / total_frames) * 100
    print(f"Processed {percentage_processed:.2f}% of frames.")

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
video_format = ""
video_size = ""
video_encoding = ""
try:
    video_format = blob.format
    video_size = blob.size
    video_encoding = blob.encoding
except:
    pass

print(f"Video uploaded to Firebase Storage. Public URL: {firebase_url} ; Format: {video_format} ; Size: {video_size} ; Encoding: {video_encoding}")

