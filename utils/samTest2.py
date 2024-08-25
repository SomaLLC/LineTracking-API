import firebase_admin
from firebase_admin import credentials, storage
import cv2
import os
import random
from ultralytics import SAM
import numpy as np
import torch
import torchvision.transforms as transforms

# Initialize Firebase Admin SDK
cred = credentials.Certificate("../../credentials.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'test-421b9.appspot.com'
})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the SAM model
model = SAM("../models/sam2_t.pt").to(device)

# Ask user for video path
#video_path = input("Enter the path to the video file: ")

#Hardcode video path for easy debugging
#video_path = "../content/cloth-sample.mp4"
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
output_video_path = "output_video_sam.avi"
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Randomly skip frames to start processing at a random point
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
start_frame = random.randint(0, total_frames // 2)
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

# Process the video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # You can define a bounding box or points for segmentation as needed
    height, width, _ = frame.shape
    bbox = [width // 4, height // 4, 3 * width // 4, 3 * height // 4]  # Example bounding box

    # Run SAM model on the frame with bounding box prompt
    results = model(frame, bboxes=[bbox])

    #results = results.cpu()

    # Apply the mask to the frame (assuming the model returns a mask)
    for result in results:
        if result.masks is not None:
            for mask in result.masks:
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
print(f"Video uploaded to Firebase Storage. Public URL: {firebase_url}")
