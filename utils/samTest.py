import firebase_admin
from firebase_admin import credentials, storage
import cv2
import os
import random
from ultralytics import SAM

# Initialize Firebase Admin SDK
cred = credentials.Certificate("../../credentials.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'test-421b9.appspot.com'
})

# Load the SAM model
model = SAM("../models/sam2_t.pt")

# Ask user for video path
#video_path = input("Enter the path to the video file: ")

#Hardcode video path for easy debugging
video_path = "../content/cloth-sample.mp4"

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

    print(f"Results: {results}")

    # Apply the mask to the frame (assuming the model returns a mask)
    for result in results:
        if result.masks is not None:
            for mask in results.masks:
                mask_array = mask.data.cpu().numpy()  # Convert mask tensor to numpy array
                mask_array = cv2.resize(mask_array[0], (width, height))  # Resize mask to frame size
                
                # Apply the mask to each channel (R, G, B)
                for i in range(3):
                    frame[:, :, i] = frame[:, :, i] * mask_array
                
                # Optional: Apply a color to the masked area
                frame[mask_array > 0] = [0, 255, 0]  # Apply a green color to the masked areas

    # Save the masked frame as an image
    output_path = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
    cv2.imwrite(output_path, frame)
    
    # Write the masked frame to the video file
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
