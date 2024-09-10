import cv2
import numpy as np
import mediapipe as mp
import urllib.request
import os
from collections import deque

import firebase_admin
from firebase_admin import credentials, storage


# Initialize Firebase Admin SDK
cred = credentials.Certificate("../../credentials.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'test-421b9.appspot.com'
})

cascade_file = "cat.xml"

face_cascade = cv2.CascadeClassifier(cascade_file)

def segment_lips_and_teeth(frame):
    """
    Segment lips and teeth from a frame and make the rest of the frame transparent.
    
    :param frame: Input frame (BGR format)
    :return: Frame with only lips and teeth visible, rest is transparent
    """
    mp_face_mesh = mp.solutions.face_mesh
    
    with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame and find facial landmarks
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            
            # Get lip and mouth landmarks
            mouth_landmarks = [
                # Outer lip contour
                61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
                # Inner lip contour
                78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191,
                # Additional points for better coverage
                76, 77, 90, 180, 85, 16, 315, 404, 320, 307, 306, 184, 74, 73, 72, 11, 302, 303, 304, 408,
                # Teeth area
                62, 96, 89, 179, 86, 15, 316, 403, 319, 325, 292, 407, 272, 271, 268, 12, 38, 41, 42, 183
            ]
            
            # Extract mouth coordinates
            h, w = frame.shape[:2]
            mouth_points = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in mouth_landmarks]
            
            # Create a mask for the mouth region
            mask = np.zeros((h, w), dtype=np.uint8)
            
            # Find the outermost points to create a single polygon
            hull = cv2.convexHull(np.array(mouth_points))
            
            # Fill the polygon completely
            cv2.fillPoly(mask, [hull], 255)
            
            # Apply Gaussian blur to soften the mask edges
            mask = cv2.GaussianBlur(mask, (5, 5), 0)
            
            # Create the output frame (transparent background)
            output_frame = np.zeros((h, w, 4), dtype=np.uint8)
            
            # Apply the mask to the original frame and combine with the transparent background
            output_frame[:,:,0:3] = cv2.bitwise_and(frame, frame, mask=mask)
            output_frame[:,:,3] = mask
            
            return output_frame, hull
        else:
            # If no face is detected, return a transparent frame and None for hull
            return np.zeros((frame.shape[0], frame.shape[1], 4), dtype=np.uint8), None

def detect_cat_face(frame):
    """
    Detect the cat's face in the frame using Haar cascade classifier.
    
    :param frame: Input frame (BGR format)
    :return: List of bounding boxes of cat faces (x, y, w, h)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect cat faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    
    return faces.tolist() if len(faces) > 0 else []

def process_video(human_video_path, cat_video_path, output_path):
    """
    Process videos to segment lips and teeth from the human video,
    and overlay them on the cat's mouth in the cat video.
    
    :param human_video_path: Path to the human video file
    :param cat_video_path: Path to the cat video file
    :param output_path: Path to save the output video file
    """
    human_cap = cv2.VideoCapture(human_video_path)
    cat_cap = cv2.VideoCapture(cat_video_path)
    
    # Get video properties (assuming both videos have the same properties)
    width = int(cat_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cat_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cat_cap.get(cv2.CAP_PROP_FPS))
    
    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Initialize deque to store recent cat face detections
    recent_faces = deque(maxlen=30)  # Store last 30 frames (1 second at 30 fps)
    
    # Load the Haar cascade for cat face detection
    cascade_file = "haarcascade_frontalcatface_extended.xml"
    face_cascade = cv2.CascadeClassifier(cascade_file)

    # Pre-process cat video to detect faces
    print("Pre-processing cat video to detect faces...")
    face_detections = []
    while cat_cap.isOpened():
        ret, frame = cat_cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cat_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
        face_detections.extend(cat_faces)
    
    # Reset video capture to the beginning
    cat_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Calculate the most common cat face area
    if face_detections:
        face_detections = np.array(face_detections)
        
        # Use a histogram approach to find the most common face area
        x_hist = np.histogram(face_detections[:, 0], bins=50)
        y_hist = np.histogram(face_detections[:, 1], bins=50)
        w_hist = np.histogram(face_detections[:, 2], bins=20)
        h_hist = np.histogram(face_detections[:, 3], bins=20)
        
        common_x = x_hist[1][np.argmax(x_hist[0])]
        common_y = y_hist[1][np.argmax(y_hist[0])]
        common_w = w_hist[1][np.argmax(w_hist[0])]
        common_h = h_hist[1][np.argmax(h_hist[0])]
        
        common_face = [int(common_x), int(common_y), int(common_w), int(common_h)]
    else:
        print("No cat face detected in the video.")
        return None
    
    print("Processing video...")
    while human_cap.isOpened() and cat_cap.isOpened():
        human_ret, human_frame = human_cap.read()
        cat_ret, cat_frame = cat_cap.read()
        
        if not human_ret or not cat_ret:
            break
        
        # Process the human frame to get segmented lips
        segmented_lips, hull = segment_lips_and_teeth(human_frame)
        
        if hull is not None:
            # Detect cat faces in the current frame
            current_cat_faces = detect_cat_face(cat_frame)
            
            # If faces are detected in the current frame, use the one closest to the common face
            if current_cat_faces:
                current_cat_face = min(current_cat_faces, key=lambda f: np.linalg.norm(np.array(f) - common_face))
            else:
                current_cat_face = common_face
            
            x, y, w, h = current_cat_face
            
            # Draw bounding box around cat's face
            cv2.rectangle(cat_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Estimate cat's mouth region (lower third of the face)
            mouth_y = y + int(2*h/3)
            mouth_h = int(h/3)
            
            # Adjust the size of the lip mask to cover the cat's mouth region
            human_mouth_w = np.max(hull[:, 0, 0]) - np.min(hull[:, 0, 0])
            human_mouth_h = np.max(hull[:, 0, 1]) - np.min(hull[:, 0, 1])
            scale_x = w / human_mouth_w
            scale_y = mouth_h / human_mouth_h
            
            # Resize segmented lips to match cat's mouth size and make it 5x larger
            resized_lips = cv2.resize(segmented_lips, (w * 5, mouth_h * 5))
            
            # Calculate the position to center the enlarged lips
            start_x = max(0, x - w // 2 - resized_lips.shape[1] // 2)
            start_y = max(0, mouth_y + mouth_h // 2 - resized_lips.shape[0] // 2)
            end_x = min(cat_frame.shape[1], start_x + resized_lips.shape[1])
            end_y = min(cat_frame.shape[0], start_y + resized_lips.shape[0])
            
            # Adjust resized_lips if it goes out of frame
            resized_lips = resized_lips[:end_y-start_y, :end_x-start_x]
            
            # Create a mask for the resized lips
            mask = resized_lips[:, :, 3] / 255.0
            
            # Overlay resized lips on the cat frame
            for c in range(0, 3):
                cat_frame[start_y:end_y, start_x:end_x, c] = (1 - mask) * cat_frame[start_y:end_y, start_x:end_x, c] + \
                                                               mask * resized_lips[:, :, c]
        
        # Write the frame
        out.write(cat_frame)
    
    # Release everything
    human_cap.release()
    cat_cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Upload the output video to Firebase Storage
    bucket = storage.bucket()
    blob = bucket.blob(f'processed_videos/{os.path.basename(output_path)}')
    blob.upload_from_filename(output_path)

    # Make the blob publicly accessible
    blob.make_public()

    # Get the public URL
    output_url = blob.public_url

    # Remove the local output video file
    os.remove(output_path)

    return output_url

# Example usage:
#cat_video_url = 'https://drive.google.com/uc?export=download&id=1MW_zsdYGZP_yIE4Wrh3BSPc832gQ6o8q'
cat_video_url = 'https://drive.google.com/uc?export=download&id=1Nc7NxkaCPNR0ZaIkUTenD9OXIlwx7ct8'
human_video_url = 'https://drive.google.com/uc?export=download&id=1zPNI_dwRa53NfniDhQc3sGa0YYaY-kQj'
cat_video_path = 'temp_cat_video.mp4'
human_video_path = 'temp_human_video.mp4'
output_video = 'output_video_cat_human.mp4'

# Download videos
urllib.request.urlretrieve(cat_video_url, cat_video_path)
urllib.request.urlretrieve(human_video_url, human_video_path)

# Process videos
output_url = process_video(human_video_path, cat_video_path, output_video)

# Clean up downloaded videos
os.remove(cat_video_path)
os.remove(human_video_path)

print(f"Processed video URL: {output_url}")
